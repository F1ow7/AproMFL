import copy
import operator

import torch.optim
import torch.utils.data
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
from sklearn.cluster import KMeans
import numpy as np
import os
import random
import torch.multiprocessing
from src.test import ImageTextRetrievalTester,compute_flickr30k_recall
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn

from src.base import EngineBase
from tqdm import tqdm
import torch
from criterions.protoloss import PrototypeLoss
try:
    from apex import amp
except ImportError:
    print('failed to import apex')

from src.utils.serialize_utils import flatten_dict


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


##################################################
# step -1: Predefined function
##################################################
import torch.utils.data.sampler as sampler


class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


gpuid = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# TODO: test
is_test = False


class MMClientTrainer(EngineBase):

    def run(self, agg_proto, agg_model, img_enc, txt_enc):
        # self.old_model = copy.deepcopy(self.model)
        # self.old_model.eval().cuda()
        self.model.cuda()
        self.cluster_model.cuda()

        if self.local_epoch == 0:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O2')
            self.cluster_model, self.c_optimizer = amp.initialize(self.cluster_model, self.c_optimizer, 
                                                                  opt_level='O2')
        # 训练得到本地原型
        self.cluster_model.train()
        for i in range(self.local_epochs):
            if self.logger is not None:
                self.logger.log(f"MM Client Local Proto Training: Epoch {i}")
            
            for idx, (images, captions, _, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                # captions = captions.to(self.device)

                output = self.cluster_model(img_enc(images), txt_enc(captions))
            
                loss_o= self.criterion(output['image_output'],output['caption_output'])
                fusion = (output['image_output']+output['caption_output'])/2
                # 使用KMeans对fusion进行聚类
                num_clusters = self.config.train_proto.cluster_num                                                                                                                                                                                  
                kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(fusion.cpu().detach().numpy())
                cluster_assignments = torch.tensor(kmeans.labels_).to(self.device)  # 聚类标签
        
                # 计算聚类质心
                centroids = torch.tensor(kmeans.cluster_centers_).to(self.device)  # 聚类质心
                    # Intra-modal loss calculation for new clusters
                # intra_loss = 0
                # for i in range(fusion.size(0)):
                #     current_cluster = cluster_assignments[i]
                #     same_cluster_indices = (cluster_assignments == current_cluster).nonzero(as_tuple=True)[0]
                #     different_cluster_indices = (cluster_assignments != current_cluster).nonzero(as_tuple=True)[0]

                #     # Positive samples: same cluster, same modality
                #     positive_image_elements = output['image_output'][same_cluster_indices]
                #     positive_caption_elements = output['caption_output'][same_cluster_indices]
                #     current_image = output['image_output'][i].unsqueeze(0)
                #     current_caption = output['caption_output'][i].unsqueeze(0)

                #     # Negative samples: different clusters, same modality
                #     negative_image_elements = output['image_output'][different_cluster_indices]
                #     negative_caption_elements = output['caption_output'][different_cluster_indices]

                #     # Calculate similarity for positive and negative samples (image modality)
                #     positive_similarity_image = torch.exp(torch.mm(current_image, positive_image_elements.t()) / self.config.temperature)
                #     negative_similarity_image = torch.exp(torch.mm(current_image, negative_image_elements.t()) / self.config.temperature)
                #     intra_loss -= torch.mean(torch.log(positive_similarity_image / (positive_similarity_image + negative_similarity_image)))

                #     # Calculate similarity for positive and negative samples (caption modality)
                #     positive_similarity_caption = torch.exp(torch.mm(current_caption, positive_caption_elements.t()) / self.config.temperature)
                #     negative_similarity_caption = torch.exp(torch.mm(current_caption, negative_caption_elements.t()) / self.config.temperature)
                #     intra_loss -= torch.mean(torch.log(positive_similarity_caption / (positive_similarity_caption + negative_similarity_caption)))


                # 计算图像和文本的聚类损失
                # loss_i: 使图像嵌入靠近所属簇的质心，远离其他质心
                # loss_t: 使文本嵌入靠近所属簇的质心，远离其他质心
                # loss_i = 0.0
                # loss_t = 0.0
                loss_i = 0
                loss_t = 0
                for j in range(num_clusters):
                    cluster_indices = (cluster_assignments == j)
                    if cluster_indices.sum() == 0:
                        continue  # 该簇中没有数据
            
                    # 当前簇的质心
                    current_centroid = centroids[j]
            
                    # 图像嵌入的对比损失
                    image_embeddings = output['image_output'][cluster_indices]
            
                    # 正样本：与当前质心的相似度
                    positive_sim = F.cosine_similarity(image_embeddings, current_centroid.expand_as(image_embeddings))
            
                    # 负样本：与其他簇质心的相似度
                    negative_sims = []
                    for k in range(num_clusters):
                        if k != j:
                            negative_sim = F.cosine_similarity(image_embeddings, centroids[k].expand_as(image_embeddings))
                            negative_sims.append(negative_sim)
                    negative_sims = torch.stack(negative_sims, dim=1)  # [batch_size, num_negatives]

                    # 对比损失（基于InfoNCE或类似对比损失）
                    logits_i = torch.cat([positive_sim.unsqueeze(1), negative_sims], dim=1)  # [batch_size, 1 + num_negatives]
                    logits_i /= self.config.train_proto.temperature
                    labels_i = torch.zeros(logits_i.shape[0], dtype=torch.long).to(self.device)  # 0为正样本标签
                    loss_i += F.cross_entropy(logits_i, labels_i)
            
                    # 文本嵌入的对比损失
                    caption_embeddings = output['caption_output'][cluster_indices]
            
                    # 正样本：与当前质心的相似度
                    positive_sim_t = F.cosine_similarity(caption_embeddings, current_centroid.expand_as(caption_embeddings))
            
                    # 负样本：与其他簇质心的相似度
                    negative_sims_t = []
                    for k in range(num_clusters):
                        if k != j:
                            negative_sim_t = F.cosine_similarity(caption_embeddings, centroids[k].expand_as(caption_embeddings))
                            negative_sims_t.append(negative_sim_t)
                    negative_sims_t = torch.stack(negative_sims_t, dim=1)  # [batch_size, num_negatives]
            
                    # 对比损失（文本嵌入）
                    logits_t = torch.cat([positive_sim_t.unsqueeze(1), negative_sims_t], dim=1)  # [batch_size, 1 + num_negatives]
                    logits_t /= self.config.train_proto.temperature
                    labels_t = torch.zeros(logits_t.shape[0], dtype=torch.long).to(self.device)  # 0为正样本标签
                    loss_t += F.cross_entropy(logits_t, labels_t)
        
                # 平均化损失
                loss_i = loss_i / num_clusters
                loss_t = loss_t / num_clusters
        
                # 总损失
                total_loss = loss_o + loss_i + loss_t
        
                # 反向传播和优化
                self.c_optimizer.zero_grad()
                total_loss.backward()
                self.c_optimizer.step()
        
        # 原型学习完成，获取本地原型
        prototypes = {'image':[],'text':[]}
        for j in range(num_clusters):
            cluster_indices = (cluster_assignments == j)
            if cluster_indices.sum() == 0:
                continue  # 该簇中没有数据
            
            # 分别计算图像和文本嵌入的均值，得到原型
            image_prototype = output['image_output'][cluster_indices].mean(dim=0)
            caption_prototype = output['caption_output'][cluster_indices].mean(dim=0)
            
            # 将文本和图像的原型对存储
            prototypes['image'].append(image_prototype)
            prototypes['text'].append(caption_prototype)
        prototypes['image'] = torch.stack(prototypes['image'])
        prototypes['text'] = torch.stack(prototypes['text'])
                # if agg_proto is not None:
                #     loss_p = PrototypeLoss(agg_proto)
                #     l_p = loss_p(output['image_output'], output['image_output'])
                #     loss = loss_o + l_p
                # else:
                #     loss = loss_o
            

            # if self.config.train.get('use_fp16'):
            #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #             scaled_loss.backward()
            # else:
            #     loss.backward()

            # if self.config.train.grad_clip > 0:
            #     nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
            #                                        self.config.train.grad_clip)
        
        # 更新本地模型
        self.model.image_ph.fc.load_state_dict(agg_model['image'].fc.state_dict())
        self.model.text_ph.fc.load_state_dict(agg_model['text'].fc.state_dict())
        self.model.train()
        
        for i in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"MM Client Local Training: Epoch {self.local_epoch}")
            self.train_epoch(agg_proto,img_enc,txt_enc)

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/Flicker30K/Client{self.client}-model_{self.local_epoch}.pth')

        # test = ImageTextRetrievalTester(self.model, self.val_loader,self.device, img_enc, txt_enc)
        # test.run_tests()
        # self.old_model.cpu()
        # self.model.cpu()
        self.test(txt_enc=txt_enc,img_enc=img_enc)
        # del self.old_model
        import gc
        gc.collect()
        return {'image':self.model.image_ph.fc.state_dict(),'text':self.model.text_ph.fc.state_dict()}, prototypes
    
    def compute_similarity_loss(self):
        """
        Compute a loss that encourages the parameters of self.model to be close to the parameters of self.cluster_model.

            :return: Similarity loss between self.model and self.cluster_model
        """
        # Initialize the total loss
        total_loss = 0.0

        # Iterate through the parameters of both models
        for param_model, param_cluster_model in zip(self.model.parameters(), self.cluster_model.parameters()):
            # Ensure both parameters have the same shape
            if param_model.shape == param_cluster_model.shape:
                # Compute the L2 loss between the parameters
                total_loss += F.mse_loss(param_model, param_cluster_model)

        return total_loss
    
    def test(self, img_enc, txt_enc):
        image_features = []
        text_features = []
        
        self.model.eval()
        with torch.no_grad():
            for idx, (images, captions, ind, _) in enumerate(self.val_loader):
                images = images.to(self.device)
                output = self.model(img_enc(images), txt_enc(captions))
                image_features.append(output['image_output'])
                text_features.append(output['caption_output'])
        # Concatenate all embeddings for current client
        texts_emb = torch.cat(text_features, dim=0)
        images_emb = torch.cat(image_features, dim=0)
                
        # Compute recall metrics for the current client
        client_metrics = compute_flickr30k_recall(texts_emb, images_emb, self.test_same_img, recall_k_list=[1, 5, 10], device = self.device)
            

        self.logger.log(f"Test Client {self.client_idx + 1}  Recall Metrics: {client_metrics}")
                

    def train_epoch(self, agg_proto, img_enc, txt_enc):
        train_texts_emb_list = []
        train_images_emb_list = []
        clip_image_list = []
        clip_text_list = []
        for idx, (images, captions, _, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            # captions = captions.to(self.device)
    
            output = self.model(img_enc(images), txt_enc(captions))
            # print('img', output['image_features'].shape)
            # print('txt', output['caption_features'].shape)
            # 加一个本地原型对齐的损失
            # loss_local = PrototypeLoss()
            loss_o= self.criterion(output['image_output'],output['caption_output'])
            train_images_emb_list.append(output['image_output'])
            train_texts_emb_list.append(output['caption_output'])
            clip_image_list.append(output['image_embedding'])
            clip_text_list.append(output['caption_embedding'])
            if agg_proto:
                loss_p = PrototypeLoss(agg_proto)
                l_p = loss_p(output['image_output'], output['caption_output'])
                loss = loss_o + l_p+0.1*self.compute_similarity_loss()
            else:
                loss = loss_o+0.1*self.compute_similarity_loss()
            self.optimizer.zero_grad()

            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()

            if is_test:
                break
        # Concatenate all embeddings for current client
        texts_emb = torch.cat(train_texts_emb_list, dim=0)
        images_emb = torch.cat(train_images_emb_list, dim=0)
        clip_image_emb = torch.cat(clip_image_list, dim=0)
        clip_text_emb = torch.cat(clip_text_list, dim=0)
                
        # Compute recall metrics for the current client
        client_metrics = compute_flickr30k_recall(texts_emb, images_emb, self.train_same_img,recall_k_list=[1, 5, 10], device = self.device)
            

        self.logger.log(f"Client {self.client_idx}  Recall Metrics: {client_metrics}")
        
        clip_metrics = compute_flickr30k_recall(clip_image_emb, clip_text_emb, self.train_same_img, recall_k_list=[1, 5, 10], device = self.device)
        self.logger.log(f"CLIP: Client {self.client_idx}  Recall Metrics: {clip_metrics}")
        # loss_dict = {'{}{}'.format(prefix, key): val
        #              for key, val in loss_dict.items()}
        # loss_dict['step'] = cur_step(self.cur_epoch, idx, len(self.train_loader))

        # criterion = nn.CrossEntropyLoss().cuda()
        # if self.args.contrast_local_intra and self.args.contrast_local_inter:
        #     global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
        #     distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
        #     print("Start Intra & Inter Contrasting!")
        #     for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
        #         self.optimizer.zero_grad()
        #         d_idx = operator.itemgetter(*index)(distill_dict)  # idx of current batch

        #         images = images.to(self.device)
        #         captions = captions.to(self.device)
        #         caption_lens = caption_lens.to(self.device)

        #         output = self.model(images, captions, captions_word, caption_lens)

        #         out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
        #             'image_features']
        #         out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
        #             output['caption_features']

        #         target_img_feature = global_img_feature[d_idx, :].type_as(out_img)
        #         target_txt_feature = global_txt_feature[d_idx, :].type_as(out_txt)

        #         # pos
        #         pos_i = torch.sum(out_img * target_img_feature, dim=-1)
        #         pos_i = pos_i.reshape(-1, 1)
        #         pos_t = torch.sum(out_txt * target_txt_feature, dim=-1)
        #         pos_t = pos_t.reshape(-1, 1)
        #         # neg
        #         with torch.no_grad():
        #             output_o = self.old_model(images, captions, captions_word, caption_lens)
        #             out_img_o = output_o['image_features'].sum(axis=1) if len(output_o['image_features'].shape) == 3 else output_o['image_features']
        #             out_txt_o = output_o['caption_features'].sum(axis=1) if len(output_o['caption_features'].shape) == 3 else output_o['caption_features']
        #         neg_i = torch.sum(out_img * out_img_o, dim=-1)
        #         neg_t = torch.sum(out_txt * out_txt_o, dim=-1)
        #         logits_1 = torch.cat((pos_i, neg_i.reshape(-1, 1)), dim=1)
        #         logits_2 = torch.cat((pos_t, neg_t.reshape(-1, 1)), dim=1)
        #         logits = torch.cat((logits_1, logits_2), dim=0)

        #         logits /= 0.5  # temperature
        #         labels = torch.zeros(images.size(0) * 2).cuda().long()

        #         loss_intra = criterion(logits, labels)

        #         # inter contrast
        #         logits_1_inter = torch.div(torch.matmul(out_img, global_txt_feature.T), 0.5)
        #         logits_2_inter = torch.div(torch.matmul(out_txt, global_img_feature.T), 0.5)

        #         labels_inter = torch.tensor(d_idx).cuda()

        #         loss_1_inter = criterion(logits_1_inter, labels_inter)
        #         loss_2_inter = criterion(logits_2_inter, labels_inter)
        #         loss_inter = loss_1_inter + loss_2_inter

        #         if not self.args.loss_scale:
        #             loss = (loss_intra + loss_inter) * self.args.interintra_weight
        #         else:
        #             loss = (loss_intra + loss_inter / (loss_inter / loss_intra).detach()) * self.args.interintra_weight

        #         self.optimizer.zero_grad()

        #         if self.config.train.get('use_fp16'):
        #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #                 scaled_loss.backward()
        #         else:
        #             loss.backward()

        #         if self.config.train.grad_clip > 0:
        #             nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
        #                                                self.config.train.grad_clip)
        #         self.optimizer.step()

        #         if is_test:
        #             break
        # elif self.args.contrast_local_intra:
        #     global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
        #     distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
        #     print("Start Intra Contrasting!")
        #     for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
        #         self.optimizer.zero_grad()
        #         d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx

        #         images = images.to(self.device)
        #         captions = captions.to(self.device)
        #         caption_lens = caption_lens.to(self.device)

        #         output = self.model(images, captions, captions_word, caption_lens)

        #         out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
        #             'image_features']
        #         out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
        #             output['caption_features']

        #         target_img_feature = global_img_feature[d_idx, :].type_as(out_img)
        #         target_txt_feature = global_txt_feature[d_idx, :].type_as(out_txt)

        #         # pos
        #         pos_i = torch.sum(out_img * target_img_feature, dim=-1)
        #         pos_i = pos_i.reshape(-1, 1)
        #         pos_t = torch.sum(out_txt * target_txt_feature, dim=-1)
        #         pos_t = pos_t.reshape(-1, 1)
        #         # neg
        #         with torch.no_grad():
        #             output_o = self.old_model(images, captions, captions_word, caption_lens)
        #             out_img_o = output_o['image_features'].sum(axis=1) if len(output_o['image_features'].shape) == 3 else output_o['image_features']
        #             out_txt_o = output_o['caption_features'].sum(axis=1) if len(output_o['caption_features'].shape) == 3 else output_o['caption_features']
        #         neg_i = torch.sum(out_img * out_img_o, dim=-1)
        #         neg_t = torch.sum(out_txt * out_txt_o, dim=-1)
        #         logits_1 = torch.cat((pos_i, neg_i.reshape(-1, 1)), dim=1)
        #         logits_2 = torch.cat((pos_t, neg_t.reshape(-1, 1)), dim=1)
        #         logits = torch.cat((logits_1, logits_2), dim=0)

        #         logits /= 0.5  # temperature
        #         labels = torch.zeros(images.size(0) * 2).cuda().long()

        #         loss = criterion(logits, labels)

        #         self.optimizer.zero_grad()

        #         if self.config.train.get('use_fp16'):
        #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #                 scaled_loss.backward()
        #         else:
        #             loss.backward()

        #         if self.config.train.grad_clip > 0:
        #             nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
        #                                                self.config.train.grad_clip)
        #         self.optimizer.step()

        #         if is_test:
        #             break
        # elif self.args.contrast_local_inter:
        #     global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
        #     distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
        #     print("Start Inter-modal Contrasting!")
        #     for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader),
        #                                                                    total=len(global_train_loader)):
        #         self.optimizer.zero_grad()
        #         d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx

        #         images = images.to(self.device)
        #         captions = captions.to(self.device)
        #         caption_lens = caption_lens.to(self.device)

        #         output = self.model(images, captions, captions_word, caption_lens)

        #         out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
        #             'image_features']
        #         out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
        #             output['caption_features']

        #         logits_1 = torch.div(torch.matmul(out_img, global_txt_feature.T), 0.5)
        #         logits_2 = torch.div(torch.matmul(out_txt, global_img_feature.T), 0.5)

        #         labels = torch.tensor(d_idx).cuda()

        #         loss_1 = criterion(logits_1, labels)
        #         loss_2 = criterion(logits_2, labels)
        #         loss = loss_1 + loss_2

        #         self.optimizer.zero_grad()

        #         if self.config.train.get('use_fp16'):
        #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #                 scaled_loss.backward()
        #         else:
        #             loss.backward()

        #         if self.config.train.grad_clip > 0:
        #             nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
        #                                                self.config.train.grad_clip)
        #         self.optimizer.step()

        #         if is_test:
        #             break

    def generate_logits(self, dataloader):
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            img_vec = []
            txt_vec = []
            distill_index = []
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(dataloader),
                                                                           total=len(dataloader)):
                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = output['image_output'].sum(axis=1) if len(output['image_output'].shape) == 3 else output[
                    'image_output']
                out_txt = output['caption_output'].sum(axis=1) if len(output['caption_output'].shape) == 3 else \
                    output['caption_output']
                img_vec.extend(out_img)
                txt_vec.extend(out_txt)
                distill_index.extend(index)

                if is_test and idx == 1:
                    break

        img_vec = torch.cat(img_vec, dim=0).view(-1, self.args.feature_dim)
        txt_vec = torch.cat(txt_vec, dim=0).view(-1, self.args.feature_dim)

        img_vec = img_vec.cpu()
        txt_vec = txt_vec.cpu()
        self.model.cpu()

        return {'img': img_vec, 'txt': txt_vec}, distill_index

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
