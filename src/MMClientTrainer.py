import copy
import operator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.optim
import torch.utils.data
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
from sklearn.cluster import KMeans
import numpy as np
import os
import random
import torch.multiprocessing
from src.test import ImageTextRetrievalTester,compute_flickr30k_recall,compute_recall
torch.multiprocessing.set_sharing_strategy('file_system')
from criterions.DistillationLoss import DistillationLoss
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
from tqdm import tqdm

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

    def dataloader_with_indices(self, dataloader):
        start = 0
        for x, y in dataloader:
            end = start + len(x)
            inds = torch.arange(start, end)
            yield x, y, inds
            start = end
    def run(self, agg_proto, agg_model, clip_enc):
        # self.old_model = copy.deepcopy(self.model)
        # self.old_model.eval().cuda()
        self.model.cuda()
        self.cluster_model.cuda()
        self.shared_model.cuda()
        # for param in self.model.parameters():
        #     if param.dtype == torch.half:
        #         param.data = param.data.float()
        # for param in self.cluster_model.parameters():
        #     if param.dtype == torch.half:
        #         param.data = param.data.float()
        # if self.local_epoch == 0:
        #     self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
        #                                                 opt_level='O2')
        #     self.cluster_model, self.c_optimizer = amp.initialize(self.cluster_model, self.c_optimizer, 
        #                                                           opt_level='O2')
        
        
        self.cluster_model.train()
        for i in range(self.local_epochs):
            
            cluster_model_path = f'./results/mm/Client{self.client_idx}_cluster_model_global{self.cur_epoch}_local{i}.pth'
    
            if os.path.exists(cluster_model_path):
                self.logger.log(f"Model for global epoch {self.cur_epoch}, local epoch {i} found. Loading...")
                self.cluster_model.load_state_dict(torch.load(cluster_model_path))
                self.cluster_model.image_ph.unfreeze_grad()
                self.cluster_model.text_ph.unfreeze_grad()
                # self.cluster_model.cuda()
                continue
            
            if self.logger is not None:
                self.logger.log(f"MM Client Local Proto Training: Epoch {i}")
            self.cluster_model.image_ph.unfreeze_grad()
            self.cluster_model.text_ph.unfreeze_grad()
            for images, batch_captions, inds in tqdm(self.dataloader_with_indices(self.train_loader)):
                images = images.to(self.device)
                # captions = captions.to(self.device)
                batch_texts_image_index = [ind for ind, captions in zip(inds, batch_captions) for text in captions]
                output = self.cluster_model(clip_enc.img_enc(images), clip_enc.txt_enc(batch_captions, True))
                image_output_expanded = output['image_output'].repeat_interleave(5, dim=0)
                loss_o = self.criterion(image_output_expanded,output['caption_output'])
                
                fusion = (image_output_expanded +output['caption_output'])/2
                
                num_clusters = self.config.train_proto.cluster_num                                                                                                                                                                                  
                kmeans = KMeans(n_clusters=num_clusters, random_state=0,init='k-means++').fit(fusion.cpu().detach().numpy())
                cluster_assignments = torch.tensor(kmeans.labels_).to(self.device) 
        
                
                intra_loss = 0
                for i in range(fusion.size(0)):
                    current_cluster = cluster_assignments[i]
                    same_cluster_indices = (cluster_assignments == current_cluster).nonzero(as_tuple=True)[0]
                    different_cluster_indices = (cluster_assignments != current_cluster).nonzero(as_tuple=True)[0]

                    # Positive samples: same cluster, same modality
                    positive_image_elements = image_output_expanded[same_cluster_indices]
                    positive_caption_elements = output['caption_output'][same_cluster_indices]
                    current_image = image_output_expanded[i].unsqueeze(0)
                    current_caption = output['caption_output'][i].unsqueeze(0)

                    # Negative samples: different clusters, same modality
                    negative_image_elements = image_output_expanded[different_cluster_indices]
                    negative_caption_elements = output['caption_output'][different_cluster_indices]

                    # Calculate similarity for positive and negative samples (image modality)
                    positive_similarity_image = torch.exp(torch.mm(current_image, positive_image_elements.t()) / self.config.train_proto.temperature)
                    negative_similarity_image = torch.exp(torch.mm(current_image, negative_image_elements.t()) / self.config.train_proto.temperature)
                    negative_similarity_sum = torch.sum(negative_similarity_image, dim=1, keepdim=True)
                    intra_loss -= torch.mean(torch.log(positive_similarity_image / (positive_similarity_image + negative_similarity_sum)))

                    # Calculate similarity for positive and negative samples (caption modality)
                    positive_similarity_caption = torch.exp(torch.mm(current_caption, positive_caption_elements.t()) / self.config.train_proto.temperature)
                    negative_similarity_caption = torch.exp(torch.mm(current_caption, negative_caption_elements.t()) / self.config.train_proto.temperature)
                    negative_similarity_sum = torch.sum(negative_similarity_caption, dim=1, keepdim=True)
                    intra_loss -= torch.mean(torch.log(positive_similarity_caption / (positive_similarity_caption + negative_similarity_sum)))

                # Inter-modal loss calculation using fused representations
                inter_loss = 0
                for i in range(fusion.size(0)):
                    current_cluster = cluster_assignments[i]
                    same_cluster_indices = (cluster_assignments == current_cluster).nonzero(as_tuple=True)[0]

                    # Positive samples: same cluster, different modality
                    positive_caption_elements = output['caption_output'][same_cluster_indices]
                    current_image = image_output_expanded[i].unsqueeze(0)

                    # Calculate similarity for inter-modal positive and all caption samples
                    positive_similarity_inter = torch.exp(torch.mm(current_image, positive_caption_elements.t()) / self.config.train_proto.temperature)
                    all_caption_similarity_inter = torch.exp(torch.mm(current_image, output['caption_output'].t()) / self.config.train_proto.temperature)
                    inter_loss -= torch.mean(torch.log(positive_similarity_inter / torch.sum(all_caption_similarity_inter)))

    
                total_loss = loss_o + intra_loss + inter_loss
        
                
                self.c_optimizer.zero_grad()
                total_loss.backward()
                self.c_optimizer.step()
                torch.cuda.empty_cache()

            torch.save(self.cluster_model.state_dict(), cluster_model_path)

        
        self.cluster_model.eval()
        self.cluster_model.image_ph.freeze_grad()
        self.cluster_model.text_ph.freeze_grad()

        all_image_embeddings = []
        all_caption_embeddings = []
        with torch.no_grad():
            for images, batch_captions, inds in tqdm(self.dataloader_with_indices(self.train_loader)):
                images = images.to(self.device)
            
                # captions = captions.to(self.device)
                
                output = self.cluster_model(clip_enc.img_enc(images), clip_enc.txt_enc(batch_captions, True))
                all_image_embeddings.append(output['image_output'])
                all_caption_embeddings.append(output['caption_output'])
        
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_caption_embeddings = torch.cat(all_caption_embeddings, dim=0)
        re_img = all_image_embeddings.repeat_interleave(5, dim=0)
        
        fusion = (re_img + all_caption_embeddings) / 2
        num_clusters = self.config.train_proto.cluster_num                                                                                                                                                                                   
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=50, tol=1e-4, max_iter=300).fit(fusion.cpu().detach().numpy())
        
        cluster_assignments = torch.tensor(kmeans.labels_).to(self.device)  
        
        
        prototypes = {'image': [], 'text': []}
        for j in range(num_clusters):
            cluster_indices = (cluster_assignments == j)
            if cluster_indices.sum() == 0:
                continue  
            
            
            image_prototype = re_img[cluster_indices].mean(dim=0)
            caption_prototype = all_caption_embeddings[cluster_indices].mean(dim=0)
            
            
            prototypes['image'].append(image_prototype)
            prototypes['text'].append(caption_prototype)
        
       
        prototypes['image'] = torch.stack(prototypes['image'])
        prototypes['text'] = torch.stack(prototypes['text'])
        
        
        
        
        
        if self.cur_epoch>0: 
            agg_model_image_state_dict = {key: value for key, value in agg_model['image'].items() if key.startswith('fc')}
            self.shared_model.image_ph.load_state_dict(agg_model_image_state_dict)
            agg_model_text_state_dict = {key: value for key, value in agg_model['text'].items() if key.startswith('fc')}
            self.shared_model.text_ph.load_state_dict(agg_model_text_state_dict)
       
       
        self.model.train()
        
        for i in range(self.local_epochs):
            model_path = f'./results/mm/Client{self.client_idx}-global_{self.cur_epoch}-model_{i}.pth'
            if os.path.exists(model_path):
                self.logger.log(f"Model for client {self.client_idx} global epoch {self.cur_epoch}, local epoch {i} found. Loading...")
                self.model.load_state_dict(torch.load(model_path))
                continue
            if self.logger is not None:
                self.logger.log(f"MM Client Local Training: Epoch {self.local_epoch}")
            self.train_epoch(agg_proto, clip_enc)
            self.local_epoch += 1
            torch.save(self.model.state_dict(), model_path)

        # test = ImageTextRetrievalTester(self.model, self.val_loader,self.device, img_enc, txt_enc)
        # test.run_tests()
        # self.old_model.cpu()
        # self.model.cpu()
        self.test(clip_enc=clip_enc)
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
    
    def test(self, clip_enc):
        texts_image_index = []
        image_features = []
        text_features = []
        
        self.model.eval()
        with torch.no_grad():
            for images, batch_captions, inds in tqdm(self.dataloader_with_indices(self.val_loader)):
                images = images.to(self.device)
                batch_texts_image_index = [ind for ind, captions in zip(inds, batch_captions) for text in captions]
            # captions = captions.to(self.device)
    
                output = self.model(clip_enc.img_enc(images), clip_enc.txt_enc(batch_captions,True))
                texts_image_index.extend(batch_texts_image_index)
                image_features.append(output['image_output'])
                text_features.append(output['caption_output'])
        # Concatenate all embeddings for current client
        batch_size = len(image_features[0])
        
        texts_emb = torch.cat(text_features, dim=0)
        images_emb = torch.cat(image_features, dim=0)
                
        # Compute recall metrics for the current 
        # client_metrics = compute_recall(images_emb, texts_emb, [1, 5, 10])
        client_metrics = compute_flickr30k_recall(texts_emb, images_emb, ind_t_img=texts_image_index, batch_size=batch_size, recall_k_list=[1, 5], device = self.device)
            

        self.logger.log(f"Test Client {self.client_idx}  Recall Metrics: {client_metrics}")
                

    def train_epoch(self, agg_proto, clip_enc):
        train_texts_emb_list = []
        train_images_emb_list = []
        texts_image_index = []
        clip_image_list = []
        clip_text_list = []
        for images, batch_captions, inds in tqdm(self.dataloader_with_indices(self.train_loader)):
            images = images.to(self.device)
                # captions = captions.to(self.device)
            batch_texts_image_index = [ind for ind, captions in zip(inds, batch_captions) for text in captions]
            # captions = captions.to(self.device)
            img_emb = clip_enc.img_enc(images)
            txt_emb = clip_enc.txt_enc(batch_captions,True)
            output = self.model(img_emb, txt_emb)
            texts_image_index.extend(batch_texts_image_index)
            # print('img', output['image_features'].shape)
            # print('txt', output['caption_features'].shape)
            
            image_output_expanded = output['image_output'].repeat_interleave(5, dim=0)
            loss_o= self.criterion(image_output_expanded,output['caption_output'])
            train_images_emb_list.append(output['image_output'])
            train_texts_emb_list.append(output['caption_output'])
            clip_image_list.append(output['image_embedding'])
            clip_text_list.append(output['caption_embedding'])
            # print(output['image_output'].shape)
            if agg_proto:
                self.shared_model.image_ph.freeze_grad()
                self.shared_model.text_ph.freeze_grad()
                output_s = self.shared_model(img_emb, txt_emb)
                loss_s = self.criterion(output_s['image_output'].repeat_interleave(5, dim=0),output_s['caption_output'])

                disll = DistillationLoss()
                l_d = disll(output['image_output'],output_s['image_output'])+disll(output['caption_output'],output_s['caption_output'])
                loss_p = PrototypeLoss(agg_proto)
                l_p = loss_p(output['image_output'], output['caption_output'])
                l_o = loss_o.detach().clone()
                loss = loss_o + l_p+0.01*self.compute_similarity_loss()+(l_o/loss_s)*l_d
            else:
                
                loss = loss_o+0.01*self.compute_similarity_loss()
                # self.logger.log(f"loss:{loss}")
            self.optimizer.zero_grad()

            # if self.config.train.get('use_fp16'):
            #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()

            if is_test:
                break
        # Concatenate all embeddings for current client
        batch_size = len(train_images_emb_list[0])
        # print(batch_size)
        texts_emb = torch.cat(train_texts_emb_list, dim=0)
        images_emb = torch.cat(train_images_emb_list, dim=0)
        clip_image_emb = torch.cat(clip_image_list, dim=0)
        clip_text_emb = torch.cat(clip_text_list, dim=0)
        
        # client_metrics = compute_recall(images_emb, texts_emb,[1,5,10])
        # Compute recall metrics for the current client
        client_metrics = compute_flickr30k_recall(texts_emb, images_emb, ind_t_img=texts_image_index, batch_size=batch_size, recall_k_list=[1, 5], device = self.device)
            

        self.logger.log(f"Client {self.client_idx}  Recall Metrics: {client_metrics}")
        
        clip_metrics = compute_flickr30k_recall( clip_text_emb,clip_image_emb, ind_t_img=texts_image_index, batch_size=batch_size, recall_k_list=[1, 5], device = self.device)
        # clip_metrics = compute_flickr30k_recall(clip_text_emb, clip_image_emb, self.train_same_img, recall_k_list=[1, 5, 10], device = self.device)
        self.logger.log(f"CLIP: Client {self.client_idx}  Recall Metrics: {clip_metrics}")
       
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
