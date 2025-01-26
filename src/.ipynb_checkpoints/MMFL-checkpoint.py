import gc
import random

import operator
import os
from copy import deepcopy
import sys

import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.load_datasets import get_FL_trainloader, get_dataloader
from src.ClientTrainer import ClientTrainer
from src.MMClientTrainer import MMClientTrainer
from src.utils.color_lib import RGBmean, RGBstdv
import torch.nn.functional as F
from src.retrieval_trainer import TrainerEngine, rawTrainerEngine
from src.utils.config import parse_config
from src.utils.logger import PythonLogger
from sklearn.cluster import KMeans
try:
    from apex import amp
except ImportError:
    print('failed to import apex')

# TODO: test
is_test = False


class MMFL(object):
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        self.engine = None
        self.best_score = 0
        self.cur_epoch = 0

        # img & txt local dataloaders
        self.img_train_loaders, self.txt_train_loaders = None, None

        # # coco global dataloaders
        # self.dataloaders_global = None
        # # universal test dataloader
        # self.test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)

        self.img_vec, self.txt_vec = None, None

    def set_config(self, img='cifa10', txt='AG_NEWS'):
        self.config = parse_config("./coco.yaml", strict_cast=False)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = 'model_noprob'
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim

        if self.args.not_bert:
            self.config.model.not_bert = True
            self.config.model.cnn_type = 'resnet50'
        else:
            self.config.model.not_bert = False
            self.config.model.cnn_type = 'resnet101'


    def load_data(self, args):
        self.logger.log('start partition datasets')
        self.device = torch.device("cuda:%d" % args.device)

        os.makedirs(os.environ['HOME'] + f'/autodl-tmp/shared-nvme/yClient', exist_ok=True)

        # load data
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = args.image_data
            self.img_trainloaders, test_set, class_label = get_FL_trainloader(dataset, os.environ['HOME'] + f"/autodl-tmp/shared-nvme/{dataset}",
                                                                 args.num_img_clients, "hetero", 0.1, 512)
            dst = os.environ['HOME'] + f'/autodl-tmp/shared-nvme/yClient/{dataset}'
            self.img_local_trainers = []
            config_image = parse_config("./image.yaml", strict_cast=False)
            for i in range(args.num_img_clients):
                self.img_local_trainers.append(
                    ClientTrainer(args, config_image, dataset, dst, self.img_trainloaders[i], test_set, class_label, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, inter_distance=4, client_id=i, wandb=self.wandb))
                # self.img_local_trainers[i].train_loader = self.img_trainloaders[i]
                if is_test and i == 0:
                    break
        # txt clients
        if args.num_txt_clients > 0:
            dataset = 'AG_NEWS'
            self.txt_trainloaders, test_set, class_label  = get_FL_trainloader(dataset, os.environ['HOME'] + "/autodl-tmp/shared-nvme",
                                                                 args.num_txt_clients, "hetero", 0.1, 512)
            # client_id = 1
            dst = os.environ['HOME'] + f'/autodl-tmp/shared-nvme/yClient/{dataset}'
            self.txt_local_trainers = []
            config_text = parse_config("./text.yaml", strict_cast=False)
            for i in range(args.num_txt_clients):
                self.txt_local_trainers.append(
                    ClientTrainer(args, config_text, dataset, dst, self.txt_trainloaders[i], test_set, class_label, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, ctype='text', inter_distance=4, client_id=i, wandb=self.wandb))
                # self.txt_local_trainers[i].train_loader = self.txt_trainloaders[i]
                if is_test and i == 0:
                    break
        # mm clients
        if args.num_mm_clients > 0:
            # mm img models
            config = parse_config("./f30k.yaml", strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.not_bert = True
            self.mm_local_trainers = []
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    MMClientTrainer(args, config, self.logger, client=client_id, dset_name="flickr30k",
                                    device='cuda',
                                    vocab_path='./coco_vocab.pkl',
                                    mlp_local=self.args.mlp_local))
                # 添加各参与方的数据集
                if is_test and client_id == 0:
                    break
            print(f"Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers

        for i in range(len(self.total_local_trainers)):
            self.total_local_trainers[i].client_idx = i
        print('加载数据和模型完成')
            
        
    def train(self, round_n, pre_global_proto, pre_global_model, img_enc, txt_enc):
        # 当前的全局训练轮次
        self.cur_epoch = round_n
        # 参与训练的客户端
        self.cur_trainers = self.total_local_trainers
        
        # 存储聚合之后的每个客户端的全局模型，为字典类型，key为客户端编号，value为客户端模型
        agg_client_model = {}
        # 存储服务器聚合的全局原型，为字典类型，key为image或text，分别代表图像和文本的原型
        agg_global_proto = {}

        # local training and generated representations
        img_vec, img_num = [], []
        txt_vec, txt_num = [], []
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx}!")
            print(f"Training Client {trainer.client_idx}!")
            print(f'Training {trainer.type} Client')
            trainer.cur_epoch = round_n
            local_m, local_proto = trainer.run(pre_global_proto, pre_global_model[idx], img_enc, txt_enc)
            self.logger.log(f"Round {round_n}: Local Training of {trainer.type} Client {trainer.client_idx} has completed")
            agg_global_proto[trainer.client_idx] = local_proto
            agg_client_model[trainer.client_idx] = local_m
            
        # 服务器聚合本地原型
        # step 1. 原型补齐
        # 将所有多模态客户端图像和文本原型取出
        image_p = [] 
        text_p = []
        for i in range(len(self.total_local_trainers)):
            c_p = agg_global_proto.get(i)
            if c_p.get('image') is not None and c_p.get('text') is not None:
                image_p.extend(c_p.get('image')) 
                text_p.extend(c_p.get('text')) 
        # 计算单模态客户端原型与多模态原型的相似度 
        for i in range(len(self.total_local_trainers)):
            if c_p.get('image') is None and c_p.get('text') is not None: 
                text_protos = c_p.get('text') # 对于文本客户端
                completed_image_protos = []
                # 遍历每一个文本原型
                for text_proto in text_protos:
                    similarities = []

                    # 计算当前文本原型与所有多模态图像原型的相似度
                    for txt_proto in text_p:
                        sim = F.cosine_similarity(torch.tensor(text_proto), torch.tensor(txt_proto), dim=0)  # 计算余弦相似度
                        similarities.append(sim)

                    # step 3. 选择前k个最相似的多模态图像原型
                    top_k_similarities, top_k_indices = torch.topk(torch.tensor(similarities), self.config.agg.agg_k)

                    # step 4. 使用softmax将前k个相似度转换为权重值
                    weights = F.softmax(top_k_similarities, dim=0)

                    # step 5. 利用权重补齐该文本原型对应的图像原型
                    weighted_image_proto = torch.zeros_like(torch.tensor(image_p[0]))  # 初始化加权图像原型

                    for idx, weight in zip(top_k_indices, weights):
                        weighted_image_proto += weight * torch.tensor(image_p[idx])  # 进行加权和

                    # 将补齐的图像原型添加到结果列表中
                    completed_image_protos.append(weighted_image_proto.tolist())

                # step 6. 将补齐的图像原型列表保存回单模态客户端
                agg_global_proto[i]['image'] = completed_image_protos
            elif c_p.get('image') is not None and c_p.get('text') is None: 
                image_protos = c_p.get('image') # 对于图像客户端
                completed_text_protos = []
                # 遍历每一个图像原型
                for image_proto in image_protos:
                    similarities = []

                    # 计算当前文本原型与所有多模态图像原型的相似度
                    for img_proto in image_p:
                        sim = F.cosine_similarity(torch.tensor(image_proto), torch.tensor(img_proto), dim=0)  # 计算余弦相似度
                        similarities.append(sim)

                    # step 3. 选择前k个最相似的多模态图像原型
                    top_k_similarities, top_k_indices = torch.topk(torch.tensor(similarities), self.config.agg.agg_k)

                    # step 4. 使用softmax将前k个相似度转换为权重值
                    weights = F.softmax(top_k_similarities, dim=0)

                    # step 5. 利用权重补齐该图像原型对应的文本原型
                    weighted_txt_proto = torch.zeros_like(torch.tensor(text_p[0]))  # 初始化加权图像原型

                    for idx, weight in zip(top_k_indices, weights):
                        weighted_txt_proto += weight * torch.tensor(text_p[idx])  # 进行加权和

                    # 将补齐的文本原型添加到结果列表中
                    completed_text_protos.append(weighted_txt_proto.tolist())

                # step 6. 将补齐的文本原型列表保存回单模态客户端
                agg_global_proto[i]['text'] = completed_text_protos

        global_proto = self.cluster_proto(agg_global_proto,self.config.agg.cluster_k)
        # 聚合本地模型
        # 聚合图像模型
        # step 1: 计算
        image_models = []
        text_models = []
        # 提取所有客户端的图像和文本模型，如果不存在则设为 None
        for client_idx, local_m in agg_client_model.items():
            image_models.append(local_m.get('image', None))  # 使用 get 确保键不存在时返回 None
            text_models.append(local_m.get('text', None))
        image_m, text_m = self.compute_model_similarity(agg_client_model)
        # 聚合之后的模型
        weighted_image_models = self.compute_weighted_model(image_m, image_models)
        weighted_text_models = self.compute_weighted_model(text_m, text_models)
        agg_model = {}
        agg_model['image'] = weighted_image_models
        agg_model['text'] = weighted_text_models
        return global_proto, agg_model
        
        # # global distillation
        # if not self.args.disable_distill:
        #     self.distill(round_n, img_vec, txt_vec, img_num, txt_num, self.distill_index)

        # def get_lr(optimizer):
        #     for param_group in optimizer.param_groups:
        #         return param_group['lr']

        # # record after each epoch training
        # metadata = self.engine.metadata.copy()
        # metadata['cur_epoch'] = round_n + 1
        # metadata['lr'] = get_lr(self.engine.optimizer)

        # test_scores = self.engine.evaluate({'test': self._dataloaders['test']})
        # self.engine.report_scores(step=round_n + 1,
        #                           scores=test_scores,
        #                           metadata=metadata,
        #                           prefix=self.engine.eval_prefix)
        # rsum = test_scores['test']['n_fold']['i2t']['recall_1'] + test_scores['test']['n_fold']['t2i']['recall_1'] + \
        #        test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1']
        # self.wandb.log({"Server rsum_r1": rsum}, step=self.cur_epoch)
        # self.wandb.log({"Server n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
        # self.wandb.log({"Server n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
        # self.wandb.log({"Server i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
        # self.wandb.log({"Server t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)

        # if self.best_score < rsum:
        #     best_score = rsum
        #     metadata['best_score'] = best_score
        #     metadata['best_epoch'] = round_n + 1
        #     self.best_metadata, self.best_scores = metadata, test_scores

        #     torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-best_model.pt')

        # if round_n == self.args.comm_rounds - 1:
        #     torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-last_model.pt')

        # self.engine.lr_scheduler.step()

        # del img_vec, txt_vec
        # gc.collect()
        
    def compute_weighted_model(self, similarity_matrix, models):
        """
        计算加权后的模型。
    
        参数:
        - similarity_matrix: 客户端模型的相似性矩阵 (二维张量)
        - models: 客户端模型列表 (每个模型为一个张量)，模型可以为None表示不存在
    
        返回:
        - 加权后的模型列表
        """
        
        
        num_clients = len(models)
        weighted_models = []
        weight_matrix = F.softmax(similarity_matrix, dim=1)
        
        
        for i in range(num_clients):
            if models[i] is None:
                weighted_models.append(None)  # 如果当前客户端的模型不存在，跳过加权
                continue
        
        
            # 初始化加权模型
            weighted_model = torch.zeros_like(models[i])
        
            # 通过加权其他客户端的模型来聚合
            for j in range(num_clients):
                if models[j] is not None:
                    weighted_model += weight_matrix[i, j] * models[j]
        
            weighted_models.append(weighted_model)
    
        return weighted_models
        
    def compute_model_similarity(self,agg_client_model):
        """
        计算所有客户端的图像模型和文本模型相似度矩阵。
    
        参数:
        - agg_client_model: 客户端的本地模型字典 (每个客户端包含 'image' 和 'text' 模型)
    
        返回:
        - 图像模型相似度矩阵
        - 文本模型相似度矩阵
        """
        image_models = []
        text_models = []
    
        # 提取所有客户端的图像和文本模型，如果不存在则设为 None
        for client_idx, local_m in agg_client_model.items():
            image_models.append(local_m.get('image', None))  # 使用 get 确保键不存在时返回 None
            text_models.append(local_m.get('text', None))
    
        # 计算图像模型相似度矩阵
        image_similarity_matrix = self.compute_similarity_matrix(image_models)
    
        # 计算文本模型相似度矩阵
        text_similarity_matrix = self.compute_similarity_matrix(text_models)
    
        return image_similarity_matrix, text_similarity_matrix
    
    def compute_similarity_matrix(self, models):
        """
        计算客户端模型之间的相似度矩阵。
    
        参数:
        - models: 客户端模型列表 (每个模型为一个张量)
    
        返回:
        - 相似度矩阵 (二维张量)
        """
        num_clients = len(models)
        similarity_matrix = torch.zeros(num_clients, num_clients)

        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 自己与自己的相似度为 1
                elif models[i] is not None and models[j] is not None:
                    similarity_matrix[i, j] = F.cosine_similarity(models[i].flatten(), models[j].flatten(), dim=0)
                else:
                    similarity_matrix[i, j] = 0.0
    
        return similarity_matrix
        
    def cluster_proto(self,agg_proto, K):
        # step 1: 图像文本特征融合
        fused_protos = []
        image_protos = []
        text_protos = []

        for i in range(len(agg_proto)):
            c_p = agg_proto[i]
            if c_p.get('image') is not None and c_p.get('text') is not None:
                # 取出图像和文本原型
                image_proto = c_p['image']  # 图像原型
                text_proto = c_p['text']    # 文本原型

                # 融合：$\hat{p} = p_i + p_t$
                fused_proto = 0.5*image_proto + 0.5*text_proto

                fused_protos.append(fused_proto)  # 保存融合的原型
                image_protos.append(image_proto)  # 保存图像原型
                text_protos.append(text_proto)    # 保存文本原型
        # 确保 fused_protos 中的每个元素都是 Tensor
       
        fused_protos = torch.stack(fused_protos)  # 转换为tensor用于聚类

        # step 2: 对融合后的原型进行聚类，簇的数量为K
        kmeans = KMeans(n_clusters=K, random_state=0)
        labels = kmeans.fit_predict(fused_protos.cpu().detach().numpy())  # 使用K-means进行聚类，得到每个融合原型的簇标签

        # step 3: 聚类后，计算每个簇内的图像原型和文本原型的均值，得到K个全局原型对
        global_image_protos = []
        global_text_protos = []

        for k in range(K):
            # 获取属于第k个簇的所有索引
            cluster_indices = (labels == k)

            # 属于第k个簇的图像原型和文本原型
            cluster_image_protos = torch.stack([image_protos[i] for i in range(len(image_protos)) if cluster_indices[i]])
            cluster_text_protos = torch.stack([text_protos[i] for i in range(len(text_protos)) if cluster_indices[i]])

            # 计算第k个簇的全局图像原型和全局文本原型的均值
            global_image_proto = cluster_image_protos.mean(dim=0)
            global_text_proto = cluster_text_protos.mean(dim=0)

            global_image_protos.append(global_image_proto)
            global_text_protos.append(global_text_proto)
        global_proto = {'image': global_image_protos,'text':global_text_protos}
        return global_proto


    def distill(self, round_n, img_vec, txt_vec, img_num, txt_num, distill_index):

        self.engine.model.train()

        if self.config.model.use_img_client or self.config.model.use_txt_client or self.config.model.use_mm_client:
            client_loss_cri = nn.MSELoss()

        def aggregation(i_vec=img_vec, t_vec=txt_vec, i_num=img_num, t_num=txt_num):
            if self.args.agg_method == "con_w":
                contrastive_w = []
                for vec in i_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                    logits = torch.matmul(vec, self.global_txt_feature.T)  # [50000, 50000]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                    contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                for i in range(len(i_vec)):
                    i_vec[i] = (i_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                i_vec = torch.sum(torch.cat(i_vec, dim=0), dim=0)  # aggregated image vectors

                contrastive_w = []
                for vec in t_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                    logits = torch.matmul(vec, self.global_img_feature.T)  # [50000, 50000]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                    contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                for i in range(len(t_vec)):
                    t_vec[i] = (t_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                t_vec = torch.sum(torch.cat(t_vec, dim=0), dim=0)  # aggregated text vectors
            else:
                raise NotImplementedError

            return i_vec, t_vec

        # aggregation
        img_vec, txt_vec = aggregation()

        self.img_vec = img_vec
        self.txt_vec = txt_vec

        distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
        # distill
        self.logger.log("start distilling")
        for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(
                enumerate(self.dataloaders_global['train_subset' + f'_{self.args.pub_data_num}'])):
            images = images.to(self.engine.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.engine.device)  # [bs, seq_len]
            caption_lens = caption_lens.to(self.engine.device)

            output = self.engine.model(images, captions, captions_word, caption_lens)
            loss = 0

            def code_sim(output, target, config):
                output = output.sum(axis=1) if len(output.shape) == 3 else output
                target = target.type_as(output)

                return client_loss_cri(output, target.type_as(output))

            if self.args.num_img_clients > 0:
                out_img = output['image_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                loss += self.args.kd_weight * code_sim(out_img, target_img, self.config)
            if self.args.num_txt_clients > 0:
                out_txt = output['caption_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.args.kd_weight * code_sim(out_txt, target_txt, self.config)
            if self.args.num_mm_clients > 0:
                out_img = output['image_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                out_txt = output['caption_features']
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.args.kd_weight * code_sim(out_img, target_img, self.config)
                loss += self.args.kd_weight * code_sim(out_txt, target_txt, self.config)

            self.engine.optimizer.zero_grad()

            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.engine.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.engine.model.parameters(),
                                                   self.config.train.grad_clip)
            self.engine.optimizer.step()
