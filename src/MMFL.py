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
from collections import OrderedDict
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
import numpy as np
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

        os.makedirs(f'/root/autodl-fs/data/yClient', exist_ok=True)

        # load data
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = args.image_data
            self.img_trainloaders, test_set, class_label = get_FL_trainloader(dataset, f"/root/autodl-fs/data/{dataset}",
                                                                 args.num_img_clients, "hetero", 0.1, 512)
            dst = f'/root/autodl-fs/data/yClient/{dataset}'
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
            self.txt_trainloaders, test_set, class_label  = get_FL_trainloader(dataset, "/autodl-fs/data",
                                                                 args.num_txt_clients, "hetero", 0.1, 512)
            # client_id = 1
            dst = f'/autodl-fs/data/yClient/{dataset}'
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
            
        
    def train(self, round_n, pre_global_proto, pre_global_model, clip_enc):
        
        self.cur_epoch = round_n
        
        self.cur_trainers = self.total_local_trainers
        
        
        agg_client_model = {}
        
        agg_global_proto = {}

        img_vec, img_num = [], []
        txt_vec, txt_num = [], []
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx}!")
            print(f"Training Client {trainer.client_idx}!")
            print(f'Training {trainer.type} Client')
            trainer.cur_epoch = round_n
            local_m, local_proto = trainer.run(pre_global_proto, pre_global_model[idx], clip_enc)
            self.logger.log(f"Round {round_n}: Local Training of {trainer.type} Client {trainer.client_idx} has completed")
            agg_global_proto[trainer.client_idx] = local_proto
            agg_client_model[trainer.client_idx] = local_m
            
        
        image_p = [] 
        text_p = []
        for i in range(len(self.total_local_trainers)):
            c_p = agg_global_proto.get(i)
            if c_p.get('image') is not None and c_p.get('text') is not None:
                image_p.extend(c_p.get('image')) 
                text_p.extend(c_p.get('text')) 
       
        for i in range(len(self.total_local_trainers)):
            if c_p.get('image') is None and c_p.get('text') is not None: 
                text_protos = c_p.get('text') 
                completed_image_protos = []
                
                for text_proto in text_protos:
                    similarities = []

                    for txt_proto in text_p:
                        sim = F.cosine_similarity(torch.tensor(text_proto), torch.tensor(txt_proto), dim=0)  
                        similarities.append(sim)

                    
                    top_k_similarities, top_k_indices = torch.topk(torch.tensor(similarities), self.config.agg.agg_k)

                    
                    weights = F.softmax(top_k_similarities, dim=0)

                    
                    weighted_image_proto = torch.zeros_like(torch.tensor(image_p[0]))  

                    for idx, weight in zip(top_k_indices, weights):
                        weighted_image_proto += weight * torch.tensor(image_p[idx])  

                    
                    completed_image_protos.append(weighted_image_proto.tolist())

               
                agg_global_proto[i]['image'] = completed_image_protos
            elif c_p.get('image') is not None and c_p.get('text') is None: 
                image_protos = c_p.get('image') 
                completed_text_protos = []
               
                for image_proto in image_protos:
                    similarities = []

                    
                    for img_proto in image_p:
                        sim = F.cosine_similarity(torch.tensor(image_proto), torch.tensor(img_proto), dim=0)  
                        similarities.append(sim)

                    
                    top_k_similarities, top_k_indices = torch.topk(torch.tensor(similarities), self.config.agg.agg_k)

                    
                    weights = F.softmax(top_k_similarities, dim=0)

                    weighted_txt_proto = torch.zeros_like(torch.tensor(text_p[0]))  

                    for idx, weight in zip(top_k_indices, weights):
                        weighted_txt_proto += weight * torch.tensor(text_p[idx]) 

                    
                    completed_text_protos.append(weighted_txt_proto.tolist())

                
                agg_global_proto[i]['text'] = completed_text_protos

        global_proto = self.cluster_proto(agg_global_proto,self.config.agg.cluster_k)
        
        image_models = []
        text_models = []
       
        for client_idx, local_m in agg_client_model.items():
            image_models.append(local_m.get('image', None))  
            text_models.append(local_m.get('text', None))
        image_m, text_m = self.compute_model_similarity(agg_client_model)
       
        weighted_image_models = self.compute_weighted_model(image_m, image_models)
        weighted_text_models = self.compute_weighted_model(text_m, text_models)
        agg_model = []
        for i in range(len(weighted_image_models)):
            agg_model.append({'image':weighted_image_models[i],'text':weighted_text_models[i]})
        return global_proto, agg_model
        
       
        
    def compute_weighted_model(self, similarity_matrix, models: list):

        num_clients = len(models)
        weighted_models = []
        weight_matrix = F.softmax(torch.tensor(similarity_matrix), dim=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        updated_models = [self.update_model_keys(model) if model is not None else None for model in models]
        # updated_models = [{k: v.to("cuda:0") for k, v in model.items()} for model in updated_models]

        for i in range(num_clients):
            if updated_models[i] is None:
                weighted_models.append(None)  
                continue

           
            weighted_model = OrderedDict((key, torch.zeros_like(value).to(device)) for key, value in updated_models[i].items())
           
            for j in range(num_clients):
                if updated_models[j] is not None:
                    for key in updated_models[j]:
                        weighted_model[key] += weight_matrix[i, j] * updated_models[j][key].to(device)
            
            
            weighted_models.append(weighted_model)

        return weighted_models
    
    def update_model_keys(self,model):

        updated_model = OrderedDict()
        for key, value in model.items():
            
            layer_number = key.split('.')[0]
            param_type = key.split('.')[1]
            new_key = f"fc.{layer_number}.{param_type}"
            updated_model[new_key] = value
        return updated_model
        
    def compute_model_similarity(self,agg_client_model):
       
        image_models = []
        text_models = []
    
        
        for client_idx, local_m in agg_client_model.items():
            image_models.append(local_m.get('image', None))  
            text_models.append(local_m.get('text', None))
    
       
        image_similarity_matrix = self.compute_similarity_matrix(image_models)
    
        
        text_similarity_matrix = self.compute_similarity_matrix(text_models)
    
        return image_similarity_matrix, text_similarity_matrix
    
    def compute_similarity_matrix(self, models):
        
        n = len(models)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if models[i] is None or models[j] is None:
                    similarity = 0.0  
                else:
                    similarity = self.compute_model_similarity_score(models[i], models[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  
        
        return similarity_matrix

    def compute_model_similarity_score(self, model_a, model_b):

        
        model_a_vector = np.concatenate([np.array(param.detach().cpu().numpy().flatten()) for param in model_a.values()])
        model_b_vector = np.concatenate([np.array(param.detach().cpu().numpy().flatten()) for param in model_b.values()])
        
        if np.linalg.norm(model_a_vector) == 0 or np.linalg.norm(model_b_vector) == 0:
            return 0.0
        
       
        similarity = np.dot(model_a_vector, model_b_vector) / (np.linalg.norm(model_a_vector) * np.linalg.norm(model_b_vector))
        return similarity
        
    def cluster_proto(self,agg_proto, K):
        
        fused_protos = []
        image_protos = []
        text_protos = []

        for i in range(len(agg_proto)):
            c_p = agg_proto[i]
            if c_p.get('image') is not None and c_p.get('text') is not None:
                
                image_proto = c_p['image'] 
                text_proto = c_p['text']    

                
                fused_proto = 0.5*image_proto + 0.5*text_proto

                fused_protos.append(fused_proto)  
                image_protos.append(image_proto)  
                text_protos.append(text_proto)   
        
       
        fused_protos = torch.cat(fused_protos, dim=0) 
        image_protos = torch.cat(image_protos, dim=0)
        text_protos = torch.cat(text_protos, dim=0)

        
        kmeans = KMeans(n_clusters=K, random_state=0,init='k-means++')
        labels = kmeans.fit_predict(fused_protos.cpu().detach().numpy())  

        
        global_image_protos = []
        global_text_protos = []

        for k in range(K):
            
            cluster_indices = (labels == k)
            
            
            cluster_image_protos = torch.stack([image_protos[i] for i in range(len(image_protos)) if cluster_indices[i]])
            cluster_text_protos = torch.stack([text_protos[i] for i in range(len(text_protos)) if cluster_indices[i]])

        
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
