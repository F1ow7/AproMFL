import torch.optim as optim
import torch.nn as nn
import torch.optim
import torch.utils.data

from apex import amp
from sklearn.metrics import pairwise_distances

# from src import losses
from src.datasets.cifar import Cifar
from src.utils.dataset_L import caption_collate_fn, Language
# from src.networks.language_model import EncoderText
# from src.networks.resnet_client import resnet18_client
# from src.utils.Reader import ImageReader
# from src.utils.Utils import to_one_hot
from models.ImageNet import ImageNet
from models.TextNet import TextNet
torch.backends.cudnn.enabled = True

import torchvision.transforms as transforms

from tqdm import tqdm

import numpy as np
import os
import random
import torch
import torch.multiprocessing
from criterions.protoloss import  PrototypeLoss
torch.multiprocessing.set_sharing_strategy('file_system')


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


def get_result_list(query_sorted_idx, gt_list, ignore_list, top_k):
    return_retrieval_list = []
    count = 0
    while len(return_retrieval_list) < top_k:
        query_idx = query_sorted_idx[count]
        if query_idx in ignore_list:
            pass
        else:
            if query_idx in gt_list:
                return_retrieval_list.append(1)
            else:
                return_retrieval_list.append(0)
        count += 1
    return return_retrieval_list


def recall_at_k(feature, query_id, retrieval_list, top_k):
    distance = pairwise_distances(feature, feature)
    result = 0
    for i in range(len(query_id)):
        query_distance = distance[query_id[i], :]
        gt_list = retrieval_list[i][0]
        ignore_list = retrieval_list[i][1]
        query_sorted_idx = np.argsort(query_distance)
        query_sorted_idx = query_sorted_idx.tolist()
        result_list = get_result_list(query_sorted_idx, gt_list, ignore_list, top_k)
        result += 1. if sum(result_list) > 0 else 0
    result = result / float(len(query_id))
    return result


gpuid = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.to(gpuid)
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# TODO: test
is_test = False


class ClientTrainer:
    def __init__(self, args, config, dataset, dst, traindata, testdata, class_label, RGBmean, RGBstdv, data_dict, logger, global_test_set, ctype='image', inter_distance=4, loss='softmax',
                 gpuid='cuda:0', num_epochs=30, init_lr=0.1, decay=0.1, batch_size=512,
                 imgsize=256, num_workers=4, print_freq=10, save_step=10, scale=128, pool_type='max_avg', client_id=-1, wandb=None):
        seed_torch()
        self.type = ctype
        self.args = args
        
        self.client_idx = client_id
        self.dset_name = dataset
        self.config = config

        self.dst = dst  # save dir
        self.gpuid = gpuid if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.decay_time = [False, False]
        self.init_lr = self.args.lr
        self.decay_rate = decay
        self.num_epochs = num_epochs
        self.cur_epoch = -1

        self.data_dict = data_dict

        self.imgsize = imgsize
        self.class_label = class_label
        self.RGBmean = RGBmean
        self.RGBstdv = RGBstdv

        self.record = []
        self.epoch = 0
        self.print_freq = print_freq
        self.save_step = save_step
        self.loss = loss
        self.losses = AverageMeter()
        self.top1, self.test_top1 = AverageMeter(), AverageMeter()
        self.top5, self.test_top5 = AverageMeter(), AverageMeter()

        # model parameter
        self.scale = scale
        self.pool_type = pool_type
        self.inter_distance = inter_distance
        if not self.setsys(): print('system error'); return

        self.logger = logger
        self.wandb = wandb
        
        self.traindata = traindata
        self.testdata = testdata

        # self.loadData()
        self.setModel(config)

        self.old_model = None
        if self.dset_name == "cifar10" or self.dset_name == 'cifar100':
            self.local_epochs = args.img_local_epochs
        else:
            self.local_epochs = args.txt_local_epochs
        self.local_epoch = 0

        self.global_test_set = global_test_set
    
        self.cumulative_prec1 = 0
        self.cumulative_prec5 = 0
        self.total_samples = 0

    def run(self, agg_proto, agg_model,  img_enc, txt_enc):
        self.lr_scheduler(self.cur_epoch)
        
            
        
        # self.old_model = copy.deepcopy(self.model)
        # self.old_model.eval()
        # self.old_model.cuda()
        
        # 根据当前的全局轮次更新学习率
        self.model.to(self.gpuid)
        # 将本地模型的映射头设置为聚合之后的全局映射头
        if self.cur_epoch>0:
            if agg_model['image']:
                agg_model_image_state_dict = {key: value for key, value in agg_model['image'].items() if key.startswith('fc')}
                self.model.load_state_dict(agg_model_image_state_dict)
            else:
                agg_model_text_state_dict = {key: value for key, value in agg_model['text'].items() if key.startswith('fc')}
                self.model.load_state_dict(agg_model_text_state_dict)
        else:
            if agg_model['image']:
                self.model.fc.load_state_dict(agg_model['image'].fc.state_dict())
            else:
                self.model.fc.load_state_dict(agg_model['text'].fc.state_dict())
            # local training
        for i in range(self.local_epochs):
            if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
                model_path = f'./results/sm/model/img/Client{self.client_idx}-global_{self.cur_epoch}-model_Local{self.local_epoch}.pth'
            else:
                model_path = f'./results/sm/model/text/Client{self.client_idx}-global_{self.cur_epoch}-model_Local{self.local_epoch}.pth'
            if os.path.exists(model_path):
                self.logger.log(f"Model for client {self.client_idx} global epoch {self.cur_epoch}, local epoch {self.local_epoch} found. Loading...")
                self.model.load_state_dict(torch.load(model_path))
                self.model.to(self.gpuid)
                continue
            if self.logger is not None:
                self.logger.log(f"Client {self.client_idx} Local Training: Epoch {self.local_epoch}")
            self.tra(agg_proto,img_enc, txt_enc)
            self.local_epoch += 1
            
            torch.save(self.model.state_dict(), model_path)

                
            
        # 计算本地原型    
        embeddings = []
        labels = []
        
        self.model.eval()
        
        if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
            prototypes = {'image':[]}
            with torch.no_grad():
                for inputs, target in self.traindata:
                    output, _ = self.model(img_enc(inputs))  # 获取模型输出的嵌入
                    embeddings.append(output)
                    labels.append(target)
        elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
            prototypes = {'text':[]}
            with torch.no_grad():
                for inputs, target in self.traindata:
                    output, _ = self.model(txt_enc(inputs))  # 获取模型输出的嵌入
                    embeddings.append(output)
                    labels.append(target)

        # 合并所有的嵌入和标签
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
        
        for class_id in range(self.config.project_head.class_num):
        # 获取属于该类别的嵌入
            class_embeddings = embeddings[labels == class_id]
        
            if class_embeddings.size(0) > 0:
                if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
                    # 计算该类别的原型（均值）
                    prototypes['image'].append(class_embeddings.mean(dim=0))
                elif  self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    prototypes['text'].append(class_embeddings.mean(dim=0))

        # return embeddings, labels

        self.test(img_enc, txt_enc)

        
        self.model.cpu()
        # self.old_model.cpu()

        # del self.old_model
        import gc
        gc.collect()
        if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
            return {'image':self.model.fc.state_dict()}, prototypes
        elif  self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
            return {'text':self.model.fc.state_dict()}, prototypes
    ##################################################
    # step 0: System check and predefine function
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        return True



    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self,config):
        if self.logger is not None:
            self.logger.log(f'Setting {self.type} model {self.client_idx}')
        if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
            self.model = ImageNet(config.project_head.input_embedding, config.project_head.out_embedding, config.project_head.class_num, config.project_head.norm, config.project_head.hid_num, config.project_head.c_hid)
            self.criterion = nn.CrossEntropyLoss()
            params = self.model.parameters()
        elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
            self.model = TextNet(config.project_head.input_embedding, config.project_head.out_embedding, config.project_head.class_num, config.project_head.norm, config.project_head.hid_num, config.project_head.c_hid)
            self.criterion = nn.CrossEntropyLoss()
            params = self.model.parameters()
        self.center_criterion = nn.MSELoss()
        self.optimizer = optim.SGD(params, lr=self.init_lr,
                                   momentum=0.9, weight_decay=0.00005)
        return

    def lr_scheduler(self, epoch):
        if epoch >= 0.5 * self.num_epochs and not self.decay_time[0]:
            self.decay_time[0] = True
            lr = self.init_lr * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch >= 0.8 * self.num_epochs and not self.decay_time[1]:
            self.decay_time[1] = True
            lr = self.init_lr * self.decay_rate * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self, agg_proto, img_enc, txt_enc):
        def printnreset(name):
            self.logger.log('Epoch: [{0}] {1}\t'
                            # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {2} \t'
                            'Prec@5 {3} '.format(
                self.local_epoch, name, self.cumulative_prec1*100.0/self.total_samples, self.cumulative_prec5*100.0/self.total_samples ))

             # 累积准确率
            self.cumulative_prec1 = 0
            self.cumulative_prec5 = 0
            self.total_samples = 0

        # Set model to training mode
        self.model.train()
        # import nltk
        # nltk.download('punkt_tab')
        
        for i, data in enumerate(self.traindata):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    # labels_bt = labels_bt.to(torch.long)
                    inputs_var = inputs_bt.to(self.gpuid)
                    labels_var = labels_bt.to(self.gpuid)
                    
                    emb =  img_enc(inputs_var)
                    out, out_norm = self.model(emb)
                    
                    image_logits = self.model.classify(out_norm)
                    
                    loss_o = self.criterion(image_logits, labels_var) 
                    if agg_proto:
                        loss_p = PrototypeLoss(agg_proto)
                        l_p = loss_p(out_norm, out_norm)
                        loss = loss_o + l_p
                    else:
                        loss = loss_o
                        
                    # 计算 top-1 和 top-5 准确率
                    _, preds = torch.topk(image_logits, k=5, dim=1)  # 获取 top-5 预测
                    correct = preds.eq(labels_var.view(-1, 1).expand_as(preds))  # 检查预测是否正确

                    # Top-1 准确率
                    prec1 = correct[:, 0].sum().item()  # top-1
                    # Top-5 准确率
                    prec5 = correct.sum().item()  # top-5

                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    inputs_bt, labels_bt = data
                    # caplens = caplens.to(self.gpuid)
                    # if isinstance(inputs_bt, torch.Tensor):
                    #     inputs_bt = [str(x) for x in inputs_bt]
                    # inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                    #                            (inputs_bt, labels_bt))
                    # inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))
                    # inputs_bt = inputs_bt.to(self.gpuid)
                    # labels_bt = labels_bt.to(self.gpuid)
                    labels_bt = labels_bt.to(self.gpuid)
                    emb = txt_enc(inputs_bt)
                    out, out_norm = self.model(emb)
                    
                    text_logits = self.model.classify(out_norm)
                    
                    loss_o = self.criterion(text_logits, labels_bt) 
                    if agg_proto:
                        loss_p = PrototypeLoss(agg_proto)
                        loss = loss_o + loss_p
                    else:
                        loss = loss_o
                    # 计算 top-1 和 top-2 准确率
                    _, preds = torch.topk(text_logits, k=2, dim=1)  # 获取 top-2 预测
                    correct = preds.eq(labels_bt.view(-1, 1).expand_as(preds))  # 检查预测是否正确

                    # Top-1 准确率
                    prec1 = correct[:, 0].sum().item()  # top-1
                    # Top-2 准确率
                    prec5 = correct.sum().item()  # top-4

                 # 累积准确率
                self.cumulative_prec1 += prec1
                self.cumulative_prec5 += prec5
                self.total_samples += len(inputs_bt)
                

                # if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
                #     prec1, prec5 = accuracy(image_logits, labels_bt, topk=(1, 5))
                # elif self.dset_name == 'AG_NEWS':
                #     prec1, prec5 = accuracy(text_logits, labels_bt, topk=(1, 4))
                # elif self.dset_name == 'YelpReviewPolarity':
                #     prec1, prec5 = accuracy(text_logits, labels_bt, topk=(1, 2))
                # self.top1.update(prec1[0], inputs_bt.size(0))
                # self.top5.update(prec5[0], inputs_bt.size(0))

                # self.losses.update(loss.item(), inputs_bt.size(0))
                loss.backward()
                self.optimizer.step()
            if is_test:
                break

        printnreset(self.dset_name)
       


    def test(self,img_enc, txt_enc):
        def printnreset(name):
            self.logger.log('TEST Epoch: [{0}] {1}\t'
                            # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {2} \t'
                            'Prec@5 {3} '.format(
                self.local_epoch, name, self.cumulative_prec1*100.0/self.total_samples, self.cumulative_prec5*100.0/self.total_samples ))

             # 累积准确率
            self.cumulative_prec1 = 0
            self.cumulative_prec5 = 0
            self.total_samples = 0

        # test_d = torch.utils.data.DataLoader(self.testdata, batch_size=self.config.test.batch)
        with torch.no_grad():
            # for data in test_d:
            for data in self.testdata:
                if self.dset_name == 'cifar100' or self.dset_name == 'cifar10':
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    labels_bt = labels_bt.to(self.gpuid)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    
                    out, out_norm= self.model(img_enc(inputs_var))
                    image_logits = self.model.classify(out_norm)
                    
                    # 计算 top-1 和 top-5 准确率
                    _, preds = torch.topk(image_logits, k=5, dim=1)  # 获取 top-5 预测
                    correct = preds.eq(labels_bt.view(-1, 1).expand_as(preds))  # 检查预测是否正确

                    # Top-1 准确率
                    prec1 = correct[:, 0].sum().item()  # top-1
                    # Top-5 准确率
                    prec5 = correct.sum().item()  # top-5
                    
                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    inputs_bt, labels_bt = data
                    labels_bt = labels_bt.to(self.gpuid)
                    out, out_norm = self.model(txt_enc(inputs_bt))
                    text_logits = self.model.classify(out_norm)
                    
                # 计算 top-1 和 top-4 准确率
                    _, preds = torch.topk(text_logits, k=2, dim=1)  # 获取 top-4 预测
                    correct = preds.eq(labels_bt.view(-1, 1).expand_as(preds))  # 检查预测是否正确

                    # Top-1 准确率
                    prec1 = correct[:, 0].sum().item() # top-1
                    # Top-2 准确率
                    prec5 = correct.sum().item()  # top-4

                # 累积准确率
                self.cumulative_prec1 += prec1 
                self.cumulative_prec5 += prec5 
                self.total_samples += len(inputs_bt)

                # # on_hot vector
                # labels_var_one_hot = to_one_hot(labels_var, n_dims=self.classSize)
                # # inter_class_distance
                # fvec = fvec - self.inter_distance * labels_var_one_hot.to(self.gpuid)
                # if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                #     prec1, prec5 = accuracy(image_logits, labels_bt, topk=(1, 5))
                # elif self.dset_name == 'AG_NEWS':
                #     prec1, prec5 = accuracy(text_logits, labels_bt, topk=(1, 4))
                # elif self.dset_name == 'YelpReviewPolarity':
                #     prec1, prec5 = accuracy(text_logits, labels_bt, topk=(1, 2))
                # self.test_top1.update(prec1[0], inputs_bt.size(0))
                # self.test_top5.update(prec5[0], inputs_bt.size(0))

        printnreset(self.dset_name)
        # self.model.train()



    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O2')

    def __getattr__(self, k):
        if k.startwith("__"):
            raise AttributeError
