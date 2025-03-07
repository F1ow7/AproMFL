import hashlib
import munch

import torch
from src.eval import Evaluator
from src.utils.serialize_utils import object_loader
from src.optimizers import get_lr_scheduler,get_optimizer
from src.datasets.dataloader import prepare_f30k_dataloaders
try:
    from apex import amp
except ImportError:
    print('failed to import apex')
from src.utils.serialize_utils import torch_safe_load
from models.MMNet import MMNet
from criterions.probemb import MCSoftContrastiveLoss
from criterions.CMloss import CrossModalContrastiveLoss

def parse_config(config_path, cache_dir=None, pretrained_resnet_model_path=None, use_fp16=False):
    dict_config = object_loader(config_path)

    config = {}
    for config_key, subconfig in dict_config.items():
        if not isinstance(subconfig, dict):
            raise TypeError('unexpected type Key({}) Value({}) '
                            'All({})'.format(config_key, subconfig, config))

        for subconfig_key, subconfig_value in subconfig.items():
            if isinstance(subconfig_value, dict):
                raise ValueError('Only support two-depth configs. '
                                 'See README. All({})'.format(config))

        config[config_key] = munch.Munch(**subconfig)

    config = munch.Munch(**config)
    config.train.use_fp16 = use_fp16
    config.model.cache_dir = cache_dir
    config.model.pretrained_resnet_model_path = pretrained_resnet_model_path
    return config





class EngineBase(object):
    def __init__(self, args, config, logger, client=-1, dset_name="flickr30k", device='cuda',
                 vocab_path='./coco_vocab.pkl', mlp_local=False):
        
        self.dset_name = dset_name
        self.args = args
        self.config = config
        self.type = 'mm'
        self.device = device
        # self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.evaluator = Evaluator(eval_method=config.model.get('eval_method', 'matmul'),
                                       verbose=False,
                                       eval_device='cuda',
                                       n_crossfolds=5)

        self.logger = logger

        self.metadata = {}

        self.client_idx = client

        self.set_dset(self.dset_name, client)

        
        # self.word2idx = word2idx
        self.cluster_model, self.model = self.get_model(config) 
        self.set_criterion(self.get_criterion())
        
        params = []
        params += [param for param in self.model.parameters()
                   if param.requires_grad]
        self.set_optimizer(get_optimizer(config.optimizer.name,
                                         params,
                                         config.optimizer))
        c_params = []
        c_params += [c_param for c_param in self.cluster_model.parameters() 
                     if c_param.requires_grad]
        self.c_optimizer = get_optimizer(config.optimizer.name,
                                         c_params,
                                         config.optimizer)
        self.set_lr_scheduler(get_lr_scheduler(config.lr_scheduler.name,
                                               self.optimizer,
                                               config.lr_scheduler))
        self.evaluator.set_model(self.model)
        self.evaluator.set_criterion(self.criterion)

        # self.logger.log('Engine is created.')
        # self.logger.log(config)
        #
        # self.logger.update_tracker({'full_config': munch.unmunchify(config)}, keys=['full_config'])
        self.cur_epoch = 0
        self.old_model = None

        self.local_epochs = args.local_epochs
        self.local_epoch = 0

    def set_dset(self, dset_name, client=-1):
        if dset_name == "flickr30k":
            dataloaders = prepare_f30k_dataloaders(self.config.dataloader, client=client)
            self.train_loader = dataloaders['train']
            self.val_loader = dataloaders['test']
            # self.train_same_img = train_img
            # self.test_same_img = test_img
        else:
            assert False
        # return vocab.word2idx
    
    def get_model(self,config):
        if self.logger is not None:
            self.logger.log(f"Setting MM model {self.client_idx}")
        self.shared_model = MMNet(config)
        return MMNet(config), MMNet(config)
    def get_criterion(self):
        # return MCSoftContrastiveLoss(config)
        return CrossModalContrastiveLoss()
    def model_to_device(self):
        self.model.to(self.device)
        if self.criterion:
            self.criterion.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        # self.evaluator.set_logger(self.logger)

    #
    # def set_logger(self, logger):
    #     self.logger = logger

    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O2')

    @torch.no_grad()
    def evaluate(self, val_loaders, n_crossfolds=None, **kwargs):
        # if self.evaluator is None:
        #     self.logger.log('[Evaluate] Warning, no evaluator is defined. Skip evaluation')
        #     return

        self.model_to_device()
        self.model.eval()

        if not isinstance(val_loaders, dict):
            val_loaders = {'te': val_loaders}

        scores = {}
        n_crossfolds = self.evaluator.n_crossfolds if n_crossfolds is None else n_crossfolds
        for key, data_loader in val_loaders.items():
            # self.logger.log('Evaluating {}...'.format(key))
            _n_crossfolds = -1 if key == 'val' else n_crossfolds
            scores[key] = self.evaluator.evaluate(data_loader,
                                                  n_crossfolds=_n_crossfolds,
                                                  key=key,
                                                  n_images_per_crossfold=int(
                                                      data_loader.dataset.n_images / _n_crossfolds),
                                                  n_captions_per_crossfold=int(
                                                      len(data_loader.dataset) / _n_crossfolds),
                                                  **kwargs)
        return scores

    def save_models(self, save_to, metadata=None):
        state_dict = {
            'model': self.model.state_dict(),
            # 'criterion': self.criterion.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            # 'lr_scheduler': self.lr_scheduler.state_dict(),
            'config': munch.unmunchify(self.config),
            # 'word2idx': self.word2idx,
            'metadata': metadata,
        }
        torch.save(state_dict, save_to)
        new_meta = {}
        for k in metadata.keys():
            if k != 'code':
                new_meta[k] = metadata[k]
        # self.logger.log('state dict is saved to {}, metadata: {}'.format(
        #     save_to, json.dumps(new_meta, indent=4)))

    def load_models(self, state_dict_path, load_keys=None):
        with open(state_dict_path, 'rb') as fin:
            model_hash = hashlib.sha1(fin.read()).hexdigest()
            self.metadata['pretrain_hash'] = model_hash

        state_dict = torch.load(state_dict_path, map_location='cpu')

        if 'model' not in state_dict:
            torch_safe_load(self.model, state_dict, strict=False)
            return

        if not load_keys:
            load_keys = ['model', 'criterion', 'optimizer', 'lr_scheduler']
        for key in load_keys:
            try:
                torch_safe_load(getattr(self, key), state_dict[key])
            except RuntimeError as e:
                # self.logger.log('Unable to import state_dict, missing keys are found. {}'.format(e))
                torch_safe_load(getattr(self, key), state_dict[key], strict=False)
        # self.logger.log('state dict is loaded from {} (hash: {}), load_key ({})'.format(state_dict_path,
        #                                                                                 model_hash,
        #                                                                                 load_keys))

    def load_state_dict(self, state_dict_path, load_keys=None):
        state_dict = torch.load(state_dict_path)
        config = parse_config(state_dict['config'])
        self.create(config, state_dict['word2idx'])
        self.load_models(state_dict_path, load_keys)