import os
import argparse
from src.utils.helper import Helper as helper
import random
from models.clip import ImageEncoder, TextEncoder, clipEncoder
from models.project_head import ImageHead, TextHead
from torchvision import models
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch
from datetime import datetime

def init_wandb(args):
    """
  wandb will automatically save the log

  wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
  print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

  wandb.log({"test_accuracy": correct / total})

  # Save the model in the exchangeable ONNX format
  torch.onnx.export(model, images, "model.onnx")
  wandb.save("model.onnx")

  """

    import wandb
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    name = f"Test-{date}"

    wandb.init(
        project="MFL_proto_20",
        name=name,
        resume=None,
        # dir=os.path.join(args.exp_dir, args.name),
        config=args
    )

    return wandb


def args():
    parser.add_argument('--name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='./experiments/',
                        help='Locations to save different experimental runs.')
    parser.add_argument('--local_epochs', type=int, default=10)
    parser.add_argument('--img_local_epochs', type=int, default=5)
    parser.add_argument('--txt_local_epochs', type=int, default=5)
    parser.add_argument('--comm_rounds', type=int, default=5)

    parser.add_argument('--model', type=str, default='resnet34', help='Target model name (default: resnet34_8x)')
    parser.add_argument('--img_model_local', type=str, default='resnet10')
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000), metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--num_img_clients', type=int, default=2)
    parser.add_argument('--num_txt_clients', type=int, default=2)
    parser.add_argument('--num_mm_clients', type=int, default=2)

    parser.add_argument('--client_num_per_round', type=int, default=10)

    # === dataloader ===
    parser.add_argument('--image_data', type=str, default='cifar10', choices=['svhn', 'cifar10', 'cifar100'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default="/root/autodl-fs/data")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--alpha', type=float, default=0.5)

    # === optimization ===
    parser.add_argument('--server_lr', type=float, default=0.0002)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl', 'l1softmax'], )
    parser.add_argument('--scheduler', type=str, default='multistep',
                        choices=['multistep', 'cosine', 'exponential', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.05, 0.15, 0.3, 0.5, 0.75], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=0.1, help="Fractional decrease in lr")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    # === logs ===
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--disable_distill', action="store_true", default=False)

    parser.add_argument('--agg_method', type=str, default='con_w', help='representation aggregation method')
    parser.add_argument('--contrast_local_intra', action="store_true", default=False)
    parser.add_argument('--contrast_local_inter', action="store_true", default=False)

    parser.add_argument('--mlp_local', action="store_true", default=False)

    parser.add_argument('--kd_weight', type=float, default=0.3, help='coefficient of kd')
    parser.add_argument('--interintra_weight', type=float, default=0.5, help='coefficient of inter+intra')

    parser.add_argument('--loss_scale', action='store_true', default=False)
    parser.add_argument('--save_client', action='store_true', default=False)

    parser.add_argument('--data_local', action='store_true', default=False,
                        help='change data directory to ~/data_local')

    # parser.add_argument('--pub_data_num', type=int, default=50000, help='communication')
    parser.add_argument('--feature_dim', type=int, default=256)

    parser.add_argument('--not_bert', action='store_true', default=False, help="server bert, client not bert")


parser = argparse.ArgumentParser(description='Federated Learning')
args()
args = parser.parse_args()
global_proto = {}

if __name__ == "__main__":

    from src.MMFL import MMFL

    wandb = init_wandb(args)

   

    Algo = MMFL(args, wandb)

    args.save_dirs = helper.get_save_dirs(args.exp_dir, args.name)
    args.log_dir = args.save_dirs['logs']
    helper.set_seed(args.seed)
    print(args.seed)

    clip_enc = clipEncoder()

    Algo.load_data(args)
    
    global_init_model =[]
    
    # 初始化全局模型
    total_num = args.num_img_clients + args.num_txt_clients + args.num_mm_clients
    for i in range(total_num):
        if i < args.num_img_clients:
            mod = {'image': ImageHead(768,512,True,[1024,1024]), 'text': None}
            global_init_model.append(mod)
        elif i>= args.num_img_clients and i < (args.num_img_clients + args.num_txt_clients):
            mod = {'image': None, 'text': TextHead(768,512,True,[1024,1024])}
            global_init_model.append(mod)
        elif i>= (args.num_img_clients + args.num_txt_clients) and i < total_num:
            mod = {'image': ImageHead(768,512,True,[1024,1024]),'text':TextHead(768,512,True,[1024,1024])}
            global_init_model.append(mod)
    # print(global_init_model)
            
    for round_n in range(args.comm_rounds):
        proto, model = Algo.train(round_n, pre_global_proto = global_proto, pre_global_model = global_init_model, clip_enc=clip_enc)
        global_init_model = model 
        global_proto = proto
        print(f'Global Training {round_n}')
    # Algo.logger.log("Best:")
    Algo.engine.report_scores(step=args.comm_rounds,
                              scores=Algo.best_scores,
                              metadata=Algo.best_metadata,
                              prefix=Algo.engine.eval_prefix)