import argparse

def args_parser():
    
    parser = argparse.ArgumentParser()
    # General DL parameters
    parser.add_argument("--net_name", type = str, default="ViT-small",  help="Basic Name of this run with detailed network-architecture selection. ")
    parser.add_argument("--FL_platform", type = str, default="ViT-FedAVG", choices=[ "Swin-FedAVG", "ViT-FedAVG", "Swin-FedAVG", "EfficientNet-FedAVG", "ResNet-FedAVG"],  help="Choose of different FL platform. ")
    parser.add_argument("--dataset", choices=["cifar10", "Retina"], default="cifar10", help="Which dataset.")
    parser.add_argument("--data_path", type=str, default='./data/', help="Where is dataset located.")

    parser.add_argument("--save_model_flag",  action='store_true', default=False,  help="Save the best model for each client.")
    parser.add_argument("--cfg",  type=str, default="configs/swin_tiny_patch4_window7_224.yaml", metavar="FILE", help='path to args file for Swin-FL',)

    parser.add_argument('--Pretrained', action='store_true', default=True, help="Whether use pretrained or not")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/swin_tiny_patch4_window7_224.pth", help="Where to search for pretrained ViT models. [ViT-B_16.npz,  imagenet21k+imagenet2012_R50+ViT-B_16.npz]")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Ways for optimization.")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply some. 0 for SGD and 0.05 for AdamW in paper")
    parser.add_argument('--grad_clip', action='store_true', default=True, help="whether gradient clip to 1 or not")

    parser.add_argument("--img_size", default=224, type=int, help="Final train resolution")
    parser.add_argument("--batch_size", default=32, type=int,  help="Local batch size for training.")
    parser.add_argument("--gpu_ids", type=str, default='1', help="gpu ids: e.g. 0  0,1,2")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization") #99999

    ## section 2:  DL learning rate related
    parser.add_argument("--decay_type", choices=["cosine", "linear", "step"], default="cosine",  help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Step of training to perform learning rate warmup for if set for cosine and linear deacy.")
    parser.add_argument("--step_size", default=30, type=int, help="Period of learning rate decay for step size learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # parser.add_argument("--learning_rate", default=3e-2, type=float, choices=[5e-4, 3e-2, 1e-3],  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    # 1e-5 for ViT central

    ## FL related parameters
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,  help="Total communication rounds")
    parser.add_argument("--num_local_clients", default=-1, choices=[10, -1], type=int, help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str, choices=["split_1", "split_2", "split_3", "real", "central"], default="split_1", help="Which data partitions to use")
    
    args = parser.parse_args()
    
    return args