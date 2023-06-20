import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs',
                        type=int,
                        default=50)
    parser.add_argument('--lr',
                        type=float,
                        default=0.01)
    parser.add_argument('--opt',
                        type=str,
                        default='sgd')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5)
    parser.add_argument('--wd',
                        type=float,
                        default=0)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--pre_trained',
                        type=int,
                        help='0 no pretrained, 1 pretrained',
                        default= 1)
    parser.add_argument('--cuda',
                        type=int,
                        default=1)
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10')
    
    
    return parser.parse_args()