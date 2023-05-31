import argparse


DATASETS = ['cifar10', 'cifar100']
SERVER_ALGORITHMS = ['fedavg']
SERVER_OPTS = ['sgd', 'adam', 'adagrad', 'fedavgm']
SIM_TIMES = ['small', 'medium', 'large']

def parse_args():
    parser = argparse.ArgumentParser()
    
    #Random args
    parser.add_argument('--dataset',                    
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    default='cifar10')
    
    parser.add_argument('-seed',
                        type=int,
                        help='Choose the seed for reproducibility',
                        default=42)
    
    
    #GPU
    parser.add_argument('--device',
                    type=str,
                    default='cuda:1')
    
    
    #FEDERETED SETTINGS
    parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=10)
    parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        required=True)
    parser.add_argument('--algorithm',
                        help='algorithm used for server aggregation;',
                        choices=SERVER_ALGORITHMS,
                        default='fedavg')
    #parser.add_argument('--client-algorithm',
    #                    help='algorithm used on the client-side for regularization',
    #                    choices=CLIENT_ALGORITHMS,
    #                    default=None)
    parser.add_argument('--alpha',
                        help='alpha value to retrieve corresponding file',
                        type=float,
                        default=None)
    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting;',
                        type=int,
                        default=0)
    
    ## SERVER OPTMIZER ##
    parser.add_argument('--server-opt',
                        help='server optimizer;',
                        choices=SERVER_OPTS,
                        required=False)
    parser.add_argument('--server-lr',
                        help='learning rate for server optimizers;',
                        type=float,
                        required=False)
    parser.add_argument('--server-momentum',
                        help='momentum for server optimizers;',
                        type=float,
                        default=0,
                        required=False)
    
    ## CLIENT TRAINING ##
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                                        help='None for FedAvg, else fraction;',
                                        type=float,
                                        default=None)
    epoch_capability_group.add_argument('--num-epochs',
                                        help='number of epochs when clients train on data;',
                                        type=int,
                                        default=1)
    parser.add_argument('--lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=-1,
                        required=False)
    parser.add_argument('--weight-decay',
                        help='weight decay for local optimizers;',
                        type=float,
                        default=0,
                        required=False)
    parser.add_argument('--momentum',
                        help='Client momentum for optimizer',
                        type=float,
                        default=0)
    parser.add_argument('--mixup',
                        help='True if use mixup data augmentation at training time',
                        action='store_true',
                        default=False)
    parser.add_argument('--mixup-alpha',
                        help='Parameter alpha in mixup',
                        type=float,
                        default=1.0)
    parser.add_argument('--cutout',
                        help='apply cutout',
                        action='store_true',
                        default=False)
    
    ## ANALYSIS OPTIONS ##
    parser.add_argument('--metrics-name',
                        help='name for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('--metrics-dir',
                        help='dir for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('-t',
                        help='simulation time: small, medium, or large;',
                        type=str,
                        choices=SIM_TIMES,
                        default='large')
    
    
    ## LOAD CHECKPOINT AND RESTART OPTIONS ##
    parser.add_argument('-load',
                        action='store_true',
                        default=False)
    parser.add_argument('--wandb-run-id',
                        help='wandb run id for resuming run',
                        type=str,
                        default=None)
    parser.add_argument('-restart',
                        help='True if download model from wandb run but restart experiment with new wandb id',
                        action='store_true',
                        default=False)
    parser.add_argument('--restart-round',
                        help='Round to be restarted (default: last executed round)',
                        type=int,
                        default=None)
    
    
    return parser.parse_args()