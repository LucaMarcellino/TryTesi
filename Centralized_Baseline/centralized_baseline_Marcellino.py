import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from models import ResNet50
from reproducibility import make_it_reproducible,seed_worker
from utils import get_datasets
from tqdm import tqdm
from ViT import ClientModel
from args import parse_args
import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 0
g = torch.Generator()




#INPUT PARAMETERS
args = parse_args()
opt = args.opt
lr = args.lr
momentum = args.momentum
wd = args.wd
epochs = args.epochs

trainset,testset = get_datasets()
criterion = nn.CrossEntropyLoss()

output_metrics = dict()
output_metrics["Epoch"] = list()
output_metrics["Lr"] = list()
output_metrics["Momentum"] = list()
output_metrics["Weight_decay"] = list()
output_metrics["Train_Loss"] = list()
output_metrics["Test_Loss"] = list()
output_metrics["Test_Accuracy"] = list()





print("Running test with:")
print("Optimizer : {}".format(opt))
print("Learning Rate : {}".format(lr))
print("Momentum: {}".format(momentum))
print("Weight_decay : {}".format(wd))


make_it_reproducible(seed)
g.manual_seed(seed)

trainloader = torch.utils.data.DataLoader(trainset,
                                    batch_size = args.batch_size,
                                    shuffle = True,
                                    num_workers = 2,
                                    worker_init_fn = seed_worker,
                                    generator = g)
testloader = torch.utils.data.DataLoader(testset,
                                batch_size = args.batch_size,
                                shuffle = False,
                                num_workers = 2,
                                worker_init_fn = seed_worker,
                                generator = g)


if args.pre_trained == 0:
    net = ClientModel(device = device)
elif args.pre_trained == 1:
    net = timm.create_model('vit_small_patch16_224', pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
net.to(device)

if opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr , momentum , wd )
elif opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr = lr , weight_decay = wd )

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 30, 40], gamma = 0.33)

for epoch in tqdm(range(epochs)):

    #TRAIN
    net.train()
    train_loss = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
    train_loss_avg = sum(train_loss) / len(train_loss)

    #TEST
    net.eval()
    test_loss = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            test_loss.append(loss.item())
            
        test_loss_avg = sum(test_loss) / len(test_loss)
        test_accuracy = correct / total

    if epoch + 1 % 5 == 0:
        print("Epochs = {} | Train loss = {} | Test loss = {} | Test accuracy = {}".format(epoch + 1,train_loss_avg, test_loss_avg, test_accuracy))
    
    output_metrics["Epoch"].append(epoch + 1)
    output_metrics["Lr"].append(lr)
    output_metrics["Momentum"].append(momentum)
    output_metrics["Weight_decay"].append(wd)
    output_metrics["Train_Loss"].append(train_loss_avg)
    output_metrics["Test_Loss"].append(test_loss_avg)
    output_metrics["Test_Accuracy"].append(test_accuracy)

    scheduler.step()
   


output_data = pd.DataFrame(output_metrics)
output_data.to_csv("Centralized_ViT_NoPretrained.csv", index = False)

                        







