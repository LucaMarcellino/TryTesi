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
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np

args = parse_args()
device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"


seed = 0
g = torch.Generator()



wandb.init(
    # set the wandb project where this run will be logged
    
    entity = "fedvit",
    project=f"Centralized_ViT_{args.dataset}",
    
    # track hyperparameters and run metadata
    config={
    "pretrained": args.pre_trained,
    "momentum":args.momentum,
    "weight decay": args.wd,
    "learning_rate": args.lr,
    "architecture": "ViT",
    "dataset": args.dataset,
    "epochs": args.epochs,
    }
)
wandb.run.name = f"Centralized_ViT_Pretrained={args.pre_trained}_optimizer={args.opt}_lr={args.lr}_mom={args.momentum}_epochs={args.epochs}_wd={args.wd}"

#INPUT PARAMETERS

opt = args.opt
lr = args.lr
momentum = args.momentum
wd = args.wd
epochs = args.epochs

trainset,testset = get_datasets(args.dataset)
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
print("Pretrained : {}".format(args.pre_trained))
print("Dataset : {}".format(args.dataset))
print("Optimizer : {}".format(opt))
print("Learning Rate : {}".format(lr))
print("Momentum: {}".format(momentum))
print("Weight_decay : {}".format(wd))


make_it_reproducible(seed)
g.manual_seed(seed)

trainloader = torch.utils.data.DataLoader(trainset,
                                    batch_size = 1,#args.batch_size,
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

images, _ = next(iter(trainloader))


net = ClientModel(device = device, pretrained=0, num_classes=10)
optimizer = optim.SGD(net.parameters(), lr , momentum , wd )
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 30, 40], gamma = 0.33)
net.train()
with torch.no_grad():
    output, attn_weights = net(images)




def save_attention_image(image, attention_weights, output_path):
    # Normalize attention weights between 0 and 1
    #normalized_weights = (attention_weights - np.min(attention_weights)) / (
    #    np.max(attention_weights) - np.min(attention_weights)
    #)

    # Create a figure with two subplots: original image and attention map
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original image
    image = image.squeeze().detach().numpy().transpose(1, 2, 0)

# Denormalize the image from [-1, 1] to [0, 1]
    output_image = (image + 1) / 2.0
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Plot the attention map using the normalized weights
    axs[1].imshow(attention_weights, cmap='hot')
    axs[1].axis('off')
    axs[1].set_title('Attention Map')

    # Save the figure to the specified output path
    plt.savefig(output_path, bbox_inches='tight')

# Example usage
output_path = 'try.png'  # Replace with the desired output file path

save_attention_image(images, attn_weights[1:], output_path)
#print(output.size())
#print(len(attn_weights))

"""
mean_attention_weights = torch.mean(attn_weights.squeeze(), dim=(1, 2))

# Flatten the mean attention weights for easier sorting
flatten_weights = mean_attention_weights.view(-1)

# Sort the flattened attention weights in descending order
sorted_indices = torch.argsort(flatten_weights, descending=True)

# Plot the top 16 most important patches
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
for i in range(4):
    for j in range(4):
        patch_idx = sorted_indices[i * 4 + j]
        patch = images[0][:, patch_idx // 8 * 4: (patch_idx // 8 + 1) * 4, patch_idx % 8 * 4: (patch_idx % 8 + 1) * 4]
        ax[i, j].imshow(patch.permute(1, 2, 0))
        ax[i, j].axis('off')
        ax[i, j].set_title(f'Patch {patch_idx.item()}')

plt.tight_layout()
plt.savefig('two.png')
plt.show()
"""





"""
# Convert tensor to NumPy array and transpose dimensions
first_image = first_image.permute(1, 2, 0).numpy()

# Display the image
plt.imshow(first_image)
plt.axis('off')  # Turn off axis labels
plt.savefig('one.png')
plt.show()
"""

"""
if args.pre_trained == 0:
    if args.dataset == "cifar10":
        net = ClientModel(device = device, pretrained=0, num_classes=10)
    elif args.dataset == "cifar100":
        net = ClientModel(device = device, pretrained=0, num_classes=100)

elif args.pre_trained == 1:
    net = timm.create_model('vit_small_patch16_224', pretrained=True)
    if args.dataset == "cifar10":
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.dataset == "cifar100":
        net.head = nn.Linear(net.head.in_features, 100)
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
    
    wandb.log({"Test_accuracy": test_accuracy, "Test_loss": test_loss_avg, "Train_loss": train_loss_avg})

    
    print("Epochs = {} | Train loss = {} | Test loss = {} | Test accuracy = {}".format(epoch + 1,train_loss_avg, test_loss_avg, test_accuracy))
    
    output_metrics["Epoch"].append(epoch + 1)
    output_metrics["Lr"].append(lr)
    output_metrics["Momentum"].append(momentum)
    output_metrics["Weight_decay"].append(wd)
    output_metrics["Train_Loss"].append(train_loss_avg)
    output_metrics["Test_Loss"].append(test_loss_avg)
    output_metrics["Test_Accuracy"].append(test_accuracy)

    scheduler.step()


wandb.finish()
output_data = pd.DataFrame(output_metrics)
output_data.to_csv(f"Centralized_ViT_Pretrained:{args.pre_trained}_optimizer:{args.opt}_lr:{args.lr}_mom:{args.momentum}_epochs:{args.epochs}_wd:{args.wd}.csv', index = False.csv", index = False)
"""







