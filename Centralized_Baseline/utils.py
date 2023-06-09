from torchvision import transforms, datasets

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
cifar100_mean =  [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]


def get_datasets(dataset, augmentation=False):
    if dataset == 'cifar10':
        trainset = datasets.CIFAR10("./data", train=True, download=True)
        testset = datasets.CIFAR10("./data", train=False, download=True)

        transform = transforms.Compose(
            [transforms.Resize((224,224)), 
            transforms.ToTensor(), 
            transforms.Normalize(cifar10_mean, cifar10_std),])
        if augmentation:
            transform_train = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.RandomCrop(224, padding=4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(cifar10_mean, cifar10_std)])
            trainset.transform = transform_train
        else:
            trainset.transform = transform
        testset.transform = transform
    
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100("./data", train=True, download=True)
        testset = datasets.CIFAR100("./data", train=False, download=True)

        transform = transforms.Compose(
            [transforms.Resize((224,224)), 
            transforms.ToTensor(), 
            transforms.Normalize(cifar100_mean, cifar100_std),])
        if augmentation:
            transform_train = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.RandomCrop(224, padding=4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(cifar100_mean, cifar100_std)])
            trainset.transform = transform_train
        else:
            trainset.transform = transform
        testset.transform = transform
    
    
    return trainset, testset