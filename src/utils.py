from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.CIFAR10(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return loader