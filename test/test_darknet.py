import unittest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from typing import Tuple
from lumen.models.darknet import Darknet53, CSPDarknet53

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_dataset(path: str = '/home/pumetu/data/', dataset: str = 'mnist', transform: bool = False, batch_size: int = 32):
    if dataset == 'mnist':
        path += 'mnist'
        if transform:
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((224, 224), antialias=True), 
                transforms.Normalize((0.1307,),(0.3081,))
                ])
        train_dataset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
    elif dataset == 'cifar100':
        pass

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

class Reshape(nn.Module):
    def __init__(self, shape: Tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

def train(model, dataloader, optimizer, epoch, lossfn, log_interval):
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device) 
            out = model(images)['c5'] 
            loss = lossfn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx*(len(images))}/{len(dataloader.dataset)} Loss: {loss.item():.2f}]')

def test(model, dataloader, lossfn):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            out = model(images)['c5']
            loss = lossfn(out, labels)
            pred = out.data.max(1, keepdim=True)[1] 
            correct += pred.eq(labels.data.view_as(pred)).sum()
    loss /= len(dataloader.dataset) 
    print(f'Test Average Loss: {loss:.2f}, Accuracy {correct}/{len(dataloader.dataset)} ({100*correct/len(dataloader.dataset)})')
        
class TestDarknet(unittest.TestCase):
    def test_darknet_forward(self):
        model = Darknet53(in_channels=3, output=('c1', 'c2', 'c3', 'c4', 'c5'))
        x = torch.randn(1, 3, 416, 416)
        out = model(x)
        assert out['c1'].shape == (1, 64, 208, 208)
        assert out['c2'].shape == (1, 128, 104, 104)
        assert out['c3'].shape == (1, 256, 52, 52)
        assert out['c4'].shape == (1, 512, 26, 26)
        assert out['c5'].shape == (1, 1024, 13, 13)

    def test_cspdarknet_forward(self):
        model = CSPDarknet53(in_channels=3, output=('c2', 'c3', 'c4', 'c5'))
        x = torch.randn(1, 3, 416, 416)
        out = model(x)
        assert out['c2'].shape == (1, 128, 104, 104)
        assert out['c3'].shape == (1, 256, 52, 52)
        assert out['c4'].shape == (1, 512, 26, 26)
        assert out['c5'].shape == (1, 1024, 13, 13)

    def test_mnist_darknet(self):
        train_dataloader, test_dataloader = load_dataset(dataset='mnist', transform=True, batch_size=32)
        model = Darknet53(in_channels=1, output=('c5')) 
        mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Reshape((-1, 1024)), 
            nn.Linear(1024, 10)
        )
        model.c5 = model.c5.append(mlp)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lossfn = nn.CrossEntropyLoss()
        for epoch in range(1, 2):
            train(model, train_dataloader, optimizer, epoch, lossfn, log_interval=5000)
            test(model, test_dataloader, lossfn) 

    def test_mnist_cspdarknet(self):
        train_dataloader, test_dataloader = load_dataset(dataset='mnist', transform=True, batch_size=32)
        model = CSPDarknet53(in_channels=1, output=('c5'))
        mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Reshape((-1, 1024)),
            nn.Linear(1024, 10)
        )
        model.c5 = model.c5.append(mlp)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lossfn = nn.CrossEntropyLoss()
        for epoch in range(1, 2):
            train(model, train_dataloader, optimizer, epoch, lossfn, log_interval=5000)
            test(model, test_dataloader, lossfn)
    
    def test_depthwise_speed(self):
        x = torch.randn(1, 3, 416, 416).to(device)
        # Depthwise Separable
        start_time = time.time()
        dw_model = CSPDarknet53(in_channels=3, depthwise=True, output=('c5')).to(device)
        dw_out = dw_model(x)
        dw_time = time.time() - start_time
        dw_params = sum(p.numel() for p in dw_model.parameters())

        # Normal
        start_time = time.time()
        model = Darknet53(in_channels=3, output=('c5')).to(device)
        out = model(x)
        n_time = time.time() - start_time
        n_params = sum(p.numel() for p in model.parameters())

        print(f'dw: {dw_time}, n: {n_time}')
        print(f'dw params: {dw_params}, n_params: {n_params}')

if __name__ == "__main__":
    unittest.main()

