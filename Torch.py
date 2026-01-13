import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307,0.3801)])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

print("训练集样本数:",len(train_dataset))
print("测试集样本数:",len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=64, shuffle=False)

class DNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10),
        )

        def forward(self,x):
            return self.net(x)
