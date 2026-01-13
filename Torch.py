import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

# 数据预处理
transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(0.1307, 0.3801)
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

print("训练集样本数:", len(train_dataset))
print("测试集样本数:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 修正变量名

# DNN (深度神经网络)
class DNN(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 dropout 防止过拟合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)  # 修正：使用 self.model 而不是 self.net

# CNN (卷积神经网络)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道1，输出32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 池化层，大小2x2
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 自适应平均池化，确保输出固定尺寸
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建并训练 DNN 模型
print("\n=== 训练 DNN 模型 ===")
dnn_model = DNN().to(device)
dnn_criterion = nn.CrossEntropyLoss()
dnn_optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)

for epoch in range(5):  # 训练5个epoch
    train_loss, train_acc = train_model(dnn_model, train_loader, dnn_criterion, dnn_optimizer, device)
    test_loss, test_acc = test_model(dnn_model, test_loader, dnn_criterion, device)
    print(f'Epoch [{epoch+1}/5], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

# 创建并训练 CNN 模型
print("\n=== 训练 CNN 模型 ===")
cnn_model = CNN().to(device)
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

for epoch in range(5):  # 训练5个epoch
    train_loss, train_acc = train_model(cnn_model, train_loader, cnn_criterion, cnn_optimizer, device)
    test_loss, test_acc = test_model(cnn_model, test_loader, cnn_criterion, device)
    print(f'Epoch [{epoch+1}/5], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

print("\n模型训练完成！")

# 评估两个模型的性能
print("\n=== 最终性能比较 ===")
dnn_test_loss, dnn_test_acc = test_model(dnn_model, test_loader, dnn_criterion, device)
cnn_test_loss, cnn_test_acc = test_model(cnn_model, test_loader, cnn_criterion, device)

print(f"DNN 模型准确率: {dnn_test_acc:.2f}%")
print(f"CNN 模型准确率: {cnn_test_acc:.2f}%")

# 保存模型
torch.save(dnn_model.state_dict(), 'dnn_model.pth')
torch.save(cnn_model.state_dict(), 'cnn_model.pth')
print("模型已保存!")