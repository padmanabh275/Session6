import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolution block
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Shortcut connection (identity mapping)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add identity
        out += identity
        out = F.relu(out)
        
        # Apply dropout after addition and activation
        out = self.dropout(out)
        return out

class MNIST_DNN(nn.Module):
    def __init__(self):
        super(MNIST_DNN, self).__init__()
        
        # Initial conv layer (similar to ResNet)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers (reduced channels for MNIST)
        self.layer1 = self._make_layer(32, 32, 2)      # Similar to ResNet18 layer1
        self.layer2 = self._make_layer(32, 64, 2, 2)   # Similar to ResNet18 layer2
        self.layer3 = self._make_layer(64, 128, 2, 2)  # Similar to ResNet18 layer3
        self.layer4 = self._make_layer(128, 256, 2, 2) # Similar to ResNet18 layer4
        
        # Global Average Pooling and FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_planes, planes, num_blocks, stride=1):
        # Create layers with strides (similar to ResNet)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Weight initialization (similar to ResNet paper)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution (similar to ResNet)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling and classification
        x = self.avgpool(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 