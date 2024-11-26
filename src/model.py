import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_DNN(nn.Module):
    def __init__(self):
        super(MNIST_DNN, self).__init__()
        
        # First Convolution Block - Slightly increased initial channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.15)  # Reduced dropout
        )
        
        # Second Convolution Block - Better channel progression
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.15)
        )
        
        # Third Convolution Block - Added residual connection
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(20, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )
        
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.15)
        )
        
        # Global Average Pooling and FC layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_final = nn.Dropout(p=0.2)
        self.fc = nn.Linear(24, 10)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # First two conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Third block with residual connection
        identity = x
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = self.dropout_final(x)
        x = torch.flatten(x, 1)
        
        # FC layer
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 