import torch.nn as nn
from torchvision import models
from torch.nn.functional import sigmoid

class Network(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        
        x = self.model(x)
        #x = sigmoid(x)

        return x