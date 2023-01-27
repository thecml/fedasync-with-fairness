import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(-1, 1, 28, 28)
    
    
class Model(nn.Sequential):
    def __init__(self):
        super().__init__(
            View(),
            nn.Conv2d(stride=1, kernel_size=3, out_channels=64, in_channels=1),
            nn.ReLU(),
            nn.Conv2d(stride=1, kernel_size=3, out_channels=64, in_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(stride=1, kernel_size=3, out_channels=128, in_channels=64),
            nn.ReLU(),
            nn.Conv2d(stride=1, kernel_size=3, out_channels=128, in_channels=128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(stride=1, kernel_size=3, out_channels=256, in_channels=128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(num_features=256),

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(out_features=512, in_features=256),
            nn.ReLU(),
            nn.Linear(out_features=10, in_features=512),
        )