import torch
import torch.nn as nn

class IoTFeatureExtractor(nn.Module):

    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class FogContextLearner(nn.Module):

    def __init__(self, input_channels: int = 32):
        super().__init__()
        self.mid_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        return self.mid_layers(smashed_data)


class CloudGlobalModel(nn.Module):

    def __init__(self, input_dim: int = 64 * 4 * 4, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, regional_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(regional_features)
