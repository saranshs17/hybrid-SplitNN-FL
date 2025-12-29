from typing import Tuple
import torch
from ..models.definitions import IoTFeatureExtractor

class IoTClient:
    def __init__(self, client_id: str, dataloader, device='cpu'):
        self.client_id = client_id
        self.model = IoTFeatureExtractor().to(device)
        self.data = dataloader
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        self.device = device

    def forward_pass(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data, labels = next(iter(self.data))
        data = data.to(self.device)
        labels = labels.to(self.device)
        smashed = self.model(data)
        return smashed.detach().requires_grad_(), labels

    def backward_pass(self, grad_smashed: torch.Tensor):
        pass
