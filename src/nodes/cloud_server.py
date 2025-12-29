import torch
import torch.nn.functional as F
from ..models.definitions import CloudGlobalModel

class CloudServer:
    def __init__(self, device='cpu'):
        self.model = CloudGlobalModel().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = device

    def infer_and_loss(self, regional_features, labels):
        regional_features = regional_features.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(regional_features)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
