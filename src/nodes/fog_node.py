import torch
from ..models.definitions import FogContextLearner

class FogNode:
    def __init__(self, fog_id: int, device='cpu'):
        self.fog_id = fog_id
        self.model = FogContextLearner().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = device

    def forward_pass(self, smashed):
        smashed = smashed.to(self.device)
        return self.model(smashed)

    def backward_and_update(self, grad_from_cloud):
        self.optimizer.zero_grad()
        pass
