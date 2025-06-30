import torch
from torch import nn
import math

class LoRAMoE(nn.Module):
    def __init__(self, num_experts, dim, rank):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.rank = rank
        self.alpha = rank*2
        self.expert_w_a = nn.Parameter(torch.Tensor(num_experts, rank, dim))
        self.expert_w_b = nn.Parameter(torch.Tensor(num_experts, dim, rank))
        self.router = nn.Linear(dim, num_experts, bias=False)
    def forward(self, x):
        routing = torch.nn.functional.softmax(self.router(x), dim=-1)
        out_a = torch.einsum('...d, erd -> ...er', x, self.expert_w_a)
        out_b = torch.einsum('...er, edr -> ...ed', out_a, self.expert_w_b)
        out = torch.einsum('...ed, ...e -> ...d', out_b, routing)
        return out * (self.alpha / self.rank)
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.expert_w_a, a=math.sqrt(5))
        nn.init.zeros_(self.expert_w_b)
