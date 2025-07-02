import torch
from torch import nn
import math

class MoERouter(nn.Module):
    def __init__(self, num_experts, dim):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.router = nn.Linear(dim, num_experts, bias=False)
    def forward(self, x):
        routing = torch.nn.functional.softmax(self.router(x), dim=-1)
        return routing

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        nn.init.trunc_normal_(
            self.router.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

class LoRAMoE(nn.Module):
    def __init__(self, num_experts, in_dim, out_dim, rank):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = rank*2
        self.expert_w_a = nn.Parameter(torch.Tensor(num_experts, rank, in_dim))
        self.expert_w_b = nn.Parameter(torch.Tensor(num_experts, out_dim, rank))
    def forward(self, x, routing):
        out_a = torch.einsum('...d, erd -> ...er', x, self.expert_w_a)
        out_b = torch.einsum('...er, edr -> ...ed', out_a, self.expert_w_b)
        out = torch.einsum('...ed, ...e -> ...d', out_b, routing) * (self.alpha / self.rank)
        return out
    def reset_parameters(self, init_std=None, factor=1.0):
        nn.init.kaiming_uniform_(self.expert_w_a, a=math.sqrt(5))
        nn.init.zeros_(self.expert_w_b)
