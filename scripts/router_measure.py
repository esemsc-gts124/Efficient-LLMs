import torch
#import matplotlib.pyplot as plt

base_dir = "/home/jovyan/shared/muchane/lingua-repo/dumps/lora_moe/ffn_0"
bsize = 32
slen = 2048
layers = 12
exps = 8
#comp_tensor = torch.full((bsize*slen*layers, exps), 1/exps)
mean_diffs = []
for i in range(500, 10500, 500):
    x = torch.load(f"{base_dir}/routing_{i}.pt").to(torch.float32)
    #print(x.shape)
    x_ = x.reshape(bsize, slen, layers, exps)
    #x_ = torch.mean(x, dim=0)
    #print(diff.shape)
    #print(diff)
    diff = 1/2*torch.sum(torch.abs(x_ - 1/exps), dim=-1)
    print(diff.shape)
    worst_layer = torch.argmax(diff)
    print(worst_layer)
    print(x_[worst_layer])
    #diff = 1/2*torch.sum(torch.abs(x - 1/exps), dim=-1)
    #print(diff.shape)
    #diff = torch.cdist(x.reshape(bsize*slen*layers, exps), comp_tensor, p=1)
    #mean_diffs.append(diff.mean().item())
    mean_diffs.append(diff.mean().item())
    print(f"Mean difference for {i} tokens: {mean_diffs[-1]}")
