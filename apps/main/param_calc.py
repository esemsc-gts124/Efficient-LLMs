import torch
from dataclasses import dataclass
from typing import Optional

# Note: The following classes are copied from the provided dim_transformer.py and transformer.py
# to make this a standalone script. In practice, you can import them directly.

@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"
    max_seqlen: int = 1024

@dataclass
class LMTransformerArgs(BaseTransformerArgs):
    seed: int = 42
    rank: int = -1
    vocab_size: int = -1
    weight_tying: bool = False
    sliding_window: Optional[int] = None
    D_emb: int | None = None

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=False)

    def _precompute_freqs_cis(self):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        t = torch.arange(self.max_seqlen, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        cos, sin = freqs.cos(), freqs.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

    def reset_parameters(self):
        self.freqs_cis[...] = self._precompute_freqs_cis()

    def forward(self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None):
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]
        raise ValueError("Provide at least seqlen or tok_idx")

class Attention(torch.nn.Module):
    def __init__(self, dim: int, head_dim: int, n_heads: int, n_kv_heads: int, rope_theta: float):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.wq = torch.nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = torch.nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = torch.nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = torch.nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor, tok_idx: Optional[torch.Tensor] = None, mask=None, attn_impl="sdpa"):
        # Simplified forward for param count; actual impl not needed
        return x  # Dummy

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        for w in [self.wq, self.wk, self.wv]:
            torch.nn.init.trunc_normal_(w.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        torch.nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std / factor, a=-3*init_std, b=3*init_std)

class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.w1.in_features ** (-0.5))
        out_init_std = init_std or (self.w2.in_features ** (-0.5)) / factor
        for w in [self.w1, self.w3]:
            torch.nn.init.trunc_normal_(w.weight, mean=0.0, std=in_init_std, a=-3*in_init_std, b=3*in_init_std)
        torch.nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=out_init_std, a=-3*out_init_std, b=3*out_init_std)

class TransformerBlock(torch.nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        head_dim = args.head_dim or args.dim // args.n_heads
        n_heads = args.n_heads or args.dim // head_dim
        n_kv_heads = args.n_kv_heads or n_heads
        self.attention = Attention(args.dim, head_dim, n_heads, n_kv_heads, args.rope_theta)
        self.feed_forward = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freq_cis, tok_idx=None, mask=None, attn_impl="sdpa"):
        return x  # Dummy

class CustomFeedForward(FeedForward):
    def __init__(self, input_dim, output_dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__(dim=input_dim, hidden_dim=hidden_dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)
        self.w2 = torch.nn.Linear(self.w1.out_features, output_dim, bias=False)

class CustomTransformerBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        head_dim = args.dim // args.n_heads  # Assuming n_heads provided
        self.attention = Attention(input_dim, head_dim, args.n_heads, args.n_heads, args.rope_theta)
        self.attention_norm = RMSNorm(input_dim, eps=args.norm_eps)
        self.ffn = CustomFeedForward(input_dim, output_dim, 4 * input_dim, args.multiple_of, args.ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(input_dim, eps=args.norm_eps)
        self.residual_projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, freq_cis, tok_idx, mask, attn_impl):
        return x  # Dummy

class TiedLinear(torch.nn.Module):
    def __init__(self, tied_module: torch.nn.Module):
        super().__init__()
        self.tied_module = tied_module

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.linear(x, self.tied_module.weight)

class FactorisedTiedLinear(torch.nn.Module):
    def __init__(self, tok_embeddings1: torch.nn.Embedding, tok_embeddings2: torch.nn.Linear):
        super().__init__()
        self.tok_embeddings1 = tok_embeddings1
        self.tok_embeddings2 = tok_embeddings2

    def forward(self, x: torch.Tensor):
        intermediate = torch.matmul(x, self.tok_embeddings2.weight)
        logits = torch.matmul(intermediate, self.tok_embeddings1.weight.t())
        return logits

class BaseTransformer(torch.nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = args.init_std_factor
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(args.rope_theta, args.head_dim or args.dim // args.n_heads, args.max_seqlen)
        self.layers = torch.nn.ModuleList(TransformerBlock(args) for _ in range(args.n_layers))

    def forward(self, h, tok_idx=None, mask=None, attn_impl="sdpa"):
        return h  # Dummy

    def reset_parameters(self):
        self.rope_embeddings.reset_parameters()

class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        assert args.vocab_size > 0
        if args.rank > 0:
            self.use_factorised = True
            self.tok_embeddings1 = torch.nn.Embedding(args.vocab_size, args.rank)
            self.tok_embeddings2 = torch.nn.Linear(args.rank, args.D_emb or args.dim)
        else:
            self.use_factorised = False
            self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.D_emb or args.dim)
        self.first_block = CustomTransformerBlock(args.D_emb or args.dim, args.dim, args)
        self.layers = torch.nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers - 1)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        if args.weight_tying:
            if self.use_factorised:
                self.output = FactorisedTiedLinear(self.tok_embeddings1, self.tok_embeddings2)
            else:
                self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = torch.nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, token_values: torch.Tensor, target=None, tok_idx=None, mask=None, attn_impl="sdpa"):
        return token_values  # Dummy

    def reset_parameters(self, init_std=None):
        super().reset_parameters()
        self.norm.reset_parameters()
        self.first_block.attention_norm.reset_parameters()
        self.first_block.ffn_norm.reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        if self.use_factorised:
            torch.nn.init.trunc_normal_(self.tok_embeddings1.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
            torch.nn.init.trunc_normal_(self.tok_embeddings2.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        else:
            torch.nn.init.trunc_normal_(self.tok_embeddings.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        if not self.weight_tying:
            torch.nn.init.trunc_normal_(self.output.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        torch.nn.init.trunc_normal_(self.first_block.residual_projection.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
        torch.nn.init.trunc_normal_(self.first_block.ffn.w2.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)

# Model config from the provided example
args = LMTransformerArgs(
    dim=216,
    D_emb=80,
    n_layers=4,
    n_heads=4,
    vocab_size=128256,  # Meta Llama 3 tokenizer vocab size is 128,256<grok-card data-id="a76da5" data-type="citation_card"></grok-card><grok-card data-id="7e59c1" data-type="citation_card"></grok-card><grok-card data-id="205afe" data-type="citation_card"></grok-card>
    # Other defaults are used as per the dataclass
)

model = LMTransformer(args)

# Calculate total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")