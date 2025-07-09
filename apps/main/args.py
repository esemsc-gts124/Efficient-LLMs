# args.py
from dataclasses import dataclass, field
from typing import Optional, List
from lingua.transformer import BaseTransformerArgs


@dataclass
class VocabArgs:
    d_emb: int = 32
    factorise: bool = False
    d_factorised: Optional[int] = None
    proj_out: Optional[bool] = None # use an untied projector or
    # have last layer(s) inverse the ProjectUpLayers


@dataclass
class ProjectUpLayerArgs:
    d_attn_val: Optional[int] = None
    d_attn_kq: Optional[int] = None
    d_attn_out: Optional[int] = None
    attn_proj_rand: bool = False
    d_ffn: Optional[int] = None
    ffn_proj_rand: bool = False
    in_dim: int = 0


@dataclass
class LMTransformerArgs(BaseTransformerArgs):
    seed: int = 42
    rank: int = -1
    vocab_size: int = -1
    weight_tying: bool = False
    sliding_window: Optional[int] = None
    factorised_vocab: VocabArgs = field(default_factory=VocabArgs)
    project_layers: Optional[List[ProjectUpLayerArgs]] = None
