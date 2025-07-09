from .transformer import LMTransformer
from .args import LMTransformerArgs, VocabArgs, ProjectUpLayerArgs
from .train import get_num_params
def main():
    vocab_args = VocabArgs(
        d_emb=16,
        factorise=True,
        d_factorised=32,
        proj_out=True
    )

    project_layer_args = [ProjectUpLayerArgs(
        d_attn_kq=4,
        d_attn_val=8,
        d_attn_out=256,
        d_ffn=512
    )]

    model_args = LMTransformerArgs(
        dim=512,
        ffn_dim_multiplier=1,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        weight_tying=True,
        vocab_size=128256,
        factorised_vocab=vocab_args,
        project_layers=project_layer_args
    )

    model = LMTransformer(model_args)
    num_params = get_num_params(model)
    print(f"Model Params: {num_params}")
    print(f"% Embedding: {model.tok_embeddings.weight.numel()/num_params:.2%}")

if __name__ == "__main__":
    main()
