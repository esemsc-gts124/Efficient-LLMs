from .transformer import LMTransformer
from .args import LMTransformerArgs, VocabArgs, ProjectLayerArgs
from .train import get_num_params
def main():
    vocab_args = VocabArgs(
        d_emb=16,
        factorise=True,
        d_factorised=384,
        proj_out=True
    )

    project_layer_args = [ProjectLayerArgs(
        d_attn_kq=None,
        d_attn_val=None,
        d_attn_out=None,
        d_ffn=384
    )]

    model_args = LMTransformerArgs(
        dim=384,
        ffn_dim_multiplier=0.75,
        n_layers=20,
        n_heads=16,
        n_kv_heads=8,
        weight_tying=True,
        vocab_size=128256,
        factorised_vocab=vocab_args,
        project_up_layers=None
    )
    """
    model_args = LMTransformerArgs(
        dim=512,
        ffn_dim_multiplier=1,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        weight_tying=True,
        vocab_size=128256,
        factorised_vocab=vocab_args,
        project_up_layers=project_layer_args
    )
    """
    model = LMTransformer(model_args)
    num_params = get_num_params(model)
    print(f"Model Params: {num_params}")
    print(f"% Embedding: {model.tok_embeddings.weight.numel()/num_params:.2%}")
    for name, param in model.layers[0].attention.named_parameters():
        print(f"{name}: {param.shape}")
    #print(model_args.project_up_layers[0].in_dim)
if __name__ == "__main__":
    main()
