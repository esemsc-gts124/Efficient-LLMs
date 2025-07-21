from .rrt import LMTransformer, LMTransformerArgs
from .train import get_num_params
def main():

    model_args = LMTransformerArgs(
        dim=512,
        ffn_dim_multiplier=1.125,
        n_layers=12,
        n_heads=16,
        n_kv_heads=8,
        weight_tying=True,
        vocab_size=128256,
        rank=24,
        layer_groups=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    )
    model = LMTransformer(model_args)
    num_params = get_num_params(model)
    print(f"Model Params: {num_params}")
    print(f"% Embedding: {model.tok_embeddings1.weight.numel()/num_params:.2%}")
    #for name, param in model.layers[0].attention.named_parameters():
    #    print(f"{name}: {param.shape}")
    #print(model_args.project_up_layers[0].in_dim)
if __name__ == "__main__":
    main()
