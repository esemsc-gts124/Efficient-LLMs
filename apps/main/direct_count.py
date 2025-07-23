from .transformer import LMTransformer, LMTransformerArgs
from lingua.transformer import LoRAMoEArgs
from .train import get_num_params
def main():


    test_model_args = LMTransformerArgs(
        dim=512,
        ffn_dim_multiplier=1.125,
        n_layers=12,
        n_heads=16,
        n_kv_heads=8,
        weight_tying=True,
        vocab_size=32000,
        lora_moe=LoRAMoEArgs(
            use=True,
            num_experts=32,
            rank=18
        )
    )

    baseline_model_args = LMTransformerArgs(
        dim=512,
        ffn_dim_multiplier=1.125,
        n_layers=10,
        n_heads=16,
        n_kv_heads=8,
        weight_tying=True,
        vocab_size=32000
    )
    test_model = LMTransformer(test_model_args)
    test_num_params = get_num_params(test_model)
    print(f"Test Model Params: {test_num_params}")
    print(f"% Embedding: {test_model.tok_embeddings.weight.numel()/test_num_params:.2%}")

    baseline_model = LMTransformer(baseline_model_args)
    baseline_num_params = get_num_params(baseline_model)
    print(f"Baseline Model Params: {baseline_num_params}")
    print(f"% Embedding: {baseline_model.tok_embeddings.weight.numel()/baseline_num_params:.2%}")

    print(f"% Difference: {(test_num_params - baseline_num_params)/baseline_num_params:.2%}")
    print(f"Absolute: {abs(test_num_params - baseline_num_params)}")
if __name__ == "__main__":
    main()
