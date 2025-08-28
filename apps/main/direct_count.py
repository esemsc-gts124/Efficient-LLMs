from .rrt import LMTransformer, LMTransformerArgs
from .train import get_num_params
from lingua.tokenizer import build_tokenizer
def main():

    # # --- Step 1: Define the path to your tokenizer ---
    # tokenizer_path = "/scratch_dgxl/gts124/RDS/home/Efficient-LLMs/llama3_tokenizer"

    # # --- Step 2: Load the tokenizer to get its actual size ---
    # print(f"Loading tokenizer from: {tokenizer_path}")
    # # This assumes your tokenizer loader function is named `get_tokenizer`
    # # and accepts 'path' and 'name' arguments, just like in your training script.
    # tokenizer = build_tokenizer(path=tokenizer_path, name="bytes")
    # actual_vocab_size = tokenizer.vocab_size
    # print(f"Tokenizer loaded. Actual vocab size is: {actual_vocab_size}")


    # model_args = LMTransformerArgs(
    #     dim=512,
    #     ffn_dim_multiplier=1.125,
    #     n_layers=12,
    #     n_heads=16,
    #     n_kv_heads=8,
    #     weight_tying=True,
    #     vocab_size=128256,
    #     rank=24,
    #     layer_groups=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    # )

    baseline = LMTransformerArgs(
        dim = 480,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    fhs = LMTransformerArgs(
        dim = 576,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1, 2, 3, 4, 5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 32
    )

    fws = LMTransformerArgs(
        dim = 1392,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    hierarch = LMTransformerArgs(
        dim = 768,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    pairs = LMTransformerArgs(
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 32
    )

    shs = LMTransformerArgs(
        dim = 576,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    r20 = LMTransformerArgs(
        dim = 640,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 20
    )

    r4 = LMTransformerArgs(
        dim = 632,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 4
    )

    r8 = LMTransformerArgs(
        dim = 624,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 8
    )

    r12 = LMTransformerArgs(
        dim = 616,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 12
    )

    r16_1 = LMTransformerArgs(
        dim = 648,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 16
    )

    r16_2 = LMTransformerArgs(
        dim = 616,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 16
    )

    r40 = LMTransformerArgs(
        dim = 600,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 40
    )


    bs = LMTransformerArgs(
        dim = 480,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 0
    )

    debug_shared = LMTransformerArgs(
        dim = 512,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 16,
        rank = -1,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        # lora_rank = 0
    )

    debug_baseline = LMTransformerArgs(
        dim = 512,
        ffn_dim_multiplier=1.125,
        n_layers = 2,
        n_heads = 16,
        rank = -1,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1]],
        # lora_rank = 0
    )

    george_debug_shared = LMTransformerArgs(
        dim = 512,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 16,
        rank = -1,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        # lora_rank = 0
    )

    george_debug_baseline = LMTransformerArgs(
        dim = 512,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 16,
        rank = -1,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        # lora_rank = 0
    )

    # ---------------------------------------------------------------------------------------------------------
    # FINAL RRT STUFF
    # ---------------------------------------------------------------------------------------------------------
    sequence_DoubleLayers = LMTransformerArgs(
        dim = 400,
        ffn_dim_multiplier=1.25,
        n_layers = 24,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23]],
        lora_rank = 32
    )

    sequence_NormalLayers = LMTransformerArgs( 
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 32
    )

    baseline2 = LMTransformerArgs(
        dim = 480,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    fws2 = LMTransformerArgs(
        dim = 1392,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    sequence_150M = LMTransformerArgs(  # 150.4M params, ffn_dim = 4096
        dim = 1304,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 32
    )

    cycle_150M = LMTransformerArgs(  # 150.4M params, ffn_dim = 4096
        dim = 1304,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 32
    )

    cycle_rev_150M = LMTransformerArgs(  # 150.4M params, ffn_dim = 4096
        dim = 1296,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 11], [1, 10], [2, 9], [3, 8], [4, 7], [5, 6]],
        lora_rank = 32
    )

    universal_150M = LMTransformerArgs(  # 149.6M params, ffn_dim = 9216
        dim = 3072,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    # HIERARCHICAL ARCHITECTURES 150M--------------------------------------------------------------------------------------------------------
    hierarchical_150M = LMTransformerArgs(  # 150.78M params, ffn = 4864
        dim = 1616,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )
    hierarchical_200M = LMTransformerArgs(  # 199.1M params, ffn = 5632
        dim = 1872,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )
    hierarchical_228M = LMTransformerArgs(  
        dim = 2000,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    hierarchical_240M = LMTransformerArgs(  # 239.5M params, ffn = 6656
        dim = 1984,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    hierarchical_600M = LMTransformerArgs(  # 599.1M params, ffn = 9984
        dim = 3312,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )
    
    hierarchical2_150M = LMTransformerArgs(  
        dim = 1616,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2, 3], [4, 5, 6], [7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    bulge_150M = LMTransformerArgs(  # 149.87M params, ffn = 4096
        dim = 1312, 
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    cycle_rev = LMTransformerArgs( # 40.58M params, ffn_dim = 2048
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 11], [1, 10], [2, 9], [3, 8], [4, 7], [5, 6]],
        lora_rank = 32
    )

    cycle_2share = LMTransformerArgs( # 40.58M params, ffn_dim = 2048
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 32
    )

    cycle_3share = LMTransformerArgs( # 39.8M params, ffn_dim = 2304
        dim = 768,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
        lora_rank = 32
    )

    cycle_4share_dim = LMTransformerArgs( # 40.99M params, ffn_dim = 2816
        dim = 864,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]],
        lora_rank = 32
    )

    cycle_4share_ffn = LMTransformerArgs(  # 39.4M params, ffn_dim = 2816
        dim = 832,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]],
        lora_rank = 32
    )

    cycle_6share = LMTransformerArgs( # 40.57M params, ffn_dim = 3328
        dim = 1040,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
        lora_rank = 32
    )


    # ---------------------------------------------------------------------------------------------------------
    # RRT RANK SWEEP 40M
    # ---------------------------------------------------------------------------------------------------------
    rank4 = LMTransformerArgs( # 40.28M params, ffn = 2304
        dim = 640,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 4
    )

    rank8 = LMTransformerArgs( # 39.78M params, ffn = 2304
        dim = 624,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 8
    )

    rank16_1 = LMTransformerArgs( # 39.75M params, ffn = 2048
        dim = 648,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 16
    )

    rank16_2 = LMTransformerArgs( # 40.52M params, ffn = 2304
        dim = 616,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 16
    )

    rank32 = LMTransformerArgs( # 
        dim = 600,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 32
    )


    # ---------------------------------------------------------------------------------------------------------
    # RRT RANK SWEEP 150M
    # ---------------------------------------------------------------------------------------------------------
    rank4_150M = LMTransformerArgs(  # 149.24M params, ffn = 4096
        dim = 1360,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 4
    )

    rank8_150M = LMTransformerArgs(  # 150.55M params, ffn = 4096
        dim = 1360,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 8
    )

    rank16_150M = LMTransformerArgs(  # 150.91M params, ffn = 4096
        dim = 1344,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 16
    )

    rank32_150M = LMTransformerArgs(  # 150.4M params, ffn_dim = 4096
        dim = 1304, # this needs to be 1296!!!
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 32
    )

    rank64_150M = LMTransformerArgs(  # 150.72M params, ffn = 3840
        dim = 1280,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 64
    )

    rank128_150M = LMTransformerArgs(  # 149.81M params, ffn = 3584
        dim = 1184,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 128
    )

    rank256_150M = LMTransformerArgs(  # 150.49M params, ffn = 3328
        dim = 992,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 256
    )

    rank512_150M = LMTransformerArgs(  # 149.48M params, fffn = 2560
        dim = 752,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 512
    )

    # ---------------------------------------------------------------------------------------------------------
    # RRT 1B
    # ---------------------------------------------------------------------------------------------------------
    sequence_1B = LMTransformerArgs(  # 1.0034B # ffn = 10752
        dim = 3504,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 32
    )

    cycle_1B = LMTransformerArgs(  # 1.0034B # ffn = 10752
        dim = 3504,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 32
    )

    cycle_rev_1B = LMTransformerArgs(  # 1.0034B # ffn = 10752
        dim = 3504,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 11], [1, 10], [2, 9], [3, 8], [4, 7], [5, 6]],
        lora_rank = 32
    )

    universal_150M = LMTransformerArgs(  # 149.6M params, ffn_dim = 9216
        dim = 3072,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    hierarchical_1B = LMTransformerArgs(  # 999.8M params, ffn = 13056
        dim = 4288,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )
    hierarchical_1B_2 = LMTransformerArgs(  # 1.004B params, ffn = 13056
        dim = 4304,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    bulge_1B = LMTransformerArgs(  # 998.89M params, ffn = 10752
        dim = 3504, 
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )
    bulge_1B_2 = LMTransformerArgs(  # 1.004B params, ffn = 10752
        dim = 3520, 
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )


    smallvocab = LMTransformerArgs(
        dim = 512,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    bigvocab = LMTransformerArgs(
        dim = 512,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 100,
        vocab_size=32000,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    # ---------------------------------------------------------------------------------------------------------
    # 40M RRT COMPARISON
    # ---------------------------------------------------------------------------------------------------------
    bulge_40M = LMTransformerArgs(  # 39.75M params ffn = 2048
        dim = 624, 
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    hierarch_40M = LMTransformerArgs( # 40.07M params, ffn = 2560
        dim = 736,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    cycle_40M = LMTransformerArgs( # 40.58M params, ffn_dim = 2048
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 32
    )

    cycle_rev_40M = LMTransformerArgs( # 40.58M params, ffn_dim = 2048
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 11], [1, 10], [2, 9], [3, 8], [4, 7], [5, 6]],
        lora_rank = 32
    )

    sequence_40M = LMTransformerArgs( # 40.58M params, ffn = 2048
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 32
    )
    

    # ---------------------------------------------------------------------------------------------------------
    # 200M RRT COMPARISON
    # ---------------------------------------------------------------------------------------------------------
    # bulge_200M = LMTransformerArgs(  # 197.1M params, ffn = 4608
    #     dim = 1536, 
    #     ffn_dim_multiplier=1.125,
    #     n_layers = 12,
    #     n_heads = 8,
    #     rank = 25,
    #     vocab_size=128256,
    #     weight_tying=True,
    #     layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
    #     lora_rank = 32
    # )
    # bulge_200M = LMTransformerArgs(  # 201M params, ffn = 5120
    #     dim = 1472, 
    #     ffn_dim_multiplier=1.25,
    #     n_layers = 12,
    #     n_heads = 8,
    #     rank = 25,
    #     vocab_size=128256,
    #     weight_tying=True,
    #     layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
    #     lora_rank = 32
    # )
    bulge_200M = LMTransformerArgs(  # 199.2M params, ffn = 4864
        dim = 1504, 
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    hierarch_200M = LMTransformerArgs(  # 199.1M params, ffn = 5632
        dim = 1872,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]],
        lora_rank = 32
    )

    cycle_200M = LMTransformerArgs( # 199.1M params, ffn_dim = 4608
        dim = 1536,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
        lora_rank = 32
    )

    cycle_rev_200M = LMTransformerArgs( # 199.1M params, ffn_dim = 4608
        dim = 1536,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 11], [1, 10], [2, 9], [3, 8], [4, 7], [5, 6]],
        lora_rank = 32
    )

    sequence_200M = LMTransformerArgs( # 199.1M params, ffn_dim = 4608
        dim = 1536,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        lora_rank = 32
    )

    # baseline_200M = LMTransformerArgs(  # 204M params, ffn = 3584
    #     dim = 1104,
    #     ffn_dim_multiplier=1.15,
    #     n_layers = 12,
    #     n_heads = 8,
    #     rank = 25,
    #     vocab_size=128256,
    #     weight_tying=True,
    #     layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
    #     lora_rank = 0
    # )
    baseline_200M = LMTransformerArgs(  # 200.4M params, ffn = 3584
        dim = 1088,
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    # ---------------------------------------------------------------------------------------------------------
    # FINAL MODEL RUNS
    # ---------------------------------------------------------------------------------------------------------

    vanilla_40M = LMTransformerArgs(  # 40.59M params, ffn = 512
        dim = 144,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    final_40M = LMTransformerArgs(  # 39.74M params, ffn = 2048
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    scaled_40M = LMTransformerArgs(  # 224.8M params, ffn = 2048
        dim = 624,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    vanilla_200M = LMTransformerArgs(  # 200.8M params, ffn = 1792
        dim = 576,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    bulge_200M = LMTransformerArgs(  # 199.2M params, ffn = 4864
        dim = 1504, 
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    scaled_200M = LMTransformerArgs(  # 757.76M params, ffn = 4864
        dim = 1504, 
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    vanilla_1B = LMTransformerArgs(  # 1.008B params, ffn = 5632
        dim = 1840, 
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    final_1B = LMTransformerArgs(  # 1.004B params, ffn = 10752
        dim = 3520, 
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    scaled_1B = LMTransformerArgs(  # 2.860B params, ffn = 10752 
        dim = 3520, 
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    rank0_40M = LMTransformerArgs( # 39.61M params, ffn = 2304
        dim = 640,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 0
    )

    rank4_40M = LMTransformerArgs( # 40.17M params, ffn = 2304
        dim = 640,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 4
    )

    rank8_40M = LMTransformerArgs( # 39.56M params, ffn = 2304
        dim = 624,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 8
    )

    rank16_40M = LMTransformerArgs( # 40.6M params, ffn = 2304
        dim = 624,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 16
    )

    rank32_38M = LMTransformerArgs( # 38.63M params, ffn = 2048
        dim = 608,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    rank32_42M = LMTransformerArgs( # 42.87M params, ffn = 2304
        dim = 624,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )

    rank64_40M = LMTransformerArgs( # 40.41M params, ffn = 2048
        dim = 576,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 64
    )

    rank128_40M = LMTransformerArgs( # 40.13M params, ffn = 1792
        dim = 512,
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 128
    )

    rank0_200M = LMTransformerArgs(  # 200M params, ffn = 5120
        dim = 1520, 
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 0
    )

    rank4_200M = LMTransformerArgs(  # 200M params, ffn = 5120
        dim = 1520, 
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 4
    )

    rank8_200M = LMTransformerArgs(  # 198.73M params, ffn = 5120
        dim = 1520, 
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 8
    )


    rank16_200M = LMTransformerArgs(  # 201.28M params, ffn = 5120
        dim = 1504, 
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 16
    )


    rank32_200M = LMTransformerArgs(  # 201M params, ffn = 5120
        dim = 1472, 
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 32
    )


    rank64_200M = LMTransformerArgs(  # 201M params, ffn = 4864
        dim = 1456, 
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 64
    )


    rank128_200M = LMTransformerArgs(  # 200M params, ffn = 4608
        dim = 1376, 
        ffn_dim_multiplier=1.25,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 128
    )



    fac_40M = LMTransformerArgs(  # 40.54M params, ffn = 768
        dim = 224,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    share_40M = LMTransformerArgs(  # 40.18M params, ffn = 768
        dim = 240,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    fac_share_40M = LMTransformerArgs(  # 40.83M params, ffn = 1536
        dim = 480,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    # fac_200M = LMTransformerArgs(  # 200M params, ffn = 2560 (had to tweak ffn dim slightly)
    #     dim = 768,
    #     ffn_dim_multiplier=1.135,
    #     n_layers = 12,
    #     n_heads = 8,
    #     rank = 25,
    #     vocab_size=128256,
    #     weight_tying=False,
    #     layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
    #     lora_rank = 0
    # )
    fac_200M = LMTransformerArgs(  # 201M params, ffn = 2560 (had to tweak ffn dim slightly)
        dim = 768,
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    # share_200M = LMTransformerArgs( # 202M params, ffn = 2560
    #     dim = 784,
    #     ffn_dim_multiplier=1.135,
    #     n_layers = 12,
    #     n_heads = 8,
    #     rank = 0,
    #     vocab_size=128256,
    #     weight_tying=True,
    #     layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
    #     lora_rank = 0
    # )

    share_200M = LMTransformerArgs( # 202M params, ffn = 2560
        dim = 784,
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    # fac_share_200M = LMTransformerArgs(  # 204M params, ffn = 3584
    #     dim = 1104,
    #     ffn_dim_multiplier=1.135,
    #     n_layers = 12,
    #     n_heads = 8,
    #     rank = 25,
    #     vocab_size=128256,
    #     weight_tying=True,
    #     layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
    #     lora_rank = 0
    # )

    fac_share_200M = LMTransformerArgs(  # 200.4M params, ffn = 3584
        dim = 1088,
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 25,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    noshare_200M = LMTransformerArgs(  # 200.8M params, ffn = 1792
        dim = 576,
        ffn_dim_multiplier=1.135,
        n_layers = 12,
        n_heads = 8,
        rank = 0,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    facshare_r32_200M = LMTransformerArgs(  # 201M params, ffn = 3584
        dim = 1088,
        ffn_dim_multiplier=1.15,
        n_layers = 12,
        n_heads = 8,
        rank = 32,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    facshare_r64_200M = LMTransformerArgs(  # 199M params, ffn = 3328, emb = 4.12%
        dim = 1104,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 64,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    facshare_r128_200M = LMTransformerArgs(  # 200M params, ffn = 3328, emb = 8.2%
        dim = 1072,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 128,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    facshare_r256_200M = LMTransformerArgs(  # CHECK PARAMS!!
        dim = 1024,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 256,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    facshare_r512_200M = LMTransformerArgs(  # 201M params, ffn = 2816, emb = 32.6% 
        dim = 928,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 512,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )

    config1_200M = LMTransformerArgs(  # 199M params, ffn = 4608, emb = 4.12%
        dim = 1456,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 64,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 64
    )

    config2_200M = LMTransformerArgs(  # 198M params, ffn = 4352, emb = 8.3%
        dim = 1440,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 128,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 64
    )

    config3_200M = LMTransformerArgs(  # 205M params, ffn = 4352, emb = 16%
        dim = 1376,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 256,
        vocab_size=128256,
        weight_tying=True,
        layer_groups = [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]],
        lora_rank = 64
    )

    fac_no_share_128_200M = LMTransformerArgs(  # 198M params, ffn = 2304,
        dim = 736,
        ffn_dim_multiplier=1.125,
        n_layers = 12,
        n_heads = 8,
        rank = 128,
        vocab_size=128256,
        weight_tying=False,
        layer_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
        lora_rank = 0
    )







    args = fac_no_share_128_200M
    model = LMTransformer(args)
    num_params = get_num_params(model)
    print(f"Model Args: {args}")
    print(f"Model Params: {num_params}")
    print(f"% Embedding: {model.tok_embeddings1.weight.numel()/num_params:.2%}")
    # print(f"% Embedding: {model.tok_embeddings.weight.numel()/num_params:.2%}")
    #for name, param in model.layers[0].attention.named_parameters():
    #    print(f"{name}: {param.shape}")
    #print(model_args.project_up_layers[0].in_dim)
if __name__ == "__main__":
    main()
