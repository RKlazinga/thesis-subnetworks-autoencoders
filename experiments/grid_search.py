from experiments.progressive_mask_drawing import main
from settings import Settings

SHOTS = 2
conv_sparsity = 1e-4
linear_sparsity = 1e-3
latent_sparsity = 1e-1
for l in [8]:
    Settings.TOPOLOGY = [l, 2, 1]
    for d in [1, 2, 3, 4]:
        Settings.NUM_VARIABLES = d
        for latent_sparsity in [1e-3, 1e-2, 1e-1, 1]:
            for linear_sparsity in [1e-3, 1e-2, 1e-1]:
                Settings.LATENT_SPARSITY_PENALTY = latent_sparsity
                Settings.LINEAR_SPARSITY_PENALTY = linear_sparsity
                for shot in range(SHOTS):
                    main(f"[{l}-{d}-{latent_sparsity}-{linear_sparsity}]-")
