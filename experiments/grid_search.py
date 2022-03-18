from experiments.progressive_mask_drawing import train_and_draw_tickets, main
from utils.channel_sparsity_reg import update_bn


# for sparsity in [4e-5, 1e-3, 2e-2]:
sparsity = 1e-4
for latent_sparsity in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 2e-1]:
    for lr in [1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2]:
        update_bn.__defaults__ = (sparsity, latent_sparsity)
        train_and_draw_tickets.__defaults__ = ("grid_search", lr)
        main(f"[{sparsity},{latent_sparsity},{lr}]-")
