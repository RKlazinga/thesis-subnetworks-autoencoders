from experiments.progressive_mask_drawing import train_and_draw_tickets, main
from utils.channel_sparsity_reg import update_bn
from datasets.synthetic.flat_generators import random_sine_gaussian

SHOTS = 2
conv_sparsity = 1e-4

# linear_sparsity = 1e-4
# latent_sparsity = 1e-4
# update_bn.__defaults__ = (conv_sparsity, linear_sparsity, latent_sparsity)
# main(f"[{latent_sparsity}]-")

linear_sparsity = 1e-3
latent_sparsity = 1e-1
# for latent_sparsity in [1]:
# for latent_sparsity in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
for l in [8]:
    folder, lr, _ = train_and_draw_tickets.__defaults__
    train_and_draw_tickets.__defaults__ = (folder, lr, [l, 2, 1])
    for d in [1, 2, 3, 4]:
        size, std, num = random_sine_gaussian.__defaults__
        random_sine_gaussian.__defaults__ = (size, std, d)
        for latent_sparsity in [1e-3, 1e-2, 1e-1, 1]:
            for linear_sparsity in [1e-3, 1e-2, 1e-1]:
                # for lr in [2e-2, 4e-2]: #[1e-4, 1e-3, 2e-3, 4e-3, 1e-2, #3e-2]:
                #     linear_sparsity = latent_sparsity
                update_bn.__defaults__ = (conv_sparsity, linear_sparsity, latent_sparsity)
                for shot in range(SHOTS):
                    main(f"[{l}-{d}-{latent_sparsity}-{linear_sparsity}]-", [l, 2, 1])
