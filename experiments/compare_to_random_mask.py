import random

import torch

from models.conv_ae import ConvAE
from procedures.in_place_pruning import prune_model
from procedures.retrain import retrain_with_shots
from settings import Settings
from utils.get_run_id import last_run
from utils.misc import dev

run_id = last_run()
ratios = [0.1, 0.3, 0.5, 0.7]
shots = 1
draw_epoch = 12
draw_sub_epoch = 4


def verify_ratio_approx_correct(desired_ratio, mask):
    numer = 0
    denom = 0
    for m in mask:
        numer += torch.count_nonzero(m)
        denom += torch.numel(m)
    print(f"Desired: {desired_ratio} Actual: {1 - (numer/denom)}")


def run(ratio):
    print(f"Ratio: {ratio}, ETA: {15*2*Settings.RETRAIN_EPOCHS*shots/60} minutes")

    def get_networks():
        ticket = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=None).to(dev())

        # get ticket mask and extract ratio per layer
        ticket_masks = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch).get_mask_of_current_network_state(ratio)
        random_masks = []
        for t in ticket_masks:
            nonzero = torch.count_nonzero(t)
            r = torch.tensor([1 for _ in range(nonzero)] + [0 for _ in range(t.nelement() - nonzero)])
            r_idx = torch.randperm(torch.numel(r))
            r = r[r_idx].to(dev())
            random_masks.append(r)
        # mask_to_png(torch.load(mask_file), caption="Original mask", show=True, save=False)
        # mask_to_png(ticket_masks, caption="Expanded mask", show=True, save=False)
        # mask_to_png(random_masks, caption="Random mask with same ratio per layer", show=True, save=False)

        same_mask_random_weights = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=None).to(dev())
        same_mask_random_weights.reset_weights()

        random_ticket = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=None)
        prune_model(random_ticket, random_masks)

        utterly_random_masks = []
        t_sizes = [torch.numel(t) for t in ticket_masks]
        total = sum(t_sizes)
        protected = sum([t if t <= 6 else 0 for t in t_sizes])
        unprotected = total - protected
        desired_zeroes = round(total * ratio)
        values = [0 for _ in range(desired_zeroes)] + [1 for _ in range(unprotected - desired_zeroes)]
        random.shuffle(values)
        idx = 0
        for t in ticket_masks:
            t_size = torch.numel(t)
            if t_size <= 6:
                utterly_random_masks.append(torch.ones(t_size))
            else:
                utterly_random_masks.append(torch.tensor(values[idx:idx+t_size]))
                idx += t_size

        verify_ratio_approx_correct(ratio, utterly_random_masks)
        # mask_to_png(ticket_masks, show=True, save=False)
        # mask_to_png(utterly_random_masks, show=True, save=False)
        utterly_random_ticket = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=None)
        prune_model(utterly_random_ticket, utterly_random_masks)

        networks = [
            ("original_ticket", ticket),
            ("reinitialised", same_mask_random_weights),
            ("random_ticket", random_ticket.to(dev())),
            # ("utterly_random", utterly_random_ticket.to(dev())),
        ]
        return networks

    retrain_with_shots(get_networks, f"graph_data/retraining/random_mask-{run_id}-{ratio}.json", shots=shots)


if __name__ == '__main__':
    for r in ratios:
        run(r)
