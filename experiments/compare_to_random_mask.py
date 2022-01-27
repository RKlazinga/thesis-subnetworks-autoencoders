import torch

from evaluation.pruning_vis import mask_to_png
from models.conv_ae import ConvAE
from procedures.in_place_pruning import prune_model
from procedures.retrain import retrain_tagged_networks
from utils.file import get_topology_of_run, get_params_of_run
from datasets.get_loaders import get_loaders
from utils.get_run_id import last_run
from utils.misc import get_device

run_id = last_run()
ratio = 0.7
draw_epoch = 3
draw_sub_epoch = 4
mask_file = f"runs/{run_id}/keep-{ratio}-epoch-{draw_epoch}-{draw_sub_epoch}.pth"
device = get_device()

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()

    unpruned = ConvAE.init_from_checkpoint(run_id, None, None, None)

    ticket = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=None).to(device)

    ticket_reset = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=None).to(device)
    ticket_reset.reset_weights()

    # get ticket mask and extract ratio per layer
    random_ticket = ConvAE(*get_topology_of_run(run_id)).to(device)
    random_ticket.load_state_dict(get_params_of_run(run_id, device=device))
    ticket_masks = torch.load(mask_file)
    random_masks = []
    for t in ticket_masks:
        nonzero = torch.count_nonzero(t)
        r = torch.tensor([1 for _ in range(nonzero)] + [0 for _ in range(t.nelement() - nonzero)])
        r_idx = torch.randperm(torch.numel(r))
        r = r[r_idx]
        random_masks.append(r)
    # mask_to_png(ticket_masks, caption="Original mask", show=True, save=False)
    # mask_to_png(random_masks, caption="Random mask with same ratio per layer", show=True, save=False)
    prune_model(random_ticket, random_masks)

    networks = [
        ("unpruned", unpruned),
        ("original_ticket", ticket),
        ("reset_ticket", ticket_reset),
        ("random_ticket", random_ticket),
    ]

    retrain_tagged_networks(networks, f"graphs/graph_data/random_mask-{run_id}-{ratio}.json")
