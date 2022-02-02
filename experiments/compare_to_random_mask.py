import torch

from evaluation.pruning_vis import mask_to_png
from models.conv_ae import ConvAE
from procedures.in_place_pruning import prune_model
from procedures.retrain import retrain_tagged_networks, retrain_with_shots
from utils.file import get_topology_of_run, get_params_of_run
from datasets.get_loaders import get_loaders
from utils.get_run_id import last_run
from utils.misc import get_device

run_id = last_run()
ratio = 0.5
draw_epoch = 12
draw_sub_epoch = 4
mask_file = f"runs/{run_id}/keep-{ratio}-epoch-{draw_epoch}-{draw_sub_epoch}.pth"
device = get_device()

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()

    def get_networks():

        # unpruned = ConvAE.init_from_checkpoint(run_id, None, None, None).to(device)

        ticket = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=None).to(device)

        # ticket_eb = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=draw_epoch).to(device)

        # ticket_resume = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=1).to(device)

        # ticket_reset = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=1).to(device)
        # ticket_reset.reset_weights()

        # get ticket mask and extract ratio per layer
        ticket_masks = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch).get_mask_of_current_network_state(ratio)
        random_masks = []
        for t in ticket_masks:
            nonzero = torch.count_nonzero(t)
            r = torch.tensor([1 for _ in range(nonzero)] + [0 for _ in range(t.nelement() - nonzero)])
            r_idx = torch.randperm(torch.numel(r))
            r = r[r_idx].to(device)
            random_masks.append(r)
        # mask_to_png(torch.load(mask_file), caption="Original mask", show=True, save=False)
        # mask_to_png(ticket_masks, caption="Expanded mask", show=True, save=False)
        # mask_to_png(random_masks, caption="Random mask with same ratio per layer", show=True, save=False)

        random_ticket = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=None)
        prune_model(random_ticket, random_masks)

        # random_ticket_eb = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
        # prune_model(random_ticket_eb, random_masks)
        # random_ticket_resume = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=1)
        # prune_model(random_ticket_resume, random_masks)

        networks = [
            ("original_ticket", ticket),
            # ("original_ticket_eb", ticket_eb),
            # ("original_ticket_resume", ticket_resume),
            # ("reset_ticket", ticket_reset),
            ("random_ticket", random_ticket.to(device)),
            # ("random_ticket_eb", random_ticket_eb.to(device)),
            # ("random_ticket_resume", random_ticket_resume.to(device)),
            # ("unpruned", unpruned),
        ]
        return networks

    retrain_with_shots(get_networks, f"graph_data/retraining/random_mask-{run_id}-{ratio}.json", shots=5)
    # retrain_tagged_networks(networks, f"graphs/graph_data/random_mask-{run_id}-{ratio}.json")
