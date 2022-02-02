import json
from torch.nn import MSELoss
from torch.optim import Adam

from datasets.get_loaders import get_loaders
from models.conv_ae import ConvAE
from procedures.in_place_pruning import prune_model
from procedures.retrain import retrain_with_shots
from procedures.test import test
from procedures.ticket_drawing.with_redist import find_channel_mask_redist
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from procedures.train import train
from settings.retrain_settings import RETRAIN_LR, RETRAIN_RESUME_EPOCH, RETRAIN_EPOCHS
from utils.file import get_topology_of_run, get_params_of_run
from utils.get_run_id import last_run
from utils.misc import get_device

run_id = last_run()
ratio = 0.5
_draw_epoch = 9
device = get_device()

if __name__ == '__main__':

    def get_networks(draw_epoch):
        unpruned = ConvAE(*get_topology_of_run(run_id)).to(device)
        unpruned.load_state_dict(get_params_of_run(run_id, device=device))

        pruned_no_redist = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
        mask = list(find_channel_mask_no_redist(pruned_no_redist, ratio, 0.0).values())
        # mask_to_png(mask, "No redistribution")
        pruned_no_redist.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
        prune_model(pruned_no_redist, mask)

        pruned_no_redist_with_lim = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
        mask = list(find_channel_mask_no_redist(pruned_no_redist_with_lim, ratio, 0.2).values())
        # mask_to_png(mask, "No redistribution (layer limit)")
        pruned_no_redist_with_lim.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
        prune_model(pruned_no_redist_with_lim, mask)

        # pruned_prop_redist = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
        # mask = list(find_channel_mask_redist(pruned_prop_redist, ratio, redist_function="proportional").values())
        # mask_to_png(mask, "Proportional redistribution")
        # pruned_prop_redist.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
        # prune_model(pruned_prop_redist, mask)

        pruned_similarity_redist = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
        mask = list(find_channel_mask_redist(pruned_similarity_redist, ratio, redist_function="weightsim").values())
        # mask_to_png(mask, "Sign-similarity redistribution (eps=0)")
        pruned_similarity_redist.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
        prune_model(pruned_similarity_redist, mask)

        pruned_similarity_redist1 = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
        mask = list(find_channel_mask_redist(pruned_similarity_redist1, ratio,
                                             redist_function="weightsim-0.001").values())
        # mask_to_png(mask, "Sign-similarity redistribution (eps=0.001)")
        pruned_similarity_redist1.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
        prune_model(pruned_similarity_redist1, mask)

        networks = [
            ("unpruned", unpruned),
            ("no_redist", pruned_no_redist),
            ("no_redist_lim", pruned_no_redist_with_lim),
            ("similarity_redist", pruned_similarity_redist),
            ("similarity_redist1", pruned_similarity_redist1)
        ]
        return networks

    retrain_with_shots(get_networks, _draw_epoch, f"graph_data/retraining/prune_compare-{run_id}-{ratio}.json")
