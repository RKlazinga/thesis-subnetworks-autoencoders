from torch import nn

from models.conv_ae import ConvAE
from procedures.retrain import retrain_tagged_networks
from settings import Settings
from utils.file import get_topology_of_run, get_params_of_run
from datasets.get_loaders import get_loaders
from utils.get_run_id import last_run
from utils.misc import dev

run_id = last_run()
ratio = 0.5
draw_epoch = 9
draw_sub_epoch = 4
mask = f"{Settings.RUN_FOLDER}/{run_id}/prune-{ratio}-epoch-{draw_epoch}-{draw_sub_epoch}.pth"
device = dev()

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()

    unpruned = ConvAE(*get_topology_of_run(run_id)).to(device)
    unpruned.load_state_dict(get_params_of_run(run_id, device=device))

    pruned = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch).to(device)

    pruned_1 = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=1).to(device)

    pruned_2 = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=2).to(device)

    pruned_continue = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch, param_epoch=9).to(device)

    pruned_reset = ConvAE.init_from_checkpoint(run_id, ratio, draw_epoch, draw_sub_epoch).to(device)
    # reset weights
    for m in pruned_reset.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.bias.data.zero_()

    networks = [
        ("unpruned", unpruned),
        ("pruned_no_resume", pruned),
        ("pruned_resume_1", pruned_1),
        ("pruned_resume_2", pruned_2),
        ("pruned_continue", pruned_continue),
        ("pruned_random_init", pruned_reset)
    ]

    graph_data = {}

    retrain_tagged_networks(networks, f"graph_data/retraining/reset-{run_id}-{ratio}.json")
