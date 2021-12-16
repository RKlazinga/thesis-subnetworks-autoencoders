import json
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from datasets.get_loaders import get_loaders
from evaluation.pruning_vis import mask_to_png
from models.conv_ae import ConvAE
from procedures.in_place_pruning import prune_model
from procedures.test import test
from procedures.ticket_drawing.with_redist import find_channel_mask_redist
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from procedures.train import train
from settings.retrain_settings import RETRAIN_EPOCHS, RETRAIN_LR, RETRAIN_RESUME_EPOCH
from utils.file import get_topology_of_run, get_params_of_run

run_id = "[6, 4, 6]-425222fd6"
ratio = 0.5
draw_epoch = 9
draw_sub_epoch = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train_loader, test_loader = get_loaders()

    unpruned = ConvAE(*get_topology_of_run(run_id)).to(device)
    unpruned.load_state_dict(get_params_of_run(run_id, device=device))

    pruned_no_redist = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
    mask = list(find_channel_mask_no_redist(pruned_no_redist, ratio, 0.0).values())
    mask_to_png(mask, "No redistribution")
    pruned_no_redist.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
    prune_model(pruned_no_redist, mask)

    pruned_no_redist_with_lim = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
    mask = list(find_channel_mask_no_redist(pruned_no_redist_with_lim, ratio, 0.2).values())
    mask_to_png(mask, "No redistribution (layer limit)")
    pruned_no_redist_with_lim.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
    prune_model(pruned_no_redist_with_lim, mask)

    pruned_prop_redist = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
    mask = list(find_channel_mask_redist(pruned_prop_redist, ratio, redist_function="proportional").values())
    mask_to_png(mask, "Proportional redistribution")
    pruned_prop_redist.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
    prune_model(pruned_prop_redist, mask)

    pruned_similarity_redist = ConvAE.init_from_checkpoint(run_id, None, None, None, param_epoch=draw_epoch)
    mask = list(find_channel_mask_redist(pruned_similarity_redist, ratio, redist_function="weightsim").values())
    mask_to_png(mask, "Sign-similarity redistribution")
    pruned_similarity_redist.load_state_dict(get_params_of_run(run_id, RETRAIN_RESUME_EPOCH))
    prune_model(pruned_similarity_redist, mask)

    networks = [
        ("unpruned", unpruned),
        ("no_redist", pruned_no_redist),
        ("no_redist_lim", pruned_no_redist_with_lim),
        ("prop_redist", pruned_prop_redist),
        ("similarity_redist", pruned_similarity_redist)
    ]

    graph_data = {}

    for net_tag, net in networks:
        net.to(device)
        print(f"Retraining: {net_tag}")
        current_graph_data = []
        graph_data[net_tag] = current_graph_data
        optimiser = Adam(net.parameters(), lr=RETRAIN_LR)
        criterion = MSELoss()

        for epoch in range(RETRAIN_EPOCHS):
            train_loss = train(net, optimiser, criterion, train_loader, device)
            test_loss = test(net, criterion, test_loader, device)

            print(train_loss, test_loss)
            current_graph_data.append((epoch, train_loss, test_loss))

            with open(f"graphs/graph_data/prune_compare-{run_id}-{ratio}.json", "w") as write_file:
                write_file.write(json.dumps(graph_data))
