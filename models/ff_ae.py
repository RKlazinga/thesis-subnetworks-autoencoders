from typing import Union

import torch
from torch import nn

from procedures.in_place_pruning import prune_model
from utils.file import get_topology_of_run, get_params_of_run


class FeedforwardAE(nn.Module):

    IN_SIZE = 16

    def __init__(self, latent_size, hidden_layers, size_mult):
        super().__init__()

        # linearly interpolate between input size and latent size
        topology = [self.IN_SIZE]
        for x in range(hidden_layers):
            interp = (x+1) / (hidden_layers + 1)

            topology.append(round(size_mult * (1-interp) * self.IN_SIZE + interp * latent_size))
        topology.append(latent_size)

        encoder_steps = []
        for i in range(len(topology) - 1):
            encoder_steps.append(nn.Linear(topology[i], topology[i+1]))
            encoder_steps.append(nn.BatchNorm1d(topology[i+1]))
            encoder_steps.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_steps)

        decoder_steps = []
        topology.reverse()
        for i in range(len(topology) - 1):
            decoder_steps.append(nn.Linear(topology[i], topology[i+1]))
            if i < len(topology) - 2:
                print(f"BN layer {i}")
                decoder_steps.append(nn.BatchNorm1d(topology[i+1]))
            else:
                print(f"Skipped layer {i}")
            decoder_steps.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_steps)

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    @staticmethod
    def init_from_checkpoint(run_id, ratio: Union[float, None], epoch, sub_epoch=1, param_epoch=None):
        topology = get_topology_of_run(run_id)

        network = FeedforwardAE(*topology)
        network.load_state_dict(get_params_of_run(run_id, param_epoch))

        if ratio is not None:
            mask_file = f"runs/{run_id}/prune-{ratio}-epoch-{epoch}-{sub_epoch}.pth"
            masks = torch.load(mask_file)
            prune_model(network, masks)

        return network


if __name__ == '__main__':
    FeedforwardAE(3, 1, 1)
