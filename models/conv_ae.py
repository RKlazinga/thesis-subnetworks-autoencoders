from typing import Union

import torch
from torch import nn

from procedures.in_place_pruning import prune_model
from procedures.ticket_drawing.with_redist import find_channel_mask_redist
from procedures.ticket_drawing.without_redist import find_channel_mask_no_redist
from settings.prune_settings import PRUNE_WITH_REDIST
from utils.conv_unit import ConvUnit, ConvTransposeUnit
from utils.crop_module import Crop
from utils.file import get_topology_of_run, get_params_of_run
from utils.math import calculate_im_size


class ConvAE(nn.Module):

    IMAGE_SIZE = 28

    def __init__(self, latent_size, hidden_layers, size_mult, in_channels=1):
        super().__init__()

        topology = [in_channels] + [size_mult * 2**x for x in range(1, hidden_layers)]
        encoder_steps = []

        prev_step_size = topology[0]
        for h in topology:
            encoder_steps.append(ConvUnit(prev_step_size, h, 3, max_pool=True, bn=True, padding=1))
            prev_step_size = h
        encoder_steps.append(nn.Flatten())
        encoder_steps.append(nn.Linear(prev_step_size * calculate_im_size(self.IMAGE_SIZE, hidden_layers) ** 2,
                                       latent_size))
        encoder_steps.append(nn.BatchNorm1d(latent_size))
        encoder_steps.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_steps)

        topology.reverse()
        decoder_steps = [
            nn.Linear(latent_size, prev_step_size * calculate_im_size(self.IMAGE_SIZE, hidden_layers) ** 2),
            nn.Unflatten(dim=1, unflattened_size=(prev_step_size,
                                                  calculate_im_size(self.IMAGE_SIZE, hidden_layers),
                                                  calculate_im_size(self.IMAGE_SIZE, hidden_layers))),
            nn.BatchNorm2d(prev_step_size),
            nn.ReLU()
        ]

        for h in topology:
            decoder_steps.append(ConvTransposeUnit(prev_step_size, h, 3, bn=True,
                                                   stride=2, padding=1, output_padding=1))
            prev_step_size = h

        # resolve checkerboarding with normal convolution
        decoder_steps.append(ConvUnit(prev_step_size, in_channels, 3, padding=1, activation=nn.Sigmoid, bn=False))

        # if the input image size was not a power of 2, the output will be too large. crop down to image size
        decoder_steps.append(Crop(self.IMAGE_SIZE))

        self.decoder = nn.Sequential(*decoder_steps)
        self.reset_weights()

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def get_mask_of_current_network_state(self, ratio):
        channel_mask_func = find_channel_mask_redist if PRUNE_WITH_REDIST else find_channel_mask_no_redist
        return list(channel_mask_func(self, ratio).values())

    @staticmethod
    def init_from_checkpoint(run_id, ratio: Union[float, None], epoch, sub_epoch=1, param_epoch=None, from_disk=False):
        topology = get_topology_of_run(run_id)

        network = ConvAE(*topology)
        network.load_state_dict(get_params_of_run(run_id, param_epoch))

        if ratio is not None:
            if from_disk:
                mask_file = f"runs/{run_id}/keep-{ratio}-epoch-{epoch}-{sub_epoch}.pth"
                masks = torch.load(mask_file)
            else:
                masks = network.get_mask_of_current_network_state(ratio)
            prune_model(network, masks)
        return network
