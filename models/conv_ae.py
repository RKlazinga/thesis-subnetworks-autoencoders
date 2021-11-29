import os

import torch
from torch import nn

from procedures.in_place_pruning import prune_model
from utils.conv_unit import ConvUnit, ConvTransposeUnit
from utils.crop_module import Crop
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
        encoder_steps.append(nn.Linear(prev_step_size * calculate_im_size(self.IMAGE_SIZE, hidden_layers) ** 2, latent_size))
        encoder_steps.append(nn.BatchNorm1d(latent_size))
        encoder_steps.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_steps)

        topology.reverse()
        decoder_steps = [
            nn.Linear(latent_size, prev_step_size * calculate_im_size(self.IMAGE_SIZE, hidden_layers) ** 2),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(prev_step_size,
                                                  calculate_im_size(self.IMAGE_SIZE, hidden_layers),
                                                  calculate_im_size(self.IMAGE_SIZE, hidden_layers)))
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

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    @staticmethod
    def init_from_checkpoint(checkpoint_id, ratio, epoch, sub_epoch=1):
        checkpoint_file = [x for x in os.listdir(f"runs/{checkpoint_id}") if x.startswith("starting_params-")][0]
        checkpoint_settings = checkpoint_file.removesuffix("].pth").removeprefix("starting_params-[")
        checkpoint_settings = [int(x) for x in checkpoint_settings.split(",")]
        network = ConvAE(*checkpoint_settings)

        network.load_state_dict(torch.load(f"runs/{checkpoint_id}/{checkpoint_file}"))

        mask_file = f"runs/{checkpoint_id}/keep-{ratio}-epoch-{epoch}-{sub_epoch}.pth"

        masks = torch.load(mask_file)
        prune_model(network, masks)

        return network

