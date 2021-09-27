# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ------------------------------------
import torch
import torch.nn as nn 
import numpy as np
from vihds.config import Config
from vihds.encoders import Encoder
from vihds.decoders import Decoder

class BaseVAE(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(BaseVAE,self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.n_theta = None

    def sample_u(self, n_batch, n_samples, device=None):
        u = torch.tensor(np.random.randn(n_batch, n_samples, self.n_theta).astype(np.float32))
        return u
    
    def forward(self, data, samples, writer=None, epoch=None):
        '''
        Evaluate VAE model on data batch.
        - The data is a list of batches, each with a different ODE model and/or different vector of times.
        '''
        u = self.sample_u(len(data.inputs), samples)
        q = self.encoder(data)
        theta = q.sample(u, self.device)
        clipped_theta = self.encoder.p.clip(theta, stddevs=4)
        result, conditioned_theta = self.decoder(clipped_theta, data, writer, epoch)
        return result, conditioned_theta, q, self.encoder.p

def build_model(args, settings:Config, dataset, parameters):
    encoder = Encoder(parameters, dataset, args.verbose)
    
    # Specify whether the decoder should condition on device information
    if settings.data.device_depth > 1:
        decoder_condition_on_device = True
    else:
        print("- Only a single device being considered, so disabling device-conditioning in the decoder")
        decoder_condition_on_device = False

    decoder = Decoder(settings, decoder_condition_on_device)

    return BaseVAE(encoder, decoder, settings.device)