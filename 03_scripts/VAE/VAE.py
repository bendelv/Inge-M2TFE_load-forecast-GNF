import math
import os
import pickle
from itertools import product

import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import torch
import random

import pandas as pd
import numpy as np
import torch.nn as nn

from metrics import *
from sklearn.preprocessing import StandardScaler
from utils import *


class VAElinear(nn.Module):
    """
    VAE from A. Wehenkel with Linear layers.
    """

    def __init__(self, **kwargs):
        super(VAElinear, self).__init__()
        self.latent_s = kwargs['latent_s']  # Dim of the latent space
        self.in_size = kwargs['in_size']  # Dim of the random variable to model (PV, wind power, etc)
        self.cond_in = kwargs['cond_in']  # Dim of context (weather forecasts, etc)

        # Set GPU if available
        if kwargs['gpu']:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'

        # Build the Encoder & Decoder with Linear layers
        # x = target variable to generate samples (PV, wind, load, etc)
        # y = conditional variable (weather forecasts, etc)

        # The goal of the encoder is to model the posterior q(z|x,y)
        # q(z|x,y) is parametrized with an inference network NN_phi  that takes as input x,y
        # -> The output of the encoder are the mean and log of the covariance mu_z and log_sigma_z
        l_enc_net = [self.in_size + self.cond_in] + kwargs['enc'] + [2 * self.latent_s]

        # The goal of the decoder is to model the likelihood p(x|z,y)
        # p(x|z,y) is parametrized with a generative network NN_theta that takes as input z
        # The output of the decoder is the approximation of p(x|z,y) with z sampled from N(g(x),h(x)) = N(mu_z, sigma_z)
        l_dec_net = [self.latent_s + self.cond_in] + kwargs['dec'] + [self.in_size]

        # Build the Encoder
        self.enc_net = []
        for l1, l2 in zip(l_enc_net[:-1], l_enc_net[1:]):
            self.enc_net += [nn.Linear(l1, l2), nn.ReLU()]
        self.enc_net.pop()
        self.enc = nn.Sequential(*self.enc_net)

        # Build the Decoder
        self.dec_net = []
        for l1, l2 in zip(l_dec_net[:-1], l_dec_net[1:]):
            self.dec_net += [nn.Linear(l1, l2), nn.ReLU()]
        self.dec_net.pop()
        self.dec = nn.Sequential(*self.dec_net)

    def loss(self, x0, cond_in=None):
        """
        VAE loss function.
        Cf formulatio into the paper.
        :param x0: the random variable to fit.
        :param cond_in: context such as weather forecasts, etc.
        :return: the loss function = ELBO.
        """

        bs = x0.shape[0]
        if cond_in is None:
            cond_in = torch.empty(bs, self.cond_in)

        # Encoding -> NN_φ
        # The encoder ouputs mu_φ and log_sigma_φ
        #     μ(x,φ), log σ(x,φ)**2  = NN_φ(x)
        #     q(z∣x; φ)       = N(z; μ, σ^2*I)

        mu_phi, log_sigma_phi = torch.split(self.enc(torch.cat((x0, cond_in), 1)), self.latent_s, 1)
        # KL divergence
        # KL(q(z∣x; φ)∣∣p(z)) = 1/2 ∑ [1 + log(σ (x; φ)**2) − μ (x; φ)**2 − σ (x; φ)**2 ] because q(z∣x; φ) and p(z)follow a Normal distribution
        KL_phi = 0.5 * (1 + log_sigma_phi - mu_phi ** 2 - torch.exp(log_sigma_phi))
        KL_phi_new = - KL_phi.sum(1).mean(0)

        # old KL_phi
        # KL_phi_old = (-log_sigma_phi + (mu_phi ** 2) / 2 + torch.exp(log_sigma_phi) / 2).sum(1).mean(0)

        # The reparameterization trick:
        z = mu_phi + torch.exp(log_sigma_phi) * torch.randn(mu_phi.shape, device=self.device)

        # Decoding -> NN_θ
        mu_x_pred = self.dec(torch.cat((z, cond_in), 1))

        # E_q(z∣x;ν) [log p(x∣z;θ)] ≃ ∑ log(p(x∣z;θ)) ≃ || [x - μ(x; θ)] / σ(x; θ) || ** 2 (MSE) because p(x∣z;θ) follows a Normal distribution
        KL_x = ((mu_x_pred.view(bs, -1) - x0) ** 2).view(bs, -1).sum(1).mean(0)

        loss = KL_x + KL_phi_new

        return loss

    def forward(self, x0):
        mu_z, log_sigma_z = torch.split(self.enc(x0.view(-1, *self.img_size)), self.latent_s, 1)
        mu_x_pred = self.dec(mu_z + torch.exp(log_sigma_z) * torch.randn(mu_z.shape, device=self.device))
        return mu_x_pred

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def sample(self, n_s=1, x_cond: np.array = None):
        """
        :param n_s: number of scenarios
        :param x_cond: context (weather forecasts, etc) into an array of shape (self.cond_in,)
        :return: samples into an array of shape (nb_samples, self.in_size)
        """
        # Generate samples from a multivariate Gaussian
        z = torch.randn(n_s, self.latent_s).to(self.device)

        context = torch.tensor(np.tile(x_cond, n_s).reshape(n_s, self.cond_in)).to(self.device).float()
        scenarios = self.dec(torch.cat((z, context), 1)).view(n_s, -1).cpu().detach().numpy()

        return scenarios


def train(vae, train_load, train_context, val_load, val_context, epochs, device):
    vae = vae.to(device)
    opt = torch.optim.Adam(vae.parameters())

    loss_list = []
    for epoch in range(epochs):
        i = 0
        loss_batch = 0
        for x, x_context in zip(train_load, train_context):
            loss = vae.loss(x0=torch.tensor(x).to(device).float(),
                            cond_in=torch.tensor(x_context).to(device).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_batch += loss.item()
            i += 1

        # LS loss is the average over all the batch
        loss_ls = loss_batch / i
        i = 0
        loss_vs_batch = 0
        for xv, xv_context in zip(val_load, val_context):
            loss = vae.loss(x0=torch.tensor(xv).to(device).float(),
                            cond_in=torch.tensor(xv_context).to(device).float())
            loss_vs_batch += loss.item()
            i += 1
        loss_vs = loss_vs_batch / i
        loss_list.append([loss_ls, loss_vs])
        print("Epoch {} Mean Loss: {:3f}".format(epoch, loss_ls))

    return vae, loss_list


def test_VAE(model_dict):
    load = np.load("../../02_datasets/Sets/test_load.npy")
    context = np.load("../../02_datasets/Sets/test_context.npy")

    std_target = StandardScaler()
    load = std_target.fit_transform(load)

    DIR_MODEL = "VAE_{}_{}_{}".format(model_dict['latent_s'], model_dict['enc'], model_dict['dec'])

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    vae = prepare_VAE(device, model_dict, None, None, load, context)

    n_scen = 10
    n_days = 200
    compute_scores = True
    plot_scens = True
    if (plot_scens or compute_scores) is True:
        VAE_scens = generate_days_VAE(vae, device, n_days, n_scen, context)

    if plot_scens is True:
        n_plotdays = 10
        # plot some of generated days and correlation plots

        target = std_target.inverse_transform(load[:n_plotdays])
        scens = std_target.inverse_transform(VAE_scens[:n_plotdays].reshape(-1, 192))
        scens = scens.reshape(target.shape[0], n_plotdays, 192)
        plot_days_scenarios(DIR_MODEL, target, scens)
        plot_corr_scens(directory=DIR_MODEL, days_scens=VAE_scens)

    if compute_scores is True:
        # compute scores

        print("CRPS score")
        CRPS = []
        for day_scens, target in zip(VAE_scens, load[:n_days]):
            day_CRPS = instant_CRPS(day_scens, target)
            CRPS.append(day_CRPS)
        CRPS = np.array(CRPS)
        mean_CRPS = CRPS.mean(axis=0)
        std_CRPS = CRPS.std(axis=0)
        plot_day_CRPS(mean_CRPS, std_CRPS, DIR_MODEL)

        with open('{}/array_crps.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, CRPS)
        f.close()

        print("Energy score")
        es = energy_score(VAE_scens, np.array(load)[:n_days])
        es_mean, es_std = np.array(es).mean(), np.array(es).std()
        print(es_mean, es_std, "\n")
        with open('{}/array_es.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, es)
        f.close()

        print("Variogram score")
        vs = variogram_score(VAE_scens, np.array(load)[:n_days], 1)
        vs_mean, vs_std = np.array(vs).mean(), np.array(vs).std()
        print(vs_mean, vs_std)
        with open('{}/array_vs.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, vs)
        f.close()

        print("Variogram zone score")
        vsz = variogram_zone_score(VAE_scens, np.array(load)[:n_days], 1)
        vsz_mean, vsz_std = np.array(vsz).mean(axis=0), np.array(vsz).std(axis=0)
        print(vsz_mean, vsz_std)
        with open('{}/array_vsz.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, vsz)
        f.close()


def prepare_VAE(device, model_dict, train_load, train_context, val_load, val_context):
    DIR_MODEL = "VAE_{}_{}_{}".format(str(model_dict['latent_s']), str(model_dict['enc']),
                                      str(model_dict['dec']))
    vae = VAElinear(latent_s=model_dict['latent_s'], cond_in=model_dict['cond_in'], in_size=model_dict['in_size'],
                    enc=model_dict['enc'], dec=model_dict['dec'], gpu=True)
    vae = vae.to(device)
    nb_epochs = 100
    if os.path.exists("{}/VAE{}.pt".format(DIR_MODEL, nb_epochs)):
        print('Loading model from {}..\n'.format(DIR_MODEL))
        vae.load_state_dict(torch.load("{}/VAE{}.pt".format(DIR_MODEL, nb_epochs)))
    else:
        if not os.path.exists("{}/images".format(DIR_MODEL)):
            os.makedirs("{}/images".format(DIR_MODEL))

        vae, loss_lists = train(vae, train_load, train_context, val_load, val_context, nb_epochs, device)
        torch.save(vae.state_dict(), "{}/VAE{}.pt".format(DIR_MODEL, nb_epochs))
        print("\nVAE training finished!\n")
        print('Saving model to {}..'.format(DIR_MODEL))

        loss_lists = np.array(loss_lists)
        train_loss = loss_lists[:, 0]
        val_loss = loss_lists[:, 1]
        np.save("{}/train_loss.npy".format(DIR_MODEL), train_loss)
        np.save("{}/val_loss.npy".format(DIR_MODEL), val_loss)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(0, len(train_loss)), y=train_loss,
                                 line=dict(color='red'), name="train loss"))
        fig.add_trace(
            go.Scatter(x=np.arange(0, len(val_loss)), y=val_loss,
                       line=dict(color='blue'),
                       name="validation loss"))
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="ELBO",
            font=dict(
                size=15
            ),
            margin=dict(
                l=0,
                r=20,
                b=0,
                t=20,
                pad=0
            ),
            legend=dict(
                x=0.7,
                y=0.9
            )
        )
        fig.write_image("{}/images/losses_{}.png".format(DIR_MODEL, DIR_MODEL))
    return vae


def main():
    train_load = np.load("../../02_datasets/Sets/train_load.npy")
    val_load = np.load("../../02_datasets/Sets/val_load.npy")

    train_context = np.load("../../02_datasets/Sets/train_context.npy")
    val_context = np.load("../../02_datasets/Sets/val_context.npy")

    train_load = train_load[:int(train_load.shape[0] / 10) * 10]
    train_context = train_context[:int(train_context.shape[0] / 10) * 10]

    val_load = val_load[:int(val_load.shape[0] / 10) * 10]
    val_context = val_context[:int(val_context.shape[0] / 10) * 10]
    BATCH_SIZE = 10

    train_std_target = StandardScaler()
    train_load = train_std_target.fit_transform(train_load)
    train_load = train_load.reshape((-1, BATCH_SIZE, 192))

    val_std_target = StandardScaler()
    val_load = val_std_target.fit_transform(val_load)
    val_load = val_load.reshape((-1, BATCH_SIZE, 192))

    train_context = train_context.reshape((-1, BATCH_SIZE, 198))
    val_context = val_context.reshape((-1, BATCH_SIZE, 198))

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    # --------------------------------------- VAEs ------------------------------------------
    def grid_hparameters(hparameters):
        for params in product(*hparameters.values()):
            yield dict(zip(hparameters.keys(), params))

    dict_hparameters = dict(
        lr=[0.001],
        wd=[0.1],
        in_size=[192],
        cond_in=[198],
        latent_s=[10, 50, 100],
        enc=[[200, 200, 200], [300, 300, 300], [200, 200, 200, 200]],
        dec=[[200, 200, 200], [300, 300, 300], [200, 200, 200, 200]]
    )

    for model_hparams in grid_hparameters(dict_hparameters):
        vae = prepare_VAE(device, model_hparams, train_load, train_context, val_load, val_context)


if __name__ == "__main__":
    #main()
    model_dict = dict(
        lr=0.001,
        wd=0.1,
        in_size=192,
        cond_in=198,
        latent_s=50,
        enc=[300, 300, 300],
        dec=[300, 300, 300]
    )
    test_VAE(model_dict)
