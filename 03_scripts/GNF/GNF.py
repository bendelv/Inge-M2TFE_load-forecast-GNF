import os
from itertools import product
import pandas as pd
import numpy as np
import torch
import json

from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import NF
from EDA.correlations import compute_mi
from utils import generate_days
from metrics import *


def retrain_GNF(device, nb_epoch, model_dict, train_load, train_context, val_load, val_context):
    if model_dict['a_type'] == 'ARmat' or model_dict['a_type'] == 'Learnedmat':
        DIR_MODEL = "GNF_{}".format(model_dict['a_type'])
    elif model_dict['a_type'] == 'MImat' or model_dict['a_type'] =='PCmat':
        DIR_MODEL = "GNF_{}_{}".format(model_dict['a_type'], model_dict['thresh'])
    else:
        print("Adjacency type not accepted.")
        return -1

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    GNF_flow = prepare_GNF(device, model_dict, None, None, None, None)
    nb_epoch_check = 5
    if not os.path.exists("{}/images".format(DIR_MODEL)):
        os.makedirs("{}/images".format(DIR_MODEL))
    opt = torch.optim.Adam(GNF_flow.parameters(), model_dict['lr'], weight_decay=model_dict['wd'])
    train_loss = np.load("{}/train_loss.npy".format(DIR_MODEL)).tolist()
    val_loss = np.load("{}/val_loss.npy".format(DIR_MODEL)).tolist()

    print("Start GNF re training.")
    for epoch in range(nb_epoch):
        loss_tot = 0
        for X, context in zip(train_load, train_context):
            cur_x = torch.tensor(X, dtype=torch.float).to(device)
            context = torch.tensor(context, dtype=torch.float).to(device)

            z, jac = GNF_flow(cur_x, context)
            loss = GNF_flow.loss(z, jac)
            loss_tot += loss.detach()
            # print(GNF_flow.getConditioners()[0].A)

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        GNF_flow.step(epoch, loss_tot)

        if epoch % nb_epoch_check == 0:
            mean_train_loss = loss_tot / (train_load.shape[0])
            train_loss.append(mean_train_loss.cpu())
            print("Epoch {} Mean Loss: {:3f}".format(epoch, mean_train_loss))

            mean_val_loss = 0
            for X, context in zip(val_load, val_context):
                cur_x = torch.tensor(X, dtype=torch.float).to(device)
                context = torch.tensor(context, dtype=torch.float).to(device)
                z, jac = GNF_flow(cur_x, context)
                loss = GNF_flow.loss(z, jac)
                mean_val_loss += loss.detach().cpu()

            val_loss.append(mean_val_loss / val_load.shape[0])

    print("\nGNF training finished!\n")
    np.save("{}/train_loss{}.npy".format(DIR_MODEL, str(50+nb_epoch)), train_loss)
    np.save("{}/val_loss{}.npy".format(DIR_MODEL, str(50+nb_epoch)), val_loss)
    plot_losses(train_loss, val_loss, nb_epoch_check, DIR_MODEL)
    torch.save(GNF_flow.state_dict(), "{}/GNF{}.pt".format(DIR_MODEL, str(50+nb_epoch)))
    print('Saving model to {}..\n'.format(DIR_MODEL))


def prepare_GNF(device, model_dict, train_load, train_context, val_load, val_context):

    print("model dictionary: ", model_dict)
    if model_dict['a_type'] == 'ARmat' or model_dict['a_type'] == 'Learnedmat':
        DIR_MODEL = "GNF_{}".format(model_dict['a_type'])
    elif model_dict['a_type'] == 'MImat' or model_dict['a_type'] == 'PCmat':
        DIR_MODEL = "GNF_{}_{}".format(model_dict['a_type'], model_dict['thresh'])
    else:
        print("Adjacency type not accepted.")
        return -1

    # construct adjacency matrix
    A = None
    if train_load is None:
        pass
    else:
        if model_dict['a_type'] == "ARmat":
            A = np.ones((model_dict["in_size"], model_dict["in_size"]))
            A = np.tril(A, -1)
            A = torch.Tensor(A)
            print(A)

        elif model_dict['a_type'] == "PCmat":
            train_load_df = pd.DataFrame(train_load.reshape(-1, 192))
            corr = train_load_df.corr()
            corr = corr.abs()
            corr[corr >= model_dict['thresh']] = 1
            corr[corr < model_dict['thresh']] = 0
            A = corr.to_numpy()
            A = np.tril(A, -1)
            A = torch.Tensor(A)
            print(A)

        elif model_dict['a_type'] == "MImat":
            mi = compute_mi(train_load.reshape(-1, 192))
            mi[mi >= model_dict['thresh']] = 1
            mi[mi < model_dict['thresh']] = 0
            A = np.tril(mi, -1)
            A = torch.Tensor(A)
            print(A)

        elif model_dict['a_type'] == "Learnedmat":
            A = None
        else:
            print("Adjacency type not accepted.")
            return -1

    conditioner_args = dict(
        hidden=model_dict['cond_hidden'],
        in_size=model_dict['in_size'],
        cond_in=model_dict['cond_in'],
        out_size=model_dict['out_size'],
        nb_epoch_update=10,
        A_prior=A,
        hot_encoding=True
    )
    normalizer_args = dict(
        integrand_net=model_dict['norm_hidden'],
        cond_size=model_dict['out_size'] + model_dict['in_size'],
        nb_steps=model_dict['steps_integ']
    )

    conditioner = NF.DAGConditioner(**conditioner_args)
    normalizer = NF.MonotonicNormalizer(**normalizer_args)
    flow_steps = [NF.NormalizingFlowStep(conditioner, normalizer)]
    GNF_flow = NF.FCNormalizingFlow(flow_steps, NF.NormalLogDensity())
    GNF_flow.to(device)

    nb_epoch = 100
    if os.path.exists("{}/GNF{}.pt".format(DIR_MODEL, nb_epoch)):
        print('Loading model from {}..\n'.format(DIR_MODEL))
        GNF_flow.load_state_dict(torch.load("{}/GNF{}.pt".format(DIR_MODEL, nb_epoch)))
    else:
        nb_epoch_check = 5
        if not os.path.exists("{}/images".format(DIR_MODEL)):
            os.makedirs("{}/images".format(DIR_MODEL))
        opt = torch.optim.Adam(GNF_flow.parameters(), model_dict['lr'], weight_decay=model_dict['wd'])
        train_loss = []
        val_loss = []

        print("Start GNF training.")
        for epoch in range(nb_epoch):
            loss_tot = 0
            for X, context in zip(train_load, train_context):
                cur_x = torch.tensor(X, dtype=torch.float).to(device)
                context = torch.tensor(context, dtype=torch.float).to(device)

                z, jac = GNF_flow(cur_x, context)
                loss = GNF_flow.loss(z, jac)
                loss_tot += loss.detach()
                # print(GNF_flow.getConditioners()[0].A)

                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
            GNF_flow.step(epoch, loss_tot)

            if epoch % nb_epoch_check == 0:
                mean_train_loss = loss_tot / (train_load.shape[0])
                train_loss.append(mean_train_loss.cpu())
                print("Epoch {} Mean Loss: {:3f}".format(epoch, mean_train_loss))

                mean_val_loss = 0
                for X, context in zip(val_load, val_context):
                    cur_x = torch.tensor(X, dtype=torch.float).to(device)
                    context = torch.tensor(context, dtype=torch.float).to(device)
                    z, jac = GNF_flow(cur_x, context)
                    loss = GNF_flow.loss(z, jac)
                    mean_val_loss += loss.detach().cpu()

                val_loss.append(mean_val_loss / val_load.shape[0])

        print("\nGNF training finished!\n")
        np.save("{}/train_loss.npy".format(DIR_MODEL), train_loss)
        np.save("{}/val_loss.npy".format(DIR_MODEL), val_loss)
        plot_losses(train_loss, val_loss, nb_epoch_check, DIR_MODEL)
        torch.save(GNF_flow.state_dict(), "{}/GNF{}.pt".format(DIR_MODEL, nb_epoch))
        print('Saving model to {}..\n'.format(DIR_MODEL))
    return GNF_flow


def test_GNF(model_dict):
    load = np.load("../../02_datasets/Sets/test_load.npy")
    context = np.load("../../02_datasets/Sets/test_context.npy")

    std_target = StandardScaler()
    load = std_target.fit_transform(load)
    
    if model_dict['a_type'] == 'ARmat' or model_dict['a_type'] == 'Learnedmat':
        DIR_MODEL = "GNF_{}".format(model_dict['a_type'])
    elif model_dict['a_type'] == 'MImat' or model_dict['a_type'] =='PCmat':
        DIR_MODEL = "GNF_{}_{}".format(model_dict['a_type'], model_dict['thresh'])
    else:
        print("Adjacency type not accepted.")
        return -1

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    GNF_flow = prepare_GNF(device, model_dict, None, None, load, context)

    n_scen = 10
    n_days = 200
    compute_scores = True
    plot_scens = True
    if (plot_scens or compute_scores) is True:
        GNF_scens = generate_days(GNF_flow, device, n_days, n_scen, context)

    if plot_scens is True:
        n_plotdays = 10
        # plot some of generated days and correlation plots

        target = std_target.inverse_transform(load[:n_plotdays])
        scens = std_target.inverse_transform(GNF_scens[:n_plotdays].reshape(-1, 192))
        scens = scens.reshape(target.shape[0], n_plotdays, 192)
        plot_days_scenarios(DIR_MODEL, target, scens)
        plot_corr_scens(directory=DIR_MODEL, days_scens=GNF_scens)

    if compute_scores is True:
        print("CRPS score")
        CRPS = []
        for day_scens, target in zip(GNF_scens, load[:n_days]):
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
        es = energy_score(GNF_scens, np.array(load)[:n_days])
        es_mean, es_std = np.array(es).mean(), np.array(es).std()
        print(es_mean, es_std, "\n")
        with open('{}/array_es.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, es)
        f.close()

        print("Variogram score")
        vs = variogram_score(GNF_scens, np.array(load)[:n_days], 1)
        vs_mean, vs_std = np.array(vs).mean(), np.array(vs).std()
        print(vs_mean, vs_std)
        with open('{}/array_vs.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, vs)
        f.close()

        print("Variogram zone score")
        vsz = variogram_zone_score(GNF_scens, np.array(load)[:n_days], 1)
        print(np.array(vsz))
        vsz_mean, vsz_std = np.array(vsz).mean(axis=0), np.array(vsz).std(axis=0)
        print(vsz_mean, vsz_std)
        with open('{}/array_vsz.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, vsz)
        f.close()


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
    train_context = train_context.reshape((-1, BATCH_SIZE, 198))

    val_std_target = StandardScaler()
    val_load = val_std_target.fit_transform(val_load)
    val_load = val_load.reshape((-1, BATCH_SIZE, 192))
    val_context = val_context.reshape((-1, BATCH_SIZE, 198))

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    # ----------------------------------------- GNFs -------------------------------------------

    def grid_hparameters(hparameters):
        for params in product(*hparameters.values()):
            yield dict(zip(hparameters.keys(), params))

    dict_hparameters = dict(
        lr=[0.001],
        wd=[0.1],
        in_size=[192],
        cond_in=[198],
        out_size=[20],
        cond_hidden=[[300, 300, 300]],
        norm_hidden=[[200, 200, 200, 200]],
        steps_integ=[20],
        a_type=['MImat'],
        thresh=[0.8]
    )

    for model_hparams in grid_hparameters(dict_hparameters):
        #GNF_flow = prepare_GNF(device, model_hparams, train_load, train_context, val_load, val_context)
        retrain_GNF(device, 50, model_hparams, train_load, train_context, val_load, val_context)


if __name__ == '__main__':

    model_dict = dict(
        lr=0.001,
        wd=0.1,
        in_size=192,
        cond_in=198,
        out_size=20,
        cond_hidden=[300, 300, 300],
        norm_hidden=[200, 200, 200, 200],
        steps_integ=20,
        a_type='MImat',
        thresh=0.8
    )

    test_GNF(model_dict)
    #main()
