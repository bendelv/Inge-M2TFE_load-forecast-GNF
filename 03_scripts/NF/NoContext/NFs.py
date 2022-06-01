import NF
import json
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objs as go

from metrics import *


def generate_days(flow, device, n_scen, n_gendays):
    print("Generating scenarios based on given contexts..")
    scens = []
    for i in range(n_gendays):
        print("\r{}/{} day generation.".format(i, n_gendays), end="")

        # scenarios from GNF
        scen_samples = []
        for j in range(n_scen):
            z = torch.randn(192).unsqueeze(0).to(device)
            scen_sample = flow.invert(z).detach().cpu().numpy().reshape((-1))
            scen_samples.append(scen_sample)
        scens.append(scen_samples)

    scens = np.array(scens)
    print("\nDone!\n")
    return scens


def main():
    train_load = np.load("../../../02_datasets/Sets/train_load.npy")
    val_load = np.load("../../../02_datasets/Sets/val_load.npy")

    train_load = train_load[:int(train_load.shape[0] / 10) * 10]
    val_load = val_load[:int(val_load.shape[0] / 10) * 10]

    BATCH_SIZE = 10

    train_std_target = StandardScaler()
    train_load = train_std_target.fit_transform(train_load)
    train_load = train_load.reshape((-1, BATCH_SIZE, 192))

    val_std_target = StandardScaler()
    val_load = val_std_target.fit_transform(val_load)
    val_load = val_load.reshape((-1, BATCH_SIZE, 192))

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    # ----------------------------------------- NF -------------------------------------------
    DICT_MODEL = "NoContext"
    f = open('../models_files/{}.json'.format(DICT_MODEL))
    conditioner_args = json.load(f)
    nb_steps = conditioner_args.pop("nb_steps")

    nb_epoch = 100

    conditioner = NF.AutoregressiveConditioner(**conditioner_args)
    normalizer = NF.AffineNormalizer()

    flow_steps = [NF.NormalizingFlowStep(conditioner, normalizer)]
    NF_flow = NF.FCNormalizingFlow(flow_steps, NF.NormalLogDensity())
    NF_flow.to(device)

    trained = True
    MODEL_PATH = "NF.pt".format(DICT_MODEL)
    if trained:
        print('Loading model from {}..\n'.format(MODEL_PATH))
        NF_flow.load_state_dict(torch.load(MODEL_PATH))
    else:
        opt = torch.optim.Adam(NF_flow.parameters(), 1e-3, weight_decay=1e-5)
        train_loss = []
        val_loss = []
        print("NF start training")
        for epoch in np.arange(nb_epoch):
            loss_tot = 0
            for X in train_load:
                cur_X = torch.Tensor(X).float().to(device)

                z, jac = NF_flow(cur_X)
                loss = NF_flow.loss(z, jac)
                loss_tot += loss.detach()
                opt.zero_grad()
                loss.backward()
                opt.step()

            if epoch % 1 == 0:
                mean_train_loss = loss_tot / (train_load.shape[0])
                train_loss.append(mean_train_loss.cpu())
                print("Epoch {} Mean Loss: {:3f}".format(epoch, mean_train_loss))

        torch.save(NF_flow.state_dict(), MODEL_PATH)
        print('Saving model to {}..'.format(MODEL_PATH))

    val_load = val_load.reshape((-1, 192))

    n_scen = 10
    n_gendays = 10
    NF_scens = generate_days(NF_flow, device, n_scen, n_gendays)

    # plot some of generated days and correlation plots

    directory = "."
    target = val_std_target.inverse_transform(val_load[:n_gendays])
    scens = train_std_target.inverse_transform(NF_scens[:n_gendays].reshape(-1, 192))
    scens = scens.reshape(target.shape[0], n_scen, 192)
    plot_days_scenarios(directory, target, scens[:n_gendays])
    plot_corr_scens(directory=directory, days_scens=NF_scens)
    """
    #compute scores
    print("CRPS score")
    NF_CRPS = []
    for day_scens, target in zip(NF_scens, val_load):
        day_CRPS = instant_CRPS(day_scens, target)
        NF_CRPS.append(day_CRPS)
    NF_CRPS = np.array(NF_CRPS).reshape((-1))
    with open('array_crps.npy', 'wb') as f:
        np.save(f, NF_CRPS)
    f.close()

    print("Energy score")
    NF_es = energy_score(NF_scens, np.array(val_load))
    NF_es_mean, NF_es_var = np.array(NF_es).mean(), np.array(NF_es).var()
    print(NF_es_mean, NF_es_var, "\n")

    print("Variogram score")
    NF_vs = variogram_score(NF_scens, np.array(val_load), 1)
    NF_vs_mean, NF_vs_var = np.array(NF_vs).mean(), np.array(NF_vs).var()
    print(NF_vs_mean, NF_vs_var)

    with open('array_es.npy', 'wb') as f:
        np.save(f, NF_es)
    f.close()
    with open('array_vs.npy', 'wb') as f:
        np.save(f, NF_vs)
    f.close()
    """

if __name__ == '__main__':
    main()
