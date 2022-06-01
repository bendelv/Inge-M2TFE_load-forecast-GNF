from itertools import product
import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import NF

from utils import generate_days
from metrics import *


def retrain_NF(device, nb_epoch, model_dict, train_load, train_context, val_load, val_context):
    conditioner_args = dict(
        hidden=model_dict['cond_hidden'],
        in_size=model_dict['in_size'],
        cond_in=model_dict['cond_in'],
        out_size=model_dict['out_size'],
    )

    if model_dict['norm_type'] == 'monotonic':
        DIR_MODEL = "NFm{}_{}_{}_{}_{}_{}_{}".format(model_dict['out_size'], str(model_dict['lr']),
                                                     str(model_dict['wd']), str(model_dict['nb_steps']),
                                                     str(model_dict['steps_integ']),
                                                     str(model_dict['cond_hidden']), str(model_dict['norm_hidden']))

        normalizer_args = dict(
            integrand_net=model_dict['norm_hidden'],
            cond_size=model_dict['out_size'],
            nb_steps=model_dict['steps_integ']
        )
        normalizer = NF.MonotonicNormalizer(**normalizer_args)
    elif model_dict['norm_type'] == 'affine':
        DIR_MODEL = "NFa2_{}_{}_{}_{}".format(str(model_dict['lr']), str(model_dict['wd']), str(model_dict['nb_steps']),
                                              str(model_dict['cond_hidden']))
        conditioner_args['out_size'] = 2
        model_dict['out_size'] = 2
        normalizer = NF.AffineNormalizer()
    else:
        print("Normalizer type not allowed.")
        return -1

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    NF_flow = prepare_NF(device, model_dict, None, None, None, None)
    nb_epoch_check = 5
    if not os.path.exists("{}/images".format(DIR_MODEL)):
        os.makedirs("{}/images".format(DIR_MODEL))
    opt = torch.optim.Adam(NF_flow.parameters(), model_dict['lr'], weight_decay=model_dict['wd'])
    train_loss = np.load("{}/train_loss.npy".format(DIR_MODEL)).tolist()
    val_loss = np.load("{}/val_loss.npy".format(DIR_MODEL)).tolist()

    print("NF start re training")
    for epoch in np.arange(nb_epoch):
        loss_tot = 0
        for X, context in zip(train_load, train_context):
            cur_X = torch.Tensor(X).float().to(device)
            context = torch.Tensor(context).float().to(device)

            z, jac = NF_flow(cur_X, context=context)
            loss = NF_flow.loss(z, jac)
            loss_tot += loss.detach()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % nb_epoch_check == 0:
            mean_train_loss = loss_tot / (train_load.shape[0])
            train_loss.append(mean_train_loss.cpu())

            mean_val_loss = 0
            for X, context in zip(val_load, val_context):
                cur_x = torch.tensor(X, dtype=torch.float).to(device)
                context = torch.tensor(context, dtype=torch.float).to(device)
                z, jac = NF_flow(cur_x, context)
                loss = NF_flow.loss(z, jac)
                mean_val_loss += loss.detach().cpu()

            val_loss.append(mean_val_loss / val_load.shape[0])
            print("Epoch {} Mean Loss: {:3f}".format(epoch, mean_train_loss))
            torch.save(NF_flow.state_dict(), "{}/NF.pt".format(DIR_MODEL))

    print("\nNF training finished!\n")
    np.save("{}/train_loss{}.npy".format(DIR_MODEL, str(50+nb_epoch)), train_loss)
    np.save("{}/val_loss{}.npy".format(DIR_MODEL, str(50+nb_epoch)), val_loss)
    plot_losses(train_loss, val_loss, nb_epoch_check, DIR_MODEL)
    torch.save(NF_flow.state_dict(), "{}/NF{}.pt".format(DIR_MODEL, str(50+nb_epoch)))
    print('Saving model to {}..\n'.format(DIR_MODEL))


def prepare_NF(device, model_dict, train_load, train_context, val_load, val_context):
    """
    Train the NF if necessary.
    Load an existing model if the corresponding set of hyperparameters was already tested.
    """

    conditioner_args = dict(
        hidden=model_dict['cond_hidden'],
        in_size=model_dict['in_size'],
        cond_in=model_dict['cond_in'],
        out_size=model_dict['out_size'],
    )

    if model_dict['norm_type'] == 'monotonic':
        DIR_MODEL = "NFm{}_{}_{}_{}_{}_{}_{}".format(model_dict['out_size'], str(model_dict['lr']),
                                                     str(model_dict['wd']), str(model_dict['nb_steps']),
                                                     str(model_dict['steps_integ']),
                                                     str(model_dict['cond_hidden']), str(model_dict['norm_hidden']))

        normalizer_args = dict(
            integrand_net=model_dict['norm_hidden'],
            cond_size=model_dict['out_size'],
            nb_steps=model_dict['steps_integ']
        )
        normalizer = NF.MonotonicNormalizer(**normalizer_args)
    elif model_dict['norm_type'] == 'affine':
        DIR_MODEL = "NFa2_{}_{}_{}_{}".format(str(model_dict['lr']), str(model_dict['wd']), str(model_dict['nb_steps']),
                                              str(model_dict['cond_hidden']))
        conditioner_args['out_size'] = 2
        model_dict['out_size'] = 2
        normalizer = NF.AffineNormalizer()
    else:
        print("Normalizer type not allowed.")
        return -1

    print("model dictionary: ", model_dict)
    conditioner = NF.AutoregressiveConditioner(**conditioner_args)

    flow_steps = [NF.NormalizingFlowStep(conditioner, normalizer)] * model_dict['nb_steps']
    NF_flow = NF.FCNormalizingFlow(flow_steps, NF.NormalLogDensity())
    NF_flow.to(device)

    nb_epoch = 100
    if os.path.exists("{}/NF{}.pt".format(DIR_MODEL, nb_epoch)):
        print('Loading model from {}..\n'.format(DIR_MODEL))
        NF_flow.load_state_dict(torch.load("{}/NF{}.pt".format(DIR_MODEL, nb_epoch)))
    else:
        nb_epoch_check = 5
        if not os.path.exists("{}/images".format(DIR_MODEL)):
            os.makedirs("{}/images".format(DIR_MODEL))

        opt = torch.optim.Adam(NF_flow.parameters(), model_dict['lr'], weight_decay=model_dict['wd'])
        train_loss = []
        val_loss = []
        print("NF start training")
        for epoch in np.arange(nb_epoch):
            loss_tot = 0
            for X, context in zip(train_load, train_context):
                cur_X = torch.Tensor(X).float().to(device)
                context = torch.Tensor(context).float().to(device)

                z, jac = NF_flow(cur_X, context=context)
                loss = NF_flow.loss(z, jac)
                loss_tot += loss.detach()
                opt.zero_grad()
                loss.backward()
                opt.step()

            if epoch % nb_epoch_check == 0:
                mean_train_loss = loss_tot / (train_load.shape[0])
                train_loss.append(mean_train_loss.cpu())

                mean_val_loss = 0
                for X, context in zip(val_load, val_context):
                    cur_x = torch.tensor(X, dtype=torch.float).to(device)
                    context = torch.tensor(context, dtype=torch.float).to(device)
                    z, jac = NF_flow(cur_x, context)
                    loss = NF_flow.loss(z, jac)
                    mean_val_loss += loss.detach().cpu()

                val_loss.append(mean_val_loss / val_load.shape[0])
                print("Epoch {} Mean Loss: {:3f}".format(epoch, mean_train_loss))
                torch.save(NF_flow.state_dict(), "{}/NF.pt".format(DIR_MODEL))

        np.save("{}/train_loss.npy".format(DIR_MODEL), train_loss)
        np.save("{}/val_loss.npy".format(DIR_MODEL), val_loss)
        plot_losses(train_loss, val_loss, nb_epoch_check, DIR_MODEL)

        torch.save(NF_flow.state_dict(), "{}/NF{}.pt".format(DIR_MODEL, nb_epoch))
        print('Saving model to {}..\n'.format(DIR_MODEL))
    return NF_flow


def test_NF(model_dict):
    load = np.load("../../02_datasets/Sets/test_load.npy")
    context = np.load("../../02_datasets/Sets/test_context.npy")

    std_target = StandardScaler()
    load = std_target.fit_transform(load)

    if model_dict['norm_type'] == 'monotonic':
        DIR_MODEL = "NFm{}_{}_{}_{}_{}_{}_{}".format(model_dict['out_size'], str(model_dict['lr']),
                                                     str(model_dict['wd']), str(model_dict['nb_steps']),
                                                     str(model_dict['steps_integ']),
                                                     str(model_dict['cond_hidden']), str(model_dict['norm_hidden']))

    else:
        DIR_MODEL = "NFa2_{}_{}_{}_{}".format(str(model_dict['lr']), str(model_dict['wd']), str(model_dict['nb_steps']),
                                              str(model_dict['cond_hidden']))

    device = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('device used: ', device, "\n")

    NF_flow = prepare_NF(device, model_dict, None, None, load, context)

    n_scen = 10
    n_days = 200
    compute_scores = True
    plot_scens = True
    if (plot_scens or compute_scores) is True:
        NF_scens = generate_days(NF_flow, device, n_days, n_scen, context)

    if plot_scens is True:
        n_plotdays = 10
        # plot some of generated days and correlation plots

        target = std_target.inverse_transform(load[:n_plotdays])
        scens = std_target.inverse_transform(NF_scens[:n_plotdays].reshape(-1, 192))
        scens = scens.reshape(target.shape[0], n_plotdays, 192)
        plot_days_scenarios(DIR_MODEL, target, scens)
        plot_corr_scens(directory=DIR_MODEL, days_scens=NF_scens)

    if compute_scores is True:
        # compute scores
        print("CRPS score")
        CRPS = []
        for day_scens, target in zip(NF_scens, load[:n_days]):
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
        es = energy_score(NF_scens, np.array(load)[:n_days])
        es_mean, es_std = np.array(es).mean(), np.array(es).std()
        print(es_mean, es_std, "\n")
        with open('{}/array_es.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, es)
        f.close()

        print("Variogram score")
        vs = variogram_score(NF_scens, np.array(load)[:n_days], 1)
        vs_mean, vs_std = np.array(vs).mean(), np.array(vs).std()
        print(vs_mean, vs_std)
        with open('{}/array_vs.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, vs)
        f.close()

        print("Variogram zone score")
        vsz = variogram_zone_score(NF_scens, np.array(load)[:n_days], 1)
        vsz_mean, vsz_std = np.array(vsz).mean(axis=0), np.array(vsz).std(axis=0)
        print(vsz_mean, vsz_std)
        with open('{}/array_vsz.npy'.format(DIR_MODEL), 'wb') as f:
            np.save(f, vsz)
        f.close()


def main():
    print(os.getcwd())
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

    # ----------------------------------------- NFs -------------------------------------------
    def grid_hparameters(hparameters):
        for params in product(*hparameters.values()):
            yield dict(zip(hparameters.keys(), params))

    dict_hparameters = dict(
        lr=[0.001],
        wd=[0.1],
        in_size=[192],
        cond_in=[198],
        out_size=[20],
        nb_steps=[1],
        cond_hidden=[[300, 300, 300]],
        norm_hidden=[[200, 200, 200, 200]],
        steps_integ=[20],
        norm_type=['monotonic']
    )

    for model_hparams in grid_hparameters(dict_hparameters):
        #NF_flow = prepare_NF(device, model_hparams, train_load, train_context, val_load, val_context)
        retrain_NF(device, 50, model_hparams, train_load, train_context, val_load, val_context)


if __name__ == '__main__':
    os.chdir("..")
    model_dict = dict(
        lr=0.001,
        wd=0.1,
        in_size=192,
        cond_in=198,
        out_size=20,
        nb_steps=1,
        cond_hidden=[300, 300, 300],
        norm_hidden=[200, 200, 200, 200],
        steps_integ=20,
        norm_type='monotonic'
    )
    test_NF(model_dict)

    #main()
