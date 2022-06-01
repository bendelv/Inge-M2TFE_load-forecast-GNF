import torch
import numpy as np
import random


def generate_days(flow, device, n_days, n_scen, val_context):
    print("Generating scenarios based on given contexts..")
    random.shuffle(val_context)
    val_context = val_context[:n_days]

    scens = []
    for i, day_context in enumerate(val_context):
        print("\r{}/{} day generation.".format(i, len(val_context)), end="")
        day_context = torch.Tensor(day_context).unsqueeze(0).to(device)

        # scenarios from GNF
        scen_samples = []
        for j in range(n_scen):
            z = torch.randn(192).unsqueeze(0).to(device)
            scen_sample = flow.invert(z, context=day_context).detach().cpu().numpy().reshape((-1))
            scen_samples.append(scen_sample)
        scens.append(scen_samples)

    scens = np.array(scens)
    print("\nDone!\n")
    return scens


def generate_days_VAE(vae, device, n_days, n_scen, val_context):
    print("Generating scenarios based on given contexts..")
    random.shuffle(val_context)
    val_context = val_context[:n_days]
    scens = []
    for i, day_context in enumerate(val_context):
        print("\r{}/{} day generation.".format(i, len(val_context)), end="")
        #day_context = torch.Tensor(day_context).unsqueeze(0).to(device)
        scens.append(vae.sample(n_scen, day_context))

    scens = np.array(scens)
    print("\nDone!\n")
    return scens
