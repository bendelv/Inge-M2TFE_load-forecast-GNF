import sys

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy, chi2_contingency
from researchpy.correlation import corr_case
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def compute_corr(train_load):
    # pearson corr

    train_load_df = pd.DataFrame(train_load)
    _, corr, pval = corr_case(train_load_df)
    corr = train_load_df.corr()
    corr = np.tril(corr.to_numpy())
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(229, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, cmap=cmap, mask=mask, vmin=0.8, center=0.8,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("plots/pearsoncorr.png")


# Mutual information
def compute_mi(train_load):
    def calc_MI(x, y, bins):
        c_xy = np.histogram2d(x, y, bins)[0]
        c_xy[c_xy == 0] = 1 / (bins * bins)

        g, p, dof, expected = chi2_contingency(c_xy, correction=False, lambda_="log-likelihood")
        mi = 0.5 * g / c_xy.sum()
        return mi

    nbin = 50
    A = train_load
    n = A.shape[1]
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix + 1, n):
            matMI[ix, jx] = calc_MI(A[:, ix], A[:, jx], nbin)
    matMI = matMI.T

    if __name__ == "__main__":
        listmatMI = matMI[np.tril_indices(n)]
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(229, 20, as_cmap=True)
        mask = np.zeros_like(matMI, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(matMI, cmap=cmap, mask=mask, center=0.5,
                    square=True, linewidths=.1, cbar_kws={"shrink": .5})
        plt.savefig("plots/mutualinfo.png")
    return matMI


def corr_zone(load):
    load_df = pd.DataFrame(load)
    load_corr = load_df.corr()
    load_corr = load_corr.to_numpy()

    zones_corr = np.zeros(shape=(8, 24, 24))
    names = ['CT', 'ME', 'NH', 'RI', 'VT', 'NEMASSBOST', 'WCMASS', 'SEMASS']
    for name, i in zip(names, np.arange(8)):
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(229, 20, as_cmap=True)
        sns.heatmap(load_corr[i * 24:(i + 1) * 24, i * 24:(i + 1) * 24], cmap=cmap, vmin=0, vmax=1, center=0.5,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.savefig("plots/corr_{}.png".format(name))


def differentiatePCvsMI(train_load, PCthresh, MIthresh):
    mi = compute_mi(train_load)
    mi[mi >= MIthresh] = 1
    mi[mi < MIthresh] = 0
    A_MI = np.tril(mi, -1)

    train_load_df = pd.DataFrame(train_load.reshape(-1, 192))
    corr = train_load_df.corr().abs()
    corr[corr >= PCthresh] = 1
    corr[corr < PCthresh] = 0
    A_PC = corr.to_numpy()
    A_PC = np.tril(A_PC, -1)

    diff = np.zeros((A_PC.shape[0], A_PC.shape[1]))
    diff[(A_PC == A_MI) & (A_PC == 1)] = -2
    diff[(A_PC == A_MI) & (A_PC == 0)] = -1
    diff[(A_PC > A_MI)] = 1
    diff[(A_PC < A_MI)] = 2

    print(len(diff[diff == 1]))
    print(len(diff[diff == 2]))
    print(len(diff[diff == -1]))
    print(len(diff[diff == -2]))
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom discrete colormap
    myColors = ((0.8, 0.0, 0.0, 1.0), (0.8, 0.8, 0.8, 1.0), (0.0, 0.0, 0.8, 1.0), (0.0, 0.8, 0.0, 1.0))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

    # Draw the heatmap with the mask and correct aspect ratio
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(diff, cmap=cmap, mask=mask, square=True, linewidths=.5, vmin=-2, vmax=2)
    # Manually specify colorbar labelling after it's been generated
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-1.5, -0.5, 0.5, 1.5])
    colorbar.set_ticklabels(['PC & MI = 1', 'PC & MI = 0', 'PC > MI', 'PC < MI'])

    plt.savefig("plots/differentiatePC{}vsMI{}.png".format(PCthresh, MIthresh))


def main():
    train_load = np.load("../../02_datasets/Sets/train_load.npy")
    # compute_corr(train_load)
    # compute_mi(train_load)
    # corr_zone(train_load)
    differentiatePCvsMI(train_load, 0.9, 0.8)
    """
    len_train_load = len(train_load)
    train_load_rnd = train_load
    np.random.shuffle(train_load_rnd.reshape((-1)))
    print(train_load_rnd)
    train_load_rnd = train_load_rnd.reshape((len_train_load, -1))
    compute_mi(train_load_rnd)
    """


if __name__ == "__main__":
    main()
