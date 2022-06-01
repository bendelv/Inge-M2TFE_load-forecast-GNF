import os
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import seaborn as sns
import kaleido


def instant_CRPS(dayzones_scens, day_obs):
    """

    """
    day_crps = []
    for ensemble, obs in zip(dayzones_scens.T, day_obs):
        ensemble = np.sort(ensemble.flatten())
        l = len(ensemble)
        b1 = (1 / l) * sum(ensemble)
        b2 = 0
        for i, e in enumerate(ensemble):
            b2 += i * e
        b2 = (1 / (l * (l - 1))) * b2
        day_crps.append(((1 / l) * sum(abs(ensemble - obs))) + b1 - 2 * b2)

    return day_crps


def plot_day_CRPS(day_mean_CRPS, day_std_CRPS, path):
    colors = px.colors.qualitative.Light24[:10]
    zones = ['CT', 'ME', 'NH', 'RI', 'VT', 'NEMASSBOST', 'WCMASS', 'SEMASS']

    fig = go.Figure()
    day_mean_CRPS = day_mean_CRPS.reshape(8, 24)
    x = np.arange(0, 24, 1)
    for mean, std, zone, col in zip(day_mean_CRPS, day_std_CRPS, zones, colors):
        u = mean+std
        l = mean-std

        fig.add_trace(go.Scatter(x=x, y=mean, line=dict(color=col), name=zone))
        #fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]], axis=None),
        #                         y=np.concatenate([u, l], axis=None), fill='toself', showlegend=False))

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Load demand",
        font=dict(
            size=15
        ),
        yaxis_range=[0.5, 1.5],
        margin=dict(
            l=0,
            r=20,
            b=0,
            t=20,
            pad=0
        )
    )

    fig.write_image("{}/images/day_crps.png".format(path))
    return 0


def energy_score(s: np.array, y_true: np.array):
    """
    Compute the Energy score (ES).
    :param s: scenarios of shape (n_days, n_s, periods)
    :param y_true: observations of shape = (n_days, periods)
    :return: the ES per day of the testing set.
    """
    n_periods = y_true.shape[1]
    n_d = len(y_true)  # number of days

    n_s = s.shape[1]  # number of scenarios per day
    es = []
    # loop on all days
    for d in range(n_d):
        # select a day for both the scenarios and observations
        s_d = s[d, :, :]
        y_d = y_true[d, :]

        # compute the part of the ES
        simple_sum = np.mean([np.linalg.norm(s_d[s, :] - y_d) for s in range(n_s)])

        # compute the second part of the ES
        double_somme = 0
        for i in range(n_s):
            for j in range(n_s):
                double_somme += np.linalg.norm(s_d[i, :] - s_d[j, :])
        double_sum = double_somme / (2 * n_s * n_s)

        # ES per day
        es_d = simple_sum - double_sum
        es.append(es_d)
    return es


def variogram_score(s: np.array, y_true: np.array, beta: float):
    """
    Compute the Variogram score (VS).
    :param s: scenarios of shape (n_days, n_s, 24*zones)
    :param y_true: observations of shape = (n_days, 24)
    :param beta: order of the VS
    :return: the VS per day of the testing set.
    """
    n_periods = y_true.shape[1]
    n_d = len(y_true)  # number of days
    n_s = s.shape[1]  # number of scenarios per day
    weights = 1  # equal weights across all hours of the day
    vs = []
    # loop on all days
    for d in range(n_d):
        # select a day for both the scenarios and observations
        s_d = s[d, :, :]
        y_d = y_true[d, :]

        # Double loop on time periods of the day
        vs_d = 0
        for k1 in range(n_periods):
            for k2 in range(n_periods):
                # VS first part
                first_part = np.abs(y_d[k1] - y_d[k2]) ** beta
                second_part = 0
                # Loop on all scenarios to compute VS second part
                for i in range(n_s):
                    second_part += np.abs(s_d[i, k1] - s_d[i, k2]) ** beta
                second_part = second_part / n_s
                vs_d += weights * (first_part - second_part) ** 2
        # VS per day
        vs.append(vs_d)
    return vs


def variogram_zone_score(s: np.array, y_true: np.array, beta: float):
    """
    Compute the Variogram score (VS).
    :param s: scenarios of shape (n_days, n_s, 24*zones)
    :param y_true: observations of shape = (n_days, 24*zones)
    :param beta: order of the VS
    :return: the VS per day per zone of the testing set. 
    """
    n_periods = y_true.shape[1]
    n_d = len(y_true)  # number of days
    n_s = s.shape[1]  # number of scenarios per day
    s = s.reshape((n_d, n_s, -1, 24))
    y_true = y_true.reshape((n_d, -1, 24))
    weights = 1  # equal weights across all hours of the day
    vs = []
    # loop on all days
    for d in range(n_d):
        vs_zones = []
        for z in range(8):
            # select a day for both the scenarios and observations
            s_d = s[d, :, z, :]
            y_d = y_true[d, z, :]

            # Double loop on time periods of the day
            vs_zone = 0
            for k1 in range(24):
                for k2 in range(24):
                    # VS first part
                    first_part = np.abs(y_d[k1] - y_d[k2]) ** beta
                    second_part = 0
                    # Loop on all scenarios to compute VS second part
                    for i in range(n_s):
                        second_part += np.abs(s_d[i, k1] - s_d[i, k2]) ** beta
                    second_part = second_part / n_s
                    vs_zone += weights * (first_part - second_part) ** 2
            vs_zones.append(vs_zone)

        # VS per day
        vs.append(vs_zones)

    return vs


def plot_days_scenarios(directory, targets, days_scens):
    colors = px.colors.qualitative.Light24[:10]
    zones = ['CT', 'ME', 'NH', 'RI', 'VT', 'NEMASSBOST', 'WCMASS', 'SEMASS']

    for i, (day_obs, day_scens) in enumerate(zip(targets, days_scens)):
        day_obs = day_obs.reshape((8, 24))
        fig = go.Figure()

        for zone_obs, zone, col in zip(day_obs, zones, colors):
            fig.add_trace(go.Scatter(x=np.arange(24), y=zone_obs, line=dict(color=col), name=zone))

        for scen in day_scens:
            scen = scen.reshape((8, 24))

            for zone, col in zip(scen, colors):
                fig.add_trace(go.Scatter(x=np.arange(24), y=zone, line=dict(color=col, dash='dot'), opacity=0.6,
                                         showlegend=False))

            fig.update_layout(
                xaxis_title="Hour",
                yaxis_title="Load demand",
                font=dict(
                    size=15
                ),
                margin=dict(
                    l=0,
                    r=20,
                    b=0,
                    t=20,
                    pad=0
                )
            )
        fig.write_image("{}/images/valday{}.png".format(directory, i))

    print("Day scenarios plotted.\n")
    return 0


def plot_corr_scens(directory, days_scens):
    days_scens = np.array(days_scens)
    corr_mat = np.zeros((days_scens.shape[2], days_scens.shape[2]))
    for day in days_scens:
        day = np.transpose(day, (1, 0))
        corr = np.corrcoef(day)
        corr_mat = corr_mat + corr

    corr_mat = corr_mat / (days_scens.shape[0])
    zones = [corr_mat[i * 24:i * 24 + 24, i * 24:i * 24 + 24] for i in np.arange(0, 8)]
    names = ['CT', 'ME', 'NH', 'RI', 'VT', 'NEMASSBOST', 'WCMASS', 'SEMASS']
    for zone, name in zip(zones, names):
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        # cmap = sns.diverging_palette(229, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(zone, square=True, linewidths=.1, cbar_kws={"shrink": .5}, vmin=-0.2, vmax=1)
        plt.savefig("{}/images/scenarios_corr_{}.png".format(directory, name))


def plot_losses(train_loss, val_loss, nb_epoch_check, directory, layout_dict = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, len(train_loss) * nb_epoch_check, nb_epoch_check), y=train_loss,
                             line=dict(color='red'), name="train loss"))
    fig.add_trace(
        go.Scatter(x=np.arange(0, len(val_loss) * nb_epoch_check, nb_epoch_check), y=val_loss, line=dict(color='blue'),
                   name="validation loss"))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Negative log likelihood",
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
    if layout_dict is not None:
        fig.update_layout(**layout_dict)

    fig.write_image("{}/images/losses_{}.png".format(directory, directory))

    return 0


def boxplot_scores(models_points_score, dict_layout, path):
    fig = go.Figure()

    for model_points_score, model_name in zip(models_points_score, dict_layout['models_name']):
        fig.add_trace(go.Box(y=model_points_score, boxmean='sd', name=model_name))
    fig.update_layout(
        yaxis_title=dict_layout['score_name'],
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
        showlegend=False
    )
    fig.write_image(path)


def compute_super_zones(day_scens):
    day_scens = np.array(day_scens).reshape((-1, 8, 24))
    mass_sum = np.sum(day_scens[:, 6:, :], axis=1)
    tot_sum = np.sum(day_scens, axis=1)

    return mass_sum, tot_sum


def main():
    os.chdir("GNF")
    directory = "GNF_MImat_1.1"
    train_loss=np.load("{}/train_loss.npy".format(directory))[:50]
    val_loss=np.load("{}/val_loss.npy".format(directory))[:50]

    layout_dict = dict(
        yaxis_title="Negative log likelihood",
        yaxis_range=[-170, 170]
    )
    plot_losses(train_loss, val_loss, 5, directory, layout_dict)


if __name__ == "__main__":
    main()