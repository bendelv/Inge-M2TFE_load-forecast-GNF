import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from metrics import *
import os


def main():
    # gather scores in different arrays
    crps_file = "array_crps.npy"
    es_file = "array_es.npy"
    vs_file = "array_vs.npy"
    vsz_file = "array_vsz.npy"
    print(os.getcwd())
    dirs = ["../Gaussians/",
            "../VAE/VAE_50_[300, 300, 300]_[300, 300, 300]/",
            "../NF/NFm20_0.001_0.1_1_20_[300, 300, 300]_[200, 200, 200, 200]/",
            "../GNF/GNF_ARmat/",
            "../GNF/GNF_PCmat_0.9/",
            "../GNF/GNF_MImat_0.8/"]

    crps_files = [d+crps_file for d in dirs]
    es_files = [d+es_file for d in dirs]
    vs_files = [d+vs_file for d in dirs]
    vsz_files = [d+vsz_file for d in dirs]

    crps_scores = [np.load(csf) for csf in crps_files]
    es_scores = [np.load(esf) for esf in es_files]
    vs_scores = [np.load(vsf) for vsf in vs_files]
    vsz_scores = [np.load(vszf) for vszf in vsz_files]

    zone_ordering = ['CT', 'ME', 'NH', 'RI', 'VT', 'NEMASSBOST', 'WCMASS', 'SEMASS']
    names = ["Gauss", "VAE", "ANF", "AR GNF", "PC GNF", "MI GNF"]

    # summary information of CRPS per zone

    crps_scores = np.array(crps_scores).reshape((6, -1, 192))
    fig = go.Figure(data=[go.Bar(name=name, x=zone_ordering, y=crps_score.mean(axis=0).reshape(8, 24).mean(axis=1)) for name, crps_score in zip(names, crps_scores)])
    fig.update_layout(
        font=dict(
            size=20
        ),
        width=1000,
        height=500
    )
    fig.write_image("images/CRPS_zones.png")

    # plot quartiles, means and variances of each model for crps, es, vs scores
    score_name = "CRPS"
    dict_layout = dict(
        models_name=names,
        score_name=score_name
    )
    boxplot_scores(crps_scores.mean(axis=2), dict_layout, "images/finalmodels_boxplot_{}.png".format(score_name))

    score_name = "ES"
    dict_layout = dict(
        models_name=names,
        score_name=score_name
    )
    boxplot_scores(es_scores, dict_layout, "images/finalmodels_boxplot_{}.png".format(score_name))

    score_name = "VS"
    dict_layout = dict(
        models_name=names,
        score_name=score_name
    )
    boxplot_scores(vs_scores, dict_layout, "images/finalmodels_boxplot_{}.png".format(score_name))


    # table generation of vs zone score
    vsz_table = [[' '] + zone_ordering]
    for name, vsz_score in zip(names, vsz_scores):
        vsz_score_mean = vsz_score.mean(axis=0)
        vsz_score_std = vsz_score.std(axis=0)
        vsz_method = [name]
        for zone_mean, zone_std in zip(vsz_score_mean, vsz_score_std):
            vsz_score_str = str(round(zone_mean, 2)) + " +- " +str(round(zone_std, 2))
            vsz_method.append(vsz_score_str)
        vsz_table.append(vsz_method)

    vsz_table = pd.DataFrame(vsz_table)
    vsz_table = vsz_table.set_index(keys=0)
    vsz_table.columns = vsz_table.iloc[0]
    vsz_table = vsz_table.drop(vsz_table.index[0])
    vsz_table = vsz_table.transpose().to_latex()
    #print(vsz_table)

    CRPSs = np.array(crps_scores).reshape((6, -1)).mean(axis=1)
    print(CRPSs.shape)
    ESs = np.array(es_scores).mean(axis=1)
    VSs = np.array(vs_scores).mean(axis=1)
    scores = ['averaged CRPS', 'ES', 'VS']
    sum_table = [[' '] + scores]
    for name, aCRPS, ES, VS in zip(names, CRPSs, ESs, VSs):
        sum_table.append([name,
                          str(round(aCRPS, 2)),
                          str(round(ES, 2)),
                          str(round(VS, 2))
                          ])
    sum_table = pd.DataFrame(sum_table)
    sum_table = sum_table.set_index(keys=0)
    sum_table.columns = sum_table.iloc[0]
    sum_table = sum_table.drop(sum_table.index[0])
    sum_table = sum_table.to_latex()
    print(sum_table)

    """
    es_scores = np.array(es_scores)
    es_stats = [(round(es_score.mean(), 3), round(es_score.std(), 3)) for es_score in es_scores]
    vs_scores = np.array(vs_scores)
    vs_stats = [(round(vs_score.mean(), 3), round(vs_score.std(), 3)) for vs_score in vs_scores]
    print(es_stats)
    print(vs_stats)
    """


if __name__ == "__main__":
    main()