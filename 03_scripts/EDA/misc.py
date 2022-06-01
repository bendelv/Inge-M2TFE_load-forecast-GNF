import pandas as pd
import numpy as np
import seaborn as sns
from scipy.fft import fft, fftfreq
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot


def frequencies_content(dataraw_path):
    dr = pd.read_csv(dataraw_path, parse_dates=["ts"])
    plt.rc('font', size=15)
    tot_zone = dr[dr.zone == 'TOTAL']['demand'][dr.ts > '2003-03-01 00:00:00'][dr.ts < '2003-03-08 00:00:00']
    autocorrelation_plot(tot_zone)
    plt.savefig("plots/total_autocorrelation_week.png")
    plt.clf()

    tot_zone = dr[dr.zone == 'TOTAL']['demand'][dr.ts > '2003-03-01 00:00:00'][dr.ts < '2003-04-01 00:00:00']
    autocorrelation_plot(tot_zone)
    plt.savefig("plots/total_autocorrelation_month.png")
    plt.clf()

    tot_zone = dr[dr.zone == 'TOTAL']['demand'][dr.ts > '2003-03-01 00:00:00'][dr.ts < '2004-03-01 00:00:00']
    autocorrelation_plot(tot_zone)
    plt.savefig("plots/total_autocorrelation_year.png")
    plt.clf()


def display_loads(dataraw_path):
    # dr -> dataraw
    dr = pd.read_csv(dataraw_path, parse_dates=["ts"])
    dr = dr[(dr["zone"] != "TOTAL") & (dr["zone"] != "MASS")]

    grp_date = dr.groupby(by=["date"])

    zone_ordering = ['CT', 'ME', 'NH', 'RI', 'VT', 'NEMASSBOST', 'WCMASS', 'SEMASS']
    grpd_zone_ts = dr.groupby(by=["zone"])
    colors = px.colors.qualitative.Light24[:10]

    fig = go.Figure()
    for (zone, frame), col in zip(grpd_zone_ts, colors):
        fig.add_trace(go.Scatter(x=dr.ts.unique(), y=frame["demand"].values, line=dict(color=col), opacity=0.6,
                                 name=zone))
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Load (MW)",
        width=1000,
        height=500,
        font=dict(
            size=15
        )
    )
    fig.write_image("plots/load_zones.png")


def daily_load():
    train_load = np.load("../../02_datasets/Sets/train_load.npy")
    zone_ordering = ['CT', 'ME', 'NH', 'RI', 'VT', 'NEMASSBOST', 'WCMASS', 'SEMASS']
    means = train_load.mean(axis=0).reshape((8, 24))
    stds = train_load.std(axis=0).reshape((8, 24))

    for mean_zone, std_zone, zone in zip(means, stds, zone_ordering):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(24), y=mean_zone + std_zone, showlegend=False,
                                 line=dict(color="blue"), opacity=0.6))
        fig.add_trace(go.Scatter(x=np.arange(24), y=mean_zone - std_zone, fill='tonexty', line=dict(color="blue"),
                                 opacity=0.6, name="Standard deviation"))
        fig.add_trace(go.Scatter(x=np.arange(24), y=mean_zone, line=dict(color="red"), name="mean"))

        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Load demand (MW)",
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
                x=0.5,
                y=0.01
            )
        )

        fig.write_image("plots/daily_load{}.png".format(zone))


def main():
    dataraw_path = "../../02_datasets/Raw/gefcom2017-d.csv"
    # display_loads(dataraw_path)
    # frequencies_content(dataraw_path)
    daily_load()

    return 0


if __name__ == "__main__":
    main()
