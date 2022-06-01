import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import math
from scipy.stats import norm


def crps_illustration():
    x = np.arange(-4, 4, 0.5)
    det = np.heaviside(x, 0)
    stoch = norm.cdf(x)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=x + 1, y=det, line=dict(color='blue', shape='hv'), name="Observation")
    )
    fig.add_trace(
        go.Scatter(x=x, y=stoch, line=dict(color='red'), name="Sample cdf", fill='tonexty')
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis={'tickmode': 'array', 'tickvals': []},
        yaxis={'tickmode': 'array', 'tickvals': []},
        xaxis_zeroline=False,
        yaxis_zeroline=False,
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
            x=0.1,
            y=0.8
        )
    )

    fig.write_image("../images/crps_illustration.png")


def AGNFvsANF_loss():
    AGNF_train_loss = np.load("GNF/GNF_ARmat/train_loss.npy")
    AGNF_val_loss = np.load("GNF/GNF_ARmat/val_loss.npy")
    ANF_train_loss = np.load("NF/NFm20_0.001_0.1_1_20_[300, 300, 300]_[200, 200, 200, 200]/train_loss.npy")
    ANF_val_loss = np.load("NF/NFm20_0.001_0.1_1_20_[300, 300, 300]_[200, 200, 200, 200]/val_loss.npy")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, 50, 5), y=AGNF_train_loss, line=dict(color='red'), name="AGNF train loss"))
    fig.add_trace(go.Scatter(x=np.arange(0, 50, 5), y=AGNF_val_loss, line=dict(color='red', dash='dash'),
                             name="AGNF validation loss"))
    fig.add_trace(go.Scatter(x=np.arange(0, 50, 5), y=ANF_train_loss, line=dict(color='blue'), name="ANF train loss"))
    fig.add_trace(go.Scatter(x=np.arange(0, 50, 5), y=ANF_val_loss, line=dict(color='blue', dash='dash'),
                             name="ANF validation loss"))
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
            x=0.6,
            y=0.9
        )
    )
    fig.write_image("../images/AGNFvsANF_loss.png")


if __name__ == "__main__":
    #crps_illustration()
    AGNFvsANF_loss()
