import numpy as np
import plotly.graph_objs as go


def main():
    train_load = np.load("../../02_datasets/Sets/train_load.npy")
    val_load = np.load("../../02_datasets/Sets/val_load.npy")

    train_load = train_load.reshape((-1, 8, 24))
    train_load = train_load[:, 0, :]

    # Single quantile model
    q50 = np.quantile(train_load, 0.5, axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(24), y=q50, line=dict(dash='dash', color='rgb(52, 152, 219)'),
                             name="Quantile forecast (alpha=0.5)"))
    fig.add_trace(go.Scatter(x=np.arange(24), y=val_load[0], name="Observation"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Load demand",
        font = dict(size=15),
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
    fig.write_image("plots/forecast_singleq.png")

    # Interval model
    q05 = np.quantile(train_load, 0.05, axis=0)
    q95 = np.quantile(train_load, 0.95, axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(24), y=q05, line=dict(dash='dash', color='rgb(52, 152, 219)'), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(24), y=q95, line=dict(dash='dash', color='rgb(52, 152, 219)'), fill='tonexty',
                             name="Interval 0.05-0.95"))
    fig.add_trace(go.Scatter(x=np.arange(24), y=val_load[0], line=dict(color='red'), name="Observation"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Load demand",
        font=dict(size=15),
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
    fig.write_image("plots/forecast_intervalq.png")

    qs = np.arange(0.1, 1, 0.1)

    fig = go.Figure()
    for q1, q2 in zip(qs[:int(len(qs) / 2)], qs[::-1][:int(len(qs) / 2)]):
        qb = np.quantile(train_load, q1, axis=0)
        qu = np.quantile(train_load, q2, axis=0)
        fig.add_trace(go.Scatter(x=np.arange(24), y=qb, line=dict(dash='dash', color='rgba(52, 152, 219, {})'.format(q1*2)), showlegend=False))
        fig.add_trace(go.Scatter(x=np.arange(24), y=qu, line=dict(dash='dash', color='rgba(52, 152, 219, {})'.format(q1*2)), fill='tonexty',
                                 fillcolor='rgba(52, 152, 219, {})'.format(q1*2), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(24), y=val_load[0], line=dict(color='red'), name="Observation"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Load demand",
        font=dict(size=15),
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
    fig.write_image("plots/forecast_density.png")

    fig = go.Figure()

    for i in range(20):
        fig.add_trace(go.Scatter(x=np.arange(24), y=train_load[i], showlegend=False, line=dict(dash='dot', color='rgb(52, 152, 219)')))
    fig.add_trace(go.Scatter(x=np.arange(24), y=val_load[0], line=dict(color='red'), name="Observation"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Load demand",
        font=dict(size=15),
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
    fig.write_image("plots/forecast_scenarios.png")

    mean = train_load.mean(axis=0)
    std = (mean - train_load).std(axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(24), y=mean, line=dict(color="blue"), name="Determinitic model",
                             error_y=dict(
                                 type='data',
                                 array=std,
                                 color="purple",
                                 thickness=1.5,
                                 width=3,
                             )
                             ))
    fig.add_trace(go.Scatter(x=np.arange(24), y=val_load[0], line=dict(color="red"), name="Observation"))
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Load demand",
        font=dict(size=15),
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
    fig.write_image("plots/errorbars.png")
    return 0


if __name__ == "__main__":
    main()
