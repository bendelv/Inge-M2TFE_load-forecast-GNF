import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import torch
import NF


def scenarios_plot(generated_days, val_load, directory):
    colors = px.colors.qualitative.Light24[:10]
    for j, generated_day, val_day in enumerate(zip(generated_days, val_load)):
        val_day = val_day.reshape((8, 24))
        fig = go.Figure()

        for zone_obs in val_day:
            fig.add_trace(go.Scatter(x=np.arange(24), y=zone_obs, line=dict(color='black')))

        for scen in generated_day:
            scen = scen.reshape((8, 24))
            for zone, col in zip(scen, colors):
                fig.add_trace(go.Scatter(x=np.arange(24), y=zone, line=dict(color=col)))

        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="",
            showlegend=False,
            font=dict(
                size=20
            )
        )
        fig.write_image("{}/GNF_scenarios{}.png".format(directory, j))
