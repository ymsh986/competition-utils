
import numpy as np
import pandas as pd
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

sns.set(style='white', context='notebook', palette='deep')

plotly_template = dict(
    layout=go.Layout(
        template='plotly_dark',
        font=dict(
            family="Franklin Gothic",
            size=12
        ),
        height=500,
        width=1000,
    )
)

color_palette = {
    'Bin': ['#016CC9', '#E876A3'],
    'Cat5': ['#E876A3', '#E0A224', '#63B70D', '#6BCFF6', '#13399E'],
    'Cat10': ['#E876A3', '#E0A224', '#63B70D', '#6BCFF6', '#13399E', '#E876A3', '#E0A224', '#63B70D', '#6BCFF6', '#13399E'],
}


def plot_feature_importance(feature_importance_df, top=50):

    feature_importance_df['avg'] = feature_importance_df.mean(axis=1)
    feature_importance_top = feature_importance_df.avg.nlargest(top).sort_values(ascending=True)

    pal = sns.color_palette("YlGnBu", top).as_hex()

    fig = go.Figure()
    for i in range(len(feature_importance_top.index)):
        fig.add_shape(
            dict(
                type="line",
                y0=i,
                y1=i,
                x0=0,
                x1=feature_importance_top[i],
                line_color=pal[::-1][i],
                opacity=0.8,
                line_width=4
            )
        )

    fig.add_trace(
        go.Scatter(
            x=feature_importance_top,
            y=feature_importance_top.index,
            mode='markers',
            marker_color=pal[::-1],
            marker_size=8,
            hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'
        )
    )

    fig.update_layout(
        template=plotly_template,
        title=f'LGBM Feature Importance<br>Top {top}',
        margin=dict(l=150, t=80),
        xaxis=dict(title='Importance', zeroline=False),
        yaxis_showgrid=False,
        height=1000,
        width=800
    )

    return fig, feature_importance_top


def plot_correlation_matrix(df_corr):
    fig = go.Figure(layout=plotly_template['layout'])

    fig.add_trace(
        go.Heatmap(
            x=df_corr.columns,
            y=df_corr.index,
            z=np.array(df_corr),
            name='Corr',
            colorscale='oxy'
        ),
    )

    fig.update_layout(
        width=1500,
        height=1500,
    )

    return fig


def plot_training_curve_keras(n_folds, training_history):
    fig = make_subplots(
        rows=2,
        cols=1,
        #start_cell='bottom-left', # どのセルを起点とするか
        shared_xaxes=False, # x軸を共有する場合
        shared_yaxes=False, # y軸を共有する場合
    )

    for i in range(n_folds):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(training_history[f'fold-{i}'].history['loss']))),
                y=training_history[f'fold-{i}'].history['loss'],
                name=f'fold-{i+1}',
                mode='lines',
                line=dict(color=color_palette['Cat5'][i]),
                legendgroup='1',
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(training_history[f'fold-{i}'].history['val_loss']))),
                y=training_history[f'fold-{i}'].history['val_loss'],
                name=f'fold-{i+1}',
                mode='lines',
                line=dict(color=color_palette['Cat5'][i]),
                legendgroup='2',
            ),
            row=2,
            col=1,
        )
    fig.update_layout(
        xaxis1_title='epoch',
        yaxis1_title = 'training-loss',
        xaxis2_title='epoch',
        yaxis2_title = 'valid-loss',
        legend_tracegroupgap = 200,
        width=1000,
        height=700,
    )

    return fig
