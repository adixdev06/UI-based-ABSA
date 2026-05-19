"""
Visualizer — Plotly chart helpers for the ABSA dashboard.
All charts use a consistent dark theme.
"""

import plotly.graph_objects as go
import plotly.express as px

DARK_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e2e8f0', family='Space Grotesk'),
    margin=dict(l=0, r=0, t=30, b=0),
)
GRID = dict(gridcolor='#1e2d45')


class Visualizer:

    @staticmethod
    def stacked_bar(aspect_data):
        aspects = list(aspect_data.keys())
        pos = [aspect_data[a]['positive'] for a in aspects]
        neg = [aspect_data[a]['negative'] for a in aspects]
        neu = [aspect_data[a]['neutral'] for a in aspects]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Positive', x=aspects, y=pos, marker_color='#10b981'))
        fig.add_trace(go.Bar(name='Negative', x=aspects, y=neg, marker_color='#ef4444'))
        fig.add_trace(go.Bar(name='Neutral',  x=aspects, y=neu, marker_color='#f59e0b'))

        fig.update_layout(
            barmode='stack', **DARK_THEME, height=320,
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1),
            xaxis=GRID, yaxis=GRID,
        )
        return fig

    @staticmethod
    def donut(labels, values, colors=None):
        colors = colors or ['#10b981', '#ef4444', '#f59e0b']
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.6,
            marker_colors=colors,
        ))
        fig.update_layout(**DARK_THEME, height=320,
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        return fig

    @staticmethod
    def radar(labels, values):
        labels_closed = labels + [labels[0]]
        values_closed = values + [values[0]]
        fig = go.Figure(go.Scatterpolar(
            r=values_closed, theta=labels_closed,
            fill='toself',
            fillcolor='rgba(0,212,255,0.1)',
            line=dict(color='#00d4ff', width=2),
            marker=dict(color='#7c3aed', size=8),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor='#1e2d45', color='#64748b'),
                angularaxis=dict(gridcolor='#1e2d45', color='#e2e8f0'),
            ),
            **DARK_THEME, height=320,
        )
        return fig

    @staticmethod
    def line_chart(x, y, x_title='', y_title=''):
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode='lines+markers',
            line=dict(color='#00d4ff', width=2.5),
            marker=dict(size=10, color='#7c3aed',
                        line=dict(color='#00d4ff', width=2)),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
        ))
        fig.update_layout(
            **DARK_THEME, height=280,
            xaxis=dict(title=x_title, **GRID),
            yaxis=dict(title=y_title, **GRID),
        )
        return fig

    @staticmethod
    def horizontal_bar(names, values, colors=None):
        if colors is None:
            colors = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b',
                      '#ef4444', '#ec4899', '#8b5cf6', '#06b6d4']
        fig = go.Figure(go.Bar(
            x=values, y=names, orientation='h',
            marker=dict(color=colors[:len(names)], line_width=0),
            text=values, textposition='outside',
            textfont=dict(color='#e2e8f0'),
        ))
        fig.update_layout(
            **DARK_THEME, height=280,
            xaxis=GRID, yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        )
        return fig
