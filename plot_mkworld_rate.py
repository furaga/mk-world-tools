import argparse

import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html


class MKWorldRatePlotter:
    def __init__(self, file_path, line_color="#11ff11", show_num=50):
        self.file_path = file_path
        self.line_color = line_color
        self.show_num = show_num
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        initial_figure = go.Figure()

        self.app.layout = html.Div(
            [
                dcc.Graph(id="mkworld-rate-graph", figure=initial_figure),
                dcc.Interval(
                    id="interval-component",
                    interval=5 * 1000,
                    n_intervals=0,
                ),
            ]
        )

    def _setup_callbacks(self):
        @self.app.callback(
            Output("mkworld-rate-graph", "figure"),
            Input("interval-component", "n_intervals"),
        )
        def update_graph(n_intervals):
            file_path = self.file_path
            line_color = self.line_color
            show_num = self.show_num

            try:
                with open(file_path, "r", encoding="utf8") as f:
                    lines = f.readlines()[1:]
                    if not lines:
                        return go.Figure()

                    lines = lines[-show_num:]

                    if not lines:
                        return go.Figure()

                    my_rates = [int(line.split(",")[4]) for line in lines]
                    x = list(range(len(my_rates)))

                    starts = []
                    for i in range(len(lines) - 1):
                        try:
                            ts1 = lines[i].split(",")[0]
                            ts2 = lines[i + 1].split(",")[0]
                            if ts1.split("@")[0] != ts2.split("@")[0]:
                                starts.append(i + 1)
                        except IndexError:
                            continue

            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
                return go.Figure()
            except Exception as e:
                print(f"Error reading or processing file {file_path}: {e}")
                return go.Figure()

            if not my_rates:
                return go.Figure()

            updated_trace = go.Scatter(
                x=x,
                y=my_rates,
                mode="lines+markers",
                name="My Rate",
                yaxis="y1",
                line=dict(color=line_color, width=3),
                marker=dict(size=6, symbol="circle"),
            )

            fig = go.Figure(data=[updated_trace])

            fig.update_layout(
                paper_bgcolor="#0a0a0a",
                plot_bgcolor="#1a1a1a",
                font=dict(size=20, color="#ffffff", family="Arial Black"),
                title=dict(
                    text="<b>MK World Race Rating</b>",
                    font=dict(size=28, color="#ffdd00"),
                    x=0.5,
                    y=0.95,
                    xanchor="center",
                ),
                margin=dict(l=80, r=50, t=100, b=60),
            )

            shapes = [
                dict(
                    type="line",
                    x0=s,
                    x1=s,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="#888888", width=2, dash="dash"),
                )
                for s in starts
            ]
            if x:
                shapes.append(
                    dict(
                        type="line",
                        x0=min(x) if x else 0,
                        x1=max(x) if x else 0,
                        y0=9000,
                        y1=9000,
                        line=dict(color="#ff3333", width=2, dash="dot"),
                    )
                )
            fig.update_layout(shapes=shapes)

            fig.update_xaxes(
                showline=True,
                linewidth=3,
                linecolor="#ffffff",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, len(x) - 0.5] if x else [-0.5, 0.5],
            )

            if my_rates:
                min_rate = min(my_rates)
                max_rate = max(my_rates)
                rate_diff = max_rate - min_rate
                padding = max(rate_diff * 0.15, 100)
                range_min = min_rate - padding
                range_max = max_rate + padding
            else:
                range_min = 8000
                range_max = 10000

            fig.update_yaxes(
                showline=True,
                linewidth=3,
                linecolor="#ffffff",
                dtick=100,
                tickformat="%d",
                color="#ffffff",
                showgrid=True,
                gridcolor="#333333",
                gridwidth=1,
                range=[range_min, range_max],
                zeroline=False,
            )

            return fig

    def run(self, debug=True, host="127.0.0.1", port=8050):
        self.app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MK World Race rate plotting tool."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the CSV file containing race data",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="#11ff11",
        help="Line color for the plot (default: #11ff11)",
    )
    parser.add_argument(
        "--show_num",
        type=int,
        default=50,
        help="Number of recent data points to show (default: 50)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number (default: 8050)",
    )

    args = parser.parse_args()

    plotter = MKWorldRatePlotter(
        file_path=args.file, line_color=args.color, show_num=args.show_num
    )
    plotter.run(debug=True, host=args.host, port=args.port)