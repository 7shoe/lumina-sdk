"""Plotting methods to allow visualiziation of the power network"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

def plot_network(
    data,
    mode="combined",
    title=None,
    pos=None,
    labels=None,
    node_size=40,
    generator_size=160,
    flow_scale=6,
    voltage_limits=(0.94, 1.06),
    figsize=(12, 10),
    show=True,
):
    """Plot OPFData: topology, voltage, flow, or combined. Return (fig, ax)."""
    if mode not in {"topology", "voltage", "flow", "combined"}:
        raise ValueError("mode must be topology, voltage, flow, or combined")

    use_voltage = mode in {"voltage", "combined"}
    use_flow = mode in {"flow", "combined"}
    n = data["bus"].num_nodes
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n))

    for kind, style in [("ac_line", "solid"), ("transformer", "dashed")]:
        key = ("bus", kind, "bus")
        if key not in data.edge_types:
            continue
        store = data[key]
        edges = store.edge_index.detach().cpu().t().tolist()
        flows = (
            store.edge_label[:, 0].detach().cpu().abs().tolist()
            if use_flow else [0.0] * len(edges)
        )
        for (u, v), flow in zip(edges, flows):
            graph.add_edge(u, v, style=style, flow=flow)

    if pos is None:
        topology = nx.Graph(graph)
        pos = (
            nx.kamada_kawai_layout(topology, weight=None)
            if n <= 300 else nx.spring_layout(topology, seed=42, weight=None)
        )
    if labels is None:
        labels = n <= 150

    sizes = np.full(n, node_size, dtype=float)
    gen_key = ("generator", "generator_link", "bus")
    if mode != "topology" and gen_key in data.edge_types:
        generators = data[gen_key].edge_index[1].detach().cpu().unique().numpy()
        sizes[generators] = generator_size

    fig, ax = plt.subplots(figsize=figsize)
    for style in ["solid", "dashed"]:
        edges = [(u, v, attrs) for u, v, attrs in graph.edges(data=True)
                 if attrs["style"] == style]
        nx.draw_networkx_edges(
            graph, pos, edgelist=[(u, v) for u, v, _ in edges],
            width=[1 + flow_scale * attrs["flow"] for _, _, attrs in edges],
            style=style, edge_color="gray", alpha=0.6, ax=ax,
        )

    colors = {"node_color": "steelblue"}
    if use_voltage:
        colors = dict(
            node_color=data["bus"].y[:, 1].detach().cpu().numpy(),
            cmap="coolwarm", vmin=voltage_limits[0], vmax=voltage_limits[1],
        )
    nodes = nx.draw_networkx_nodes(
        graph, pos, node_size=sizes, edgecolors="black",
        linewidths=0.4, ax=ax, **colors,
    )
    if use_voltage:
        fig.colorbar(nodes, ax=ax, label="|V| (p.u.)", shrink=0.6)
    if labels:
        nx.draw_networkx_labels(graph, pos, font_size=6, ax=ax)

    legend = [
        Line2D([], [], color="gray", linestyle="solid", label="AC line"),
        Line2D([], [], color="gray", linestyle="dashed", label="Transformer"),
    ]
    if mode != "topology" and gen_key in data.edge_types:
        legend.append(Line2D([], [], color="gray", marker="o", linestyle="",
                             label="Larger node: generator bus"))
    ax.legend(handles=legend, loc="upper left")
    ax.set_title(title if title is not None else f"Network — {mode}")
    ax.set_axis_off()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

if __name__ == "__main__":
    import os
    from opf_data import load_dataset

    ds = load_dataset()
    plot_network(ds[int(os.environ.get("OPF_SAMPLE_INDEX", "10000"))])
