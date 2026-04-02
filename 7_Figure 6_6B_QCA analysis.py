# -*- coding: utf-8 -*-
# Network graph comparison: export T1–T4 as separate PDF files
# pip install matplotlib networkx

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import os

# ===== Global font settings =====
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 20

# ===== Fixed node order =====
NODES = ["T", "A", "B", "C", "D", "E", "F"]

# ===== Adjacency matrices =====
T1 = {
 "T":{"T":1,"A":0.935,"B":0.701,"C":0.899,"D":0.894,"E":0.705,"F":0.88},
 "A":{"T":0.946,"A":1,"B":0.699,"C":0.9,"D":0.913,"E":0.714,"F":0.893},
 "B":{"T":0.56,"A":0.552,"B":1,"C":0.578,"D":0.569,"E":0.573,"F":0.627},
 "C":{"T":0.892,"A":0.883,"B":0.718,"C":1,"D":0.879,"E":0.709,"F":0.869},
 "D":{"T":0.893,"A":0.902,"B":0.712,"C":0.885,"D":1,"E":0.709,"F":0.888},
 "E":{"T":0.672,"A":0.674,"B":0.684,"C":0.682,"D":0.676,"E":1,"F":0.71},
 "F":{"T":0.806,"A":0.808,"B":0.718,"C":0.802,"D":0.813,"E":0.681,"F":1},
}

T2 = {
 "T":{"T":1,"A":0.928,"B":0.718,"C":0.901,"D":0.902,"E":0.713,"F":0.872},
 "A":{"T":0.941,"A":1,"B":0.721,"C":0.897,"D":0.905,"E":0.718,"F":0.886},
 "B":{"T":0.549,"A":0.544,"B":1,"C":0.569,"D":0.559,"E":0.586,"F":0.622},
 "C":{"T":0.889,"A":0.873,"B":0.733,"C":1,"D":0.878,"E":0.718,"F":0.871},
 "D":{"T":0.899,"A":0.89,"B":0.727,"C":0.888,"D":1,"E":0.723,"F":0.875},
 "E":{"T":0.667,"A":0.663,"B":0.716,"C":0.681,"D":0.679,"E":1,"F":0.7},
 "F":{"T":0.786,"A":0.789,"B":0.733,"C":0.797,"D":0.792,"E":0.675,"F":1},
}

T3 = {
 "T":{"T":1,"A":0.935,"B":0.766,"C":0.904,"D":0.915,"E":0.706,"F":0.903},
 "A":{"T":0.946,"A":1,"B":0.766,"C":0.9,"D":0.92,"E":0.711,"F":0.91},
 "B":{"T":0.544,"A":0.537,"B":1,"C":0.561,"D":0.555,"E":0.586,"F":0.612},
 "C":{"T":0.893,"A":0.879,"B":0.781,"C":1,"D":0.887,"E":0.71,"F":0.901},
 "D":{"T":0.909,"A":0.904,"B":0.777,"C":0.893,"D":1,"E":0.717,"F":0.889},
 "E":{"T":0.652,"A":0.649,"B":0.762,"C":0.663,"D":0.666,"E":1,"F":0.694},
 "F":{"T":0.811,"A":0.809,"B":0.776,"C":0.82,"D":0.804,"E":0.676,"F":1},
}

T4 = {
 "T":{"T":1,"A":0.914,"B":0.76,"C":0.905,"D":0.898,"E":0.701,"F":0.886},
 "A":{"T":0.932,"A":1,"B":0.766,"C":0.907,"D":0.913,"E":0.713,"F":0.912},
 "B":{"T":0.565,"A":0.558,"B":1,"C":0.576,"D":0.57,"E":0.582,"F":0.62},
 "C":{"T":0.9,"A":0.885,"B":0.772,"C":1,"D":0.896,"E":0.714,"F":0.888},
 "D":{"T":0.903,"A":0.9,"B":0.772,"C":0.906,"D":1,"E":0.714,"F":0.889},
 "E":{"T":0.666,"A":0.663,"B":0.744,"C":0.681,"D":0.674,"E":1,"F":0.705},
 "F":{"T":0.82,"A":0.827,"B":0.772,"C":0.825,"D":0.818,"E":0.687,"F":1},
}

# ===== Graph construction =====
def build_graph(matrix, nodes, threshold=0.70):
    """Build undirected graph from symmetric matrix."""
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i_idx, i in enumerate(nodes):
        for j_idx in range(i_idx + 1, len(nodes)):
            j = nodes[j_idx]
            w = (matrix[i][j] + matrix[j][i]) / 2.0
            if w >= threshold:
                G.add_edge(i, j, weight=w)
    return G

# ===== Layout computation =====
def compute_layout(base_matrix, nodes, threshold=0.70, seed=42):
    G_base = build_graph(base_matrix, nodes, threshold)
    return nx.spring_layout(G_base, seed=seed)

# ===== Plotting =====
def draw_network(G, pos, pdf_path):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.axis("off")

    nx.draw_networkx_nodes(G, pos, node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=20)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        wmin, wmax = min(weights), max(weights)
        widths = [
            0.5 + 5.5*(w - wmin)/(wmax - wmin) if wmax != wmin else 3
            for w in weights
        ]
        nx.draw_networkx_edges(G, pos, width=widths)

    plt.tight_layout()
    fig.savefig(pdf_path)
    return fig

# ===== Main execution =====
def main():
    base_dir = os.path.dirname(__file__)        
    outdir  = os.path.join(base_dir, 'results')
    os.makedirs(outdir , exist_ok=True)

    threshold = 0.70
    pos = compute_layout(T1, NODES, threshold)

    datasets = [
        ("network_T1.pdf", T1),
        ("network_T2.pdf", T2),
        ("network_T3.pdf", T3),
        ("network_T4.pdf", T4),
    ]

    figures = []
    for filename, matrix in datasets:
        G = build_graph(matrix, NODES, threshold)
        fig = draw_network(G, pos, os.path.join(outdir, filename))
        figures.append(fig)

    for fig in figures:
        plt.figure(fig.number)
        plt.show()

    for fig in figures:
        plt.close(fig)

    print("Saved PDF files to:", outdir)

if __name__ == "__main__":
    main()
