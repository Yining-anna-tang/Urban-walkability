# -*- coding: utf-8 -*-
# 网络图结构比较：分别导出 T1–T4 单页 PDF，并用 plt.show() 依次阻塞展示（无合并多页 PDF）
# 依赖：pip install matplotlib networkx

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

# 全局字体与字号
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 20

# 节点顺序固定
NODES = ["T","A","B","C","D","E","F"]

# 数据（按你新给的表）
T1 = {
 "T":{"T":1,    "A":0.935,"B":0.701,"C":0.899,"D":0.894,"E":0.705,"F":0.88},
 "A":{"T":0.946,"A":1,    "B":0.699,"C":0.9,  "D":0.913,"E":0.714,"F":0.893},
 "B":{"T":0.56, "A":0.552,"B":1,    "C":0.578,"D":0.569,"E":0.573,"F":0.627},
 "C":{"T":0.892,"A":0.883,"B":0.718,"C":1,    "D":0.879,"E":0.709,"F":0.869},
 "D":{"T":0.893,"A":0.902,"B":0.712,"C":0.885,"D":1,    "E":0.709,"F":0.888},
 "E":{"T":0.672,"A":0.674,"B":0.684,"C":0.682,"D":0.676,"E":1,    "F":0.71 },
 "F":{"T":0.806,"A":0.808,"B":0.718,"C":0.802,"D":0.813,"E":0.681,"F":1   },
}

T2 = {
 "T":{"T":1,    "A":0.928,"B":0.718,"C":0.901,"D":0.902,"E":0.713,"F":0.872},
 "A":{"T":0.941,"A":1,    "B":0.721,"C":0.897,"D":0.905,"E":0.718,"F":0.886},
 "B":{"T":0.549,"A":0.544,"B":1,    "C":0.569,"D":0.559,"E":0.586,"F":0.622},
 "C":{"T":0.889,"A":0.873,"B":0.733,"C":1,    "D":0.878,"E":0.718,"F":0.871},
 "D":{"T":0.899,"A":0.89, "B":0.727,"C":0.888,"D":1,    "E":0.723,"F":0.875},
 "E":{"T":0.667,"A":0.663,"B":0.716,"C":0.681,"D":0.679,"E":1,    "F":0.7  },
 "F":{"T":0.786,"A":0.789,"B":0.733,"C":0.797,"D":0.792,"E":0.675,"F":1   },
}

T3 = {
 "T":{"T":1,    "A":0.935,"B":0.766,"C":0.904,"D":0.915,"E":0.706,"F":0.903},
 "A":{"T":0.946,"A":1,    "B":0.766,"C":0.9,  "D":0.92, "E":0.711,"F":0.91 },
 "B":{"T":0.544,"A":0.537,"B":1,    "C":0.561,"D":0.555,"E":0.586,"F":0.612},
 "C":{"T":0.893,"A":0.879,"B":0.781,"C":1,    "D":0.887,"E":0.71, "F":0.901},
 "D":{"T":0.909,"A":0.904,"B":0.777,"C":0.893,"D":1,    "E":0.717,"F":0.889},
 "E":{"T":0.652,"A":0.649,"B":0.762,"C":0.663,"D":0.666,"E":1,    "F":0.694},
 "F":{"T":0.811,"A":0.809,"B":0.776,"C":0.82, "D":0.804,"E":0.676,"F":1   },
}

T4 = {
 "T":{"T":1,    "A":0.914,"B":0.76, "C":0.905,"D":0.898,"E":0.701,"F":0.886},
 "A":{"T":0.932,"A":1,    "B":0.766,"C":0.907,"D":0.913,"E":0.713,"F":0.912},
 "B":{"T":0.565,"A":0.558,"B":1,    "C":0.576,"D":0.57, "E":0.582,"F":0.62 },
 "C":{"T":0.9,  "A":0.885,"B":0.772,"C":1,    "D":0.896,"E":0.714,"F":0.888},
 "D":{"T":0.903,"A":0.9,  "B":0.772,"C":0.906,"D":1,    "E":0.714,"F":0.889},
 "E":{"T":0.666,"A":0.663,"B":0.744,"C":0.681,"D":0.674,"E":1,    "F":0.705},
 "F":{"T":0.82, "A":0.827,"B":0.772,"C":0.825,"D":0.818,"E":0.687,"F":1   },
}

def build_graph(M, nodes, threshold=0.70):
    """从矩阵构建图。对称化后仅保留 >= 阈值 的边。"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i_idx, i in enumerate(nodes):
        for j_idx in range(i_idx + 1, len(nodes)):
            j = nodes[j_idx]
            w = (M[i][j] + M[j][i]) / 2.0
            if w >= threshold:
                G.add_edge(i, j, weight=w)
    return G

def normalize_positions(pos, pad=0.08):
    """把 pos 线性映射到 [pad, 1-pad] 区域，避免节点贴边被裁剪。"""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = max(maxx - minx, 1e-9)
    spany = max(maxy - miny, 1e-9)
    scaled = {}
    for k, (x, y) in pos.items():
        nx_ = pad + (x - minx) / spanx * (1 - 2*pad)
        ny_ = pad + (y - miny) / spany * (1 - 2*pad)
        scaled[k] = (nx_, ny_)
    return scaled

def compute_layout(base_matrix, nodes, threshold=0.70, seed=42, k=0.6, pad=0.10):
    """用基准矩阵生成布局，并映射到安全区间。"""
    G_base = build_graph(base_matrix, nodes, threshold=threshold)
    raw_pos = nx.spring_layout(G_base, seed=seed, k=k)
    return normalize_positions(raw_pos, pad=pad)

def make_figure(G, pos, figsize=(8, 8), node_size=1500, label_size=20):
    """返回一个绘制完毕的 Figure（无主标题、无角标，确保不裁边）。"""
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.axis("off")

    nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=label_size, ax=ax)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        wmin, wmax = min(weights), max(weights)
        widths = [3 for _ in weights] if wmax == wmin else [
            0.5 + 5.5*(w - wmin)/(wmax - wmin) for w in weights
        ]
        nx.draw_networkx_edges(G, pos, width=widths, ax=ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig

def main():
    threshold = 0.70
    seed = 42
    k = 0.6
    pad = 0.10
    figsize = (8, 8)
    node_size = 1500
    label_size = 20

    pos = compute_layout(T1, NODES, threshold=threshold, seed=seed, k=k, pad=pad)

    tasks = [
        ("network_Y2_T1.pdf", T1),
        ("network_Y2_T2.pdf", T2),
        ("network_Y2_T3.pdf", T3),
        ("network_Y2_T4.pdf", T4),
    ]

    figures = []
    for pdf_path, M in tasks:
        G = build_graph(M, NODES, threshold=threshold)
        fig = make_figure(G, pos, figsize=figsize, node_size=node_size, label_size=label_size)
        fig.savefig(pdf_path)
        figures.append(fig)

    # 用 plt.show() 阻塞方式依次展示（关闭当前窗口后才会显示下一张）
    for fig in figures:
        plt.figure(fig.number)
        plt.show()   # 阻塞，直到你关闭当前窗口

    # 关闭
    for fig in figures:
        plt.close(fig)

    print("Saved PDFs:", [p for p, _ in tasks])

if __name__ == "__main__":
    main()
