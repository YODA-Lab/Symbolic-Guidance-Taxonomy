import argparse
import os
import json
import random
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

from utils import (
    relabel_and_name_vertices,
    generate_ba_graph,
    generate_ws_graph,
    generate_delaunay_triangulation,
)

GRAPH_MODELS = {
    "ws": generate_ws_graph,
    "ba": generate_ba_graph,
    "dt": generate_delaunay_triangulation,
}

OUTPUT_DIR = "graphs"


def get_graph_path(graph_model, graph_size, num_sample):
    filename = f"graph_{graph_model}_{graph_size}_{num_sample}.json"
    return os.path.join(OUTPUT_DIR, filename)


def get_graph(graph_model, graph_size, num_sample):
    filepath = get_graph_path(graph_model, graph_size, num_sample)
    print(f"Loading graph from {filepath}")
    with open(filepath) as f:
        graph_dict = json.load(f)
    # FIX: remove unsupported edges= kwarg
    return json_graph.node_link_graph(graph_dict["graph"])


def safe_diameter(G: nx.Graph):
    """Return diameter; if disconnected, use the largest connected component."""
    if G.number_of_nodes() == 0:
        return 0
    if nx.is_connected(G):
        return nx.diameter(G)
    # fall back to diameter of the largest CC
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    return nx.diameter(G.subgraph(largest_cc_nodes).copy())


def instance_seed(base_seed: int, model_key: str, size: int, idx: int) -> int:
    """Deterministic per-instance seed (stable across Python runs)."""
    model_ids = {"ws": 1, "ba": 2, "dt": 3}
    return (
        (base_seed * 1_000_003)
        ^ (model_ids.get(model_key, 0) * 97_003)
        ^ (size * 7_001)
        ^ idx
    ) & 0xFFFFFFFF


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_models", type=str, nargs="+", default=["ws", "ba", "dt"])
    parser.add_argument("--samples_per_graph_model", type=int, default=10)
    parser.add_argument("--graph_sizes", type=int, nargs="+", default=[10,20,30,40,50,60,70,80,90,100])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for graph_model in args.graph_models:
        gen_fn = GRAPH_MODELS[graph_model]
        for graph_size in args.graph_sizes:
            for i in range(args.samples_per_graph_model):
                s = instance_seed(args.seed, graph_model, graph_size, i)
                graph = gen_fn(graph_size, seed=s)
                graph = relabel_and_name_vertices(graph)
                filepath = get_graph_path(graph_model, graph_size, i)

                with open(filepath, "w") as f:
                    json.dump(
                        {
                            # FIX: remove unsupported edges= kwarg
                            "graph": json_graph.node_link_data(graph),
                            "num_nodes": graph.number_of_nodes(),
                            "diameter": safe_diameter(graph),
                            "max_degree": max(dict(graph.degree()).values(), default=0),
                            "graph_seed": s,
                            "seed": args.seed,
                        },
                        f,
                        indent=4,
                    )

                plt.figure()
                nx.draw(graph, with_labels=True)
                plt.savefig(filepath[:-5] + ".png")
                plt.close()
                print(f"Graph saved to {filepath}")
