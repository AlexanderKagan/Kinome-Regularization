import networkx as nx
import matplotlib.pyplot as plt


def make_kinase_2_correlated_neighbor_dict(cor_matrix, cor_threshold=0.95):
    kinase_2_correlated = {}
    names = cor_matrix.columns
    for kinase in names:

        cur_row = cor_matrix[kinase][names != kinase]
        cur_row_names = names[names != kinase]

        highly_correlated_neighbors = list(cur_row_names[cur_row > cor_threshold])
        if highly_correlated_neighbors:
            kinase_2_correlated[kinase] = highly_correlated_neighbors
    return kinase_2_correlated


def draw_graph(adjacency_dict, colors=None):
    undirected_edges_list = make_undirected_edge_list(adjacency_dict)

    graph = nx.Graph()
    graph.add_edges_from(undirected_edges_list)

    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw(graph,
            ax=ax,
            edgelist=undirected_edges_list,
            nodelist=adjacency_dict.keys(),
            node_color=colors,
            node_size=100,
            with_labels=True)


def make_undirected_edge_list(adjacency_dict):
    edge_set = set()
    for v, v_adj_list in adjacency_dict.items():
        for u in v_adj_list:
            if (u, v) not in edge_set:
                edge_set.add((u, v))

    return list(edge_set)