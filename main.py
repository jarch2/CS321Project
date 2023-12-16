import numpy as np
from numpy.random import default_rng
rng = default_rng(42)  # Create a random number generator

from gtda.homology import FlagserPersistence

from gtda.plotting import plot_diagram

from ssgetpy import search
from scipy.io import mmread

import networkx as nx

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import matplotlib.pyplot as plt

# utils

def get_orc_graph(edges, edgefiltration=False, print_flag=False):
    G = nx.DiGraph(edges)
    orc_directed = OllivierRicci(G)
    
    orc_directed.compute_ricci_curvature()

    if print_flag:
        for n1, n2 in G.edges():
            if n1 != n2:
                print("%d-(%f)>%d" % (n1, orc_directed.G[n1][n2]["ricciCurvature"], n2))

    orc_graph = nx.to_numpy_array(orc_directed.G, weight="ricciCurvature", nonedge=np.inf)

    if edgefiltration:
        np.fill_diagonal(orc_graph, np.inf)

        for i in range(len(orc_graph)):
            min = np.inf
            for j in range(len(orc_graph)):
                if orc_graph[i][j] < min:
                    min = orc_graph[i][j]
                if orc_graph[j][i] < min:
                    min = orc_graph[j][i]
            orc_graph[i][i] = min
    else:
        np.fill_diagonal(orc_graph, np.min(orc_graph))

    return orc_graph

# premade special graphs
source_tree = [[1,0]]

############################################
# execution down here
############################################

# get graph(s)

graph_info = search(name_or_id = 906)[0]
graph_path = graph_info.download(extract = True)[0] + "/" + graph_info.name + ".mtx"

graph_path = "./through_flow.mtx"
y = mmread(graph_path)

orc_graph = get_orc_graph(y, print_flag=True)

X = [orc_graph]

FP = FlagserPersistence()

diagrams = FP.fit_transform(X)
fig = plot_diagram(diagrams[0])
fig.show()



