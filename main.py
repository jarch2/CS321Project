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

# get graph(s)

graph_info = search(name_or_id = 906)[0]
graph_path = graph_info.download(extract = True)[0] + "/" + graph_info.name + ".mtx"

print(graph_path)
y = mmread(graph_path)

# make curvature graphs

G = nx.DiGraph(y)
print(G)

orc_directed = OllivierRicci(G)
orc_directed.compute_ricci_curvature()

print("no")

for n1, n2 in G.edges():
    print("Directed Graph: The Ollivier-Ricci curvature of edge(%d,%d) id %f" %
          (n1, n2, orc_directed.G[n1][n2]["ricciCurvature"]))

# plot curvature graphs

X = [y]

FP = FlagserPersistence()

diagrams = FP.fit_transform(X)
fig = plot_diagram(diagrams[0])
fig.show()



