import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

# Define an example adjacency matrix (replace this with your own)
adjacency_matrix = scio.loadmat("data/A_metropolis.mat")["A_metropolis"]

# Remove self-loops by setting diagonal elements to 0
np.fill_diagonal(adjacency_matrix, 0)

# Create a graph from the adjacency matrix
G = nx.Graph(adjacency_matrix)

# Define the positions of nodes (you can use other layout algorithms as well)
pos = nx.spring_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=300, font_size=10, font_color='black')

plt.savefig("figures/topology.png", bbox_inches='tight')
plt.show()

# Define an example adjacency matrix (replace this with your own)
adjacency_matrix = scio.loadmat("data/A_uniform.mat")["A_uniform"]

# Remove self-loops by setting diagonal elements to 0
np.fill_diagonal(adjacency_matrix, 0)

# Create a graph from the adjacency matrix
G = nx.Graph(adjacency_matrix)

# Define the positions of nodes (you can use other layout algorithms as well)
pos = nx.spring_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=300, font_size=10, font_color='black')

plt.savefig("figures/topology2.png", bbox_inches='tight')
plt.show()