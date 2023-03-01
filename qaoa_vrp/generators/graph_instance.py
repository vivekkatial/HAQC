import random
import networkx as nx
import numpy as np


class GraphInstance:
    def __init__(self, G, graph_type):
        self.G = G
        self.graph_type = graph_type
        self.weight_matrix = None
        self.brute_force_sol = None
        self.removed_edges = []
        self.added_edges = []

    def __repr__(self):
        return f"This is a {self.graph_type} {self.G} graph instance"

    def allocate_random_weights(self):
        # Allocate random costs to the edges for now
        for (u, v) in self.G.edges():
            if self.graph_type == "4-Regular Graph Fixed Weights":
                self.G.edges[u, v]["weight"] = random.randint(-1, 1)
            else:
                self.G.edges[u, v]["weight"] = random.randint(0, 10)

    def compute_weight_matrix(self):
        G = self.G
        n = len(G.nodes())
        w = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = G.get_edge_data(i, j, default=0)
                if temp != 0:
                    w[i, j] = temp["weight"]
        self.weight_matrix = w

    def show_weight_matrix(self):
        print(self.weight_matrix)

    def nearly_complete(self):
        """An algorithm to make graph "nearly bipartite". Generate a p ~ unif (0,1).

        Based on p <= 0.33, 0.33 < p <= 0.66 and p > 0.66 decide whether or not to add, remove or pause the graph construction

        Raises:
            TypeError: Only works for `graph_type` is "Nearly Complete BiPartite"
        """
        if self.graph_type != "Nearly Complete BiPartite":
            raise TypeError("Inapproriate graph type")
        else:
            keep = False
            # Identify bipartite structure
            try:
                bottom_nodes, top_nodes = nx.algorithms.bipartite.sets(self.G)
            except:
                # Create a nearly compelte bi partite graph
                N = len(self.G.nodes())
                n_part_1 = random.randint(1, N - 1)
                n_part_2 = N - n_part_1
                self.G = nx.complete_bipartite_graph(n_part_1, n_part_2)
                self.nearly_complete()

            # Sample a top and bottom node
            bottom_nodes = list(bottom_nodes)
            top_nodes = list(top_nodes)
            while keep == False:
                # generate p
                prob = random.random()
                # If p <= 0.33 - we remove an edge between the partitions
                if prob <= 1 / 3:
                    u = random.sample(bottom_nodes, 1)[0]
                    v = random.sample(top_nodes, 1)[0]
                    removed_edge = (u, v)
                    self.removed_edges.append(removed_edge)

                    # Check first if edge has been removed or not
                    if removed_edge not in self.removed_edges:
                        print(f"Removing edge ({u},{v})")
                        # Remove an edge
                        self.G.remove_edge(u, v)
                    else:
                        print("Regenerating p")

                # If 0.33 < p <= 0.66 we add an edge between a single partition
                elif prob > 1 / 3 and prob <= 2 / 3:
                    partition = random.randint(0, 1)
                    # Handle for cases when partitions might be only size 1
                    if len(bottom_nodes) < 2:
                        partition = 1
                    elif len(top_nodes) < 2:
                        partition = 0

                    if partition == 0:
                        # Add in bottom partition
                        conn_nodes = random.sample(bottom_nodes, 2)
                        self.G.add_edge(conn_nodes[0], conn_nodes[1])
                    else:
                        # Add in top partition
                        conn_nodes = random.sample(top_nodes, 2)
                        self.G.add_edge(conn_nodes[0], conn_nodes[1])
                    print(f"Adding edge ({conn_nodes[0]},{conn_nodes[1]})")
                else:
                    keep = True

    def build_qubo():
        pass

    def solve_qaoa():
        pass

    def solve_vqe():
        pass


def create_graphs_from_all_sources(instance_size=11, sources="ALL"):
    N = instance_size
    G_instances = []
    # Generating a graph of erdos renyi graph
    G_unif = GraphInstance(nx.erdos_renyi_graph(N, p=0.5), "Uniform Random")
    G_instances.append(G_unif)

    # Power-Law Tree
    G_pl_tree = GraphInstance(
        nx.random_powerlaw_tree(N, gamma=3, seed=None, tries=1000), "Power Law Tree"
    )
    G_instances.append(G_pl_tree)

    # Wattz-Strogatz Graph
    G_wattz = GraphInstance(
        nx.connected_watts_strogatz_graph(N, k=4, p=0.5),
        "Watts-Strogatz small world",
    )
    G_instances.append(G_wattz)

    # Geometric Graphs
    connected = False
    geom_guess = 0
    while connected is False:
        geom_guess += 1
        # Use a radius that is connected 95% of the time
        random_radius = random.uniform(0.24, np.sqrt(2))
        g_geom = nx.random_geometric_graph(N, radius=random_radius)
        connected = nx.algorithms.components.is_connected(g_geom)
        print(
            f"Guess {geom_guess} for producing a connected Geometric Graph with r={random_radius} - connected: {connected}"
        )

    G_geom = GraphInstance(g_geom, "Geometric")
    G_instances.append(G_geom)

    # Create a nearly compelte bi partite graph
    # Randomly generate the size of one partiton
    n_part_1 = random.randint(1, N - 1)
    n_part_2 = N - n_part_1
    G_nc_bipart = GraphInstance(
        nx.complete_bipartite_graph(n_part_1, n_part_2), "Nearly Complete BiPartite"
    )
    G_nc_bipart.nearly_complete()
    G_instances.append(G_nc_bipart)

    # Create a 3-regular graph (based on https://arxiv.org/pdf/2106.10055.pdf)
    if instance_size % 2 == 0:
        G_three_regular = GraphInstance(
            nx.random_regular_graph(d=3, n=N), graph_type="3-Regular Graph"
        )
        G_instances.append(G_three_regular)

    # Create a 4-regular graph (based on https://arxiv.org/pdf/1908.08862.pdf)
    G_four_regular = GraphInstance(
        nx.random_regular_graph(d=4, n=N), graph_type="4-Regular Graph"
    )
    G_instances.append(G_four_regular)

    # Create a 4-regular graph with costs (-1,0,1) (based on https://arxiv.org/pdf/1908.08862.pdf)
    G_four_regular_fixed_weights = GraphInstance(
        nx.random_regular_graph(d=4, n=N), graph_type="4-Regular Graph Fixed Weights"
    )
    G_instances.append(G_four_regular_fixed_weights)

    if sources != "ALL":
        return [G_instances[0], G_instances[1]]
    else:
        return G_instances
