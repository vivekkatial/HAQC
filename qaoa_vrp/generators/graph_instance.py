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
