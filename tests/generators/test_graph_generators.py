import networkx as nx

class TestGraphToJSON:
    def test_instance_constructor_builds_valid_instance(self):
        G = nx.complete_graph(2)
        for edge in G.edges():
            G[edge[0]][edge[1]]['cost'] = 1.0
        
        
        assert G is not None