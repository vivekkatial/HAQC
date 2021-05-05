import networkx as nx

class TestGraphToJSON:
    def test_instance_constructor_builds_valid_instance(self):
        G = nx.complete_graph(2)
        assert G is not None