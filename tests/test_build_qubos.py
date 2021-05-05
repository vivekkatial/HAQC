import qaoa_vrp.build_circuit 
import networkx as nx
from qiskit.optimization import QuadraticProgram

class TestBuildCircuits:
    def test_if_build_qubos_returns_list_of_QuadPrograms(self):
        """Tests each item from list returned by the build_qubos function is a QUBO
        """
        G = nx.complete_graph(3)
        for edge in G.edges():
            G[edge[0]][edge[1]]['cost'] = 1.0
        clusters = {0: G, 1:G}
        depot_info = {"id": 0}
        bool = True
        for item in qaoa_vrp.build_circuit.build_qubos(clusters, depot_info):
            if isinstance(item, QuadraticProgram) == False:
                bool = False
                continue
        assert bool