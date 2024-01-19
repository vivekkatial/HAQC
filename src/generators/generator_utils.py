import networkx as nx
import numpy as np
import json
from networkx.readwrite import json_graph
from src.generators.random_instances import *
import os.path


def draw_graph_with_edge_weights(G):
    """
    This function draws the given graph with edge weights

    Args:
        G (object): Networkx graph object
    """

    for edge in G.edges:
        assert "cost" in G[edge[0]][edge[1]], "All edges must have a cost"

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, "cost")
    plt = nx.draw(G, pos=pos)
    plt = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, with_labels=True)

    return plt


def np_encoder(object):
    """Numpy encoder"""
    if isinstance(object, np.generic):
        return object.item()


def instance_constructor(
    G, num_vehicles, threshold, n_max, instance_type, p_max, vehicle_capacity=1
):
    """A constructor for the instance json

    Args:
        G (object): NetworkX Graph Object
        num_vehicles (int): Integer for number of vehicles
        threshold (float): Threshold value
        n_max (int): Number of Nelder-Mead restarts before hitting P(Success) > 0.125
        vehicle_capacity (not defined)
        instance_type (str) : Source type of instance

    Returns:
        dict: JSON Dictionary representation of the graph
    """

    # Initialise instance
    instance = {
        "graph": {"nodes": [], "edges": []},
        "vehicleCapacity": vehicle_capacity,
        "numVehicles": None,
        "n_max": None,
        "threshold": None,
        "p_max": None,
    }

    # Build JSON Graph
    graph_dict = json_graph.node_link_data(G, {"link": "edges"})

    # Attache graph methods
    instance["graph"]["nodes"] = graph_dict["nodes"]
    instance["graph"]["edges"] = graph_dict["edges"]

    # Other properties
    instance["numVehicles"] = num_vehicles
    instance["n_max"] = n_max
    instance["threshold"] = threshold
    instance["instance_type"] = instance_type
    instance["p_max"] = p_max

    return instance


def compile_and_write(
    num_nodes,
    num_vehicles,
    instance_type,
    p_max=10,
    n_max=3,
    threshold=1,
    data_folder="data",
):

    G = generate_random_instance(num_nodes, num_vehicles, instance_type=instance_type)
    instance_json = instance_constructor(
        G,
        num_vehicles=num_vehicles,
        instance_type=instance_type,
        threshold=threshold,
        n_max=n_max,
        p_max=p_max,
    )

    # Instance name
    instance_name = "instanceType_{0}_numNodes_{1}_numVehicles_{2}_{4}.json".format(
        instance_type, num_nodes, num_vehicles, p_max, uuid.uuid4().hex
    )

    # Create file path
    instance_file_path = os.path.join(data_folder, instance_name)
    with open(instance_file_path, "w") as file:
        json.dump(instance_json, file, default=np_encoder, indent=4)
