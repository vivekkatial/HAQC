import pickle
import json
import networkx as nx
import numpy as np

def load_pickle_data(pickle_file_path):
    """Load data from a pickle file."""
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def numpy_to_python(value):
    """Convert numpy data types to native Python data types for JSON serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()  # Convert arrays to lists
    elif isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
        return int(value)  # Convert numpy integers to Python int
    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
        return float(value)  # Convert numpy floats to Python float
    else:
        return value  # Return the value unchanged if not a numpy type

def process_landscape_data(landscape_data):
    """Process landscape_data to ensure all numpy types are converted to Python types."""
    for key, value in landscape_data.items():
        landscape_data[key] = numpy_to_python(value)

def graph_to_adjacency_matrix(graph):
    """Convert a networkx graph to an adjacency matrix (list of lists)."""
    adj_matrix = nx.to_numpy_matrix(graph)
    return adj_matrix.tolist()

def process_data_for_json(data):
    """Process data to make it JSON serializable, including converting graphs and landscape_data."""
    for item in data:
        # Process landscape_data
        if 'landscape_data' in item:
            process_landscape_data(item['landscape_data'])
        # Process graph
        if 'graph' in item and hasattr(item['graph'], 'G'):
            graph = item['graph'].G
            if isinstance(graph, nx.Graph):
                item['graph'] = {"adjacency_matrix": graph_to_adjacency_matrix(graph)}
    return data

def save_data_to_json(data, json_file_path):
    """Save data to a JSON file."""
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Path to your .pkl file
pickle_file_path = 'ml-model-landscape.pkl'
# Desired path for the output .json file
json_file_path = 'ml-model-landscape.json'

# Load the data from the pickle file
data = load_pickle_data(pickle_file_path)

# Process the data to convert networkx graphs to adjacency matrices and numpy types to Python types
data = process_data_for_json(data)

# Save the processed data to a JSON file
save_data_to_json(data, json_file_path)

print(f"Data, including adjacency matrices of graphs, saved to '{json_file_path}'")
