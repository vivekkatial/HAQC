import numpy as np


def compute_quantum_burden(qubos):

    evolutions = [qubo["evolution"] for qubo in qubos]
    # print(evolutions)
    zipped = list(zip(*evolutions))
    # Initiate quantum dictionary
    quantum_burden_dicts = {}
    # Looping through number of layers
    for layer in zipped:
        # Quantum burden list
        quantum_burden_list = np.array(
            [max(0.0001, k["probability_success"]) for k in layer]
        )

        # Compute sum
        quantum_burden = sum(1 / quantum_burden_list)
        # Extract of number of layers
        num_layer = layer[0]["p"]
        layer_name = "layer_{0}_quantum_burden".format(num_layer)
        # Build Quantum Burden Dictionary
        quantum_burden_dicts[layer_name] = quantum_burden

    return quantum_burden_dicts


if __name__ == "__main__":

    # Define sample Qubos
    qubo_1 = {
        "evolution": [
            {"p": 1, "probability_success": 0},
            {"p": 2, "probability_success": 0.22},
        ]
    }
    qubo_2 = {
        "evolution": [
            {"p": 1, "probability_success": 0.11},
            {"p": 2, "probability_success": 0.22},
        ]
    }
    qubo_3 = {
        "evolution": [
            {"p": 1, "probability_success": 0.11},
            {"p": 2, "probability_success": 0.22},
        ]
    }

    # Build qubo lists
    qubos = [qubo_1, qubo_2, qubo_3]

    quantum_burden = compute_quantum_burden(qubos)

    assert quantum_burden["layer_1_quantum_burden"] == (1 / 0.11 + 1 / 0.11 + 1 / 0.11)
    assert quantum_burden["layer_2_quantum_burden"] == (1 / 0.22 + 1 / 0.22 + 1 / 0.22)
    print(quantum_burden)
