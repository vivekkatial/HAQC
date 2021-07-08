import logging
from qaoa_vrp.generators.generator_utils import compile_and_write

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Building Instances")

    # Global Variables
    num_nodes = range(5,6)
    num_vehicles = range(1, 2)
    max_layers = range(2, 10)
    # instance_types = ["watts_strogatz", "complete", "newman_watts_strogatz"]
    n_rand = 100
    instance_types = ["complete"]

    for n in num_nodes:
        for v in num_vehicles:
            if 2 * v < n:
                for instance_type in instance_types:
                    for i in range(n_rand):
                        if v < (n - 1):
                            logging.info(
                                "Building and writing instance_type:\t{0} with num nodes:\t{1} and n_vehicles:\t{2}".format(
                                    instance_type, n, v
                                )
                            )
                            compile_and_write(n, v, instance_type)
