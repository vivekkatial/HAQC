# Next Steps

## Implementation Aspects

- Re-code **all** of the code base to include
  - Unit tests (PyTest)
  - CI/CD
- Setup an OOP paradigm for different instance types
- Any optimsiation needs mathematical proofs + UNIT test

## Classical

- Investigate TSP (alone - use ISA work)
- More advanced LP formulation
  - Adding in more advanced constraints
    - Capacities
    - Demands
- Generate more types of instance graphs
  - Looking at the cost matrix (are triangle inequalities being satisfied)
  - Look at euclidean TSP
  - Produce more structured examples
    - Forced cluster structure (in randomly generated instances)
    - Random costs will have lack cluster structure
- Feature Engineering
  - Add in more graph features
  - Add in TSP features (most things affecting TSP, outlier structure)
  - Remove features that are constant
- Formulation improvements
  - Identify formulation improvements which may reduce number of qubits from the problem definitions. For example:
    - Remove any degree 2 nodes (since solutions are trivial)
      - Implementation (remove via code)
      - Report (remove and demonstrate via proof)
    - Optimise the graph (instance itself)
    - Repeat above for other optimisations to forumulations
- Classical Clustering
  - Not 1 vehicle per cluster?
  - Try not to remove the depot
  - Better techniques
  - Add check to ensure clusters are feasible (for each generated instance)
  - Produce N solutions by generating different clusterings and select "optimal solution" as the best one
    - Clustering itself may be an NP hard problem in itself (Graph Cutting = MAXCUT? -- Proof)
    - Satisfy capacity constraints in clustering itself

## Quantum

- Change encoding to be the mapping for n log(n) qubits
- Implement using MPS simulator
- Use different Quantum Algorithms:
  - Recursive QAOA
    - Look at variants -- dont consider couplings (highly correlated qubits)
    - Look at each individual qubits and make them fixed based on probabilities of the states
    - Light cone -- identify which qubits can _actually_ influence this. CHARLES KNOWS THIS STUFF
  - VQE
    - Reducing number of qubits by having a robust encoding of qubits? n log n
    - Use n^2 qubits and ensure constraints aren't violated
- Circuit Optimisation - reducing circuit depth
  - Build constraints into the circuit construction
- Incorporate error mitigation
- Mapping qiskit qubits to IBM machines
- Solving on IBM-Q Systems
  - Massively reduce number of qubits via the encoding
  - Optimisations of the formulation
- Re-run Instance Space Analysis on heaps of new instances
