from typing import Callable
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import partial_trace
import numpy as np
import numpy.linalg as npl

Randomizer = Callable[[QuantumCircuit, int, int], None]

def simulate(n: int, k: int, l: int, randomize: Randomizer):
    print(f"Simulating {n=}, {k=}, {l=}, {randomize=}")

    state = QuantumState(n + k)
    state.set_zero_state()

    circuit = QuantumCircuit(n + k)
    for i in range(k):
        circuit.add_H_gate(i)
        circuit.add_CNOT_gate(i, i+k)
    randomize(circuit, k, n+k)

    circuit.update_quantum_state(state)

    trace = partial_trace(state, list(range(k, k+l)))
    base = np.eye(1 << (n + k - l)) / np.power(2, n + k - l)
    d = npl.norm(trace.get_matrix() - base, 'nuc')
    return d

def construct_lrc(circuit: QuantumCircuit, p: int, q: int):
    pass
