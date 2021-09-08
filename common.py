from functools import partial
from typing import Callable

import numpy as np
import numpy.linalg as npl
from qulacs import QuantumCircuit, QuantumState

from qulacs.state import partial_trace

Randomizer = Callable[[QuantumCircuit, int, int], None]


def simulate(n: int, k: int, l: int, randomize: Randomizer):
    print(f"Simulating {n=}, {k=}, {l=}, {randomize=}")

    state = QuantumState(n + k)
    state.set_zero_state()

    circuit = QuantumCircuit(n + k)
    for i in range(k):
        circuit.add_H_gate(i)
        circuit.add_CNOT_gate(i, i + k)
    randomize(circuit, k, n + k)

    circuit.update_quantum_state(state)

    trace = partial_trace(state, list(range(k, k + l)))
    base = np.eye(1 << (n + k - l)) / np.power(2, n + k - l)
    d = npl.norm(trace.get_matrix() - base, "nuc")
    return d


def construct_lrc(depth: int, circuit: QuantumCircuit, s: int, t: int):
    for d in range(depth):
        i = s + d % 2
        while i + 1 < t:
            circuit.add_random_unitary_gate([i, i + 1])
            i += 2


def get_randomizer_getter(kind: str):  # -> Callable[[Any], Randomizer]:
    if kind == "haar":
        return lambda **_: lambda circuit, p, q: circuit.add_random_unitary_gate(
            range(p, q)
        )
    if kind == "lrc":
        return lambda depth, **_: partial(construct_lrc, depth)
    raise RuntimeError(f"Unknown randomizer kind: {kind}")
