from qulacs import QuantumState, QuantumCircuit
from qulacs.state import partial_trace
import numpy as np
import numpy.linalg as npl

def simulate(n: int, k: int, l: int):
    state = QuantumState(n + k)
    state.set_zero_state()

    circuit = QuantumCircuit(n + k)
    for i in range(k):
        circuit.add_H_gate(i)
        circuit.add_CNOT_gate(i, i+k)
    circuit.add_random_unitary_gate(range(k, n+k))

    circuit.update_quantum_state(state)

    trace = partial_trace(state, list(range(k, k+l)))
    base = np.eye(1 << (n + k - l)) / np.power(2, n + k - l)
    d = npl.norm(trace.get_matrix() - base, 1)
    return d

def main():
    n, k = 9, 1
    r = 10
    for l in range(1, n+k):
        ds = [simulate(n, k, l) for _ in range(r)]
        # print(f"{ds=} {np.average(ds)=} {np.std(ds)=}")
        print("\t".join(map(str, [l, np.average(ds), np.std(ds)])))

if __name__ == '__main__':
    main()
