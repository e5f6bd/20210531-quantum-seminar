from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.linalg as npl
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import DenseMatrix, CPTP
from qulacs.state import partial_trace

def renyi(a: float, m) -> float:
    assert(a >= 0)
    if a == 1:
        return sum(x * np.log2(x) for x in npl.eig(m))
    else:
        print(npl.matrix_power(m, a))
        return np.log2(np.trace(npl.matrix_power(m, a))) / (1-a)

def simulate(n: int, d: int, p: float) -> float:
    assert(0 <= p <= 1)

    circuit = QuantumCircuit(n)
    for k in range(d):
        for j in range(n//2):
            i = j * 2 + k % 2
            circuit.add_random_unitary_gate([j, (j+1) % n])
        for i in range(n):
            sp, sq = np.sqrt(p), np.sqrt(1-p)
            circuit.add_gate(CPTP([
                DenseMatrix(i, matrix)
                for matrix in [
                    [[sq, 0], [0, sq]],
                    [[sp, 0], [0, 0]],
                    [[0, 0], [0, sp]],
                ]
            ]))

    state = QuantumState(n)
    state.set_zero_state()
    circuit.update_quantum_state(state)

    ms = [n * i // 4 for i in range(5)]
    ranges = [list(range(ms[i], ms[i+1])) for i in range(4)]
    ret = 0
    for i in range(1, 8):
        trace_range = [x for j, r in enumerate(ranges) for x in r if not 1 << j & i << 1]
        coef = 2 * bin(i).count('1') - 1
        ret += coef * renyi(2, partial_trace(state, trace_range).get_matrix())

    return ret

def main():
    parser = ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    parser.add_argument("r", type=int)
    parser.add_argument("ps", type=str, help="space separated floats")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    n, r, d = args.n, args.d, args.r
    ps = list(map(float, args.ps.split()))

    output_dir = Path(__file__).parent / "output" 
    output = args.output or output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pt_n{n:02d}_d{d:02d}_r{r:02d}.tsv"

    with open(output, "w") as f:
        for p in ps:
            ds = [simulate(n, d, p) for _ in range(d)]
            av, std = np.average(ds), np.std(ds)
            f.write("\t".join(map(str, [p, av, std])) + "\n")

if __name__ == '__main__':
    main()
