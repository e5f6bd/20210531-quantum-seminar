from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import numpy.linalg as npl
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import DenseMatrix, CPTP
from qulacs.state import partial_trace


def renyi(a: float, m) -> float:
    assert a >= 0
    if a == 1:
        return -sum(x * np.log2(x) for x in npl.eigvalsh(m) if x > 0)
    else:
        return np.log2(np.real(np.trace(npl.matrix_power(m, a)))) / (1 - a)


class EntanglementType(Enum):
    from enum import auto

    TRIPARTITE = auto()
    ENTANGLEMENT_ENTROPY = auto()


def simulate(
    n: int,
    d: int,
    p: float,
    a: int,
    measure_for: Optional[list[int]] = None,
    entanglement_type: EntanglementType = EntanglementType.TRIPARTITE,
    entanglement_entropy_for: Optional[list[int]] = None,
) -> float:
    assert 0 <= p <= 1

    if measure_for is None:
        measure_for = list(range(n))
    assert all(0 <= i < n for i in measure_for)
    if entanglement_type == EntanglementType.TRIPARTITE:
        assert entanglement_entropy_for is None
    if entanglement_type == EntanglementType.ENTANGLEMENT_ENTROPY:
        assert entanglement_entropy_for is not None

    circuit = QuantumCircuit(n)
    for k in range(d):
        for j in range(n // 2):
            i = j * 2 + k % 2
            circuit.add_random_unitary_gate([i, (i + 1) % n])
        for i in measure_for:
            sp, sq = np.sqrt(p), np.sqrt(1 - p)
            circuit.add_gate(
                CPTP(
                    [
                        DenseMatrix(i, matrix)
                        for matrix in [
                            [[sq, 0], [0, sq]],
                            [[sp, 0], [0, 0]],
                            [[0, 0], [0, sp]],
                        ]
                    ]
                )
            )

    state = QuantumState(n)
    state.set_zero_state()
    circuit.update_quantum_state(state)

    if entanglement_type == EntanglementType.TRIPARTITE:
        ms = [n * i // 4 for i in range(5)]
        ranges = [list(range(ms[i], ms[i + 1])) for i in range(4)]
        ret = 0
        for i in range(1, 8):
            trace_range = [
                x for j, r in enumerate(ranges) for x in r if not 1 << j & i << 1
            ]
            coef = bin(i).count("1") % 2 * 2 - 1
            entropy = renyi(a, partial_trace(state, trace_range).get_matrix())
            ret += coef * entropy
            # print(coef, trace_range, entropy)
        return ret
    elif entanglement_type == EntanglementType.ENTANGLEMENT_ENTROPY:
        return renyi(2, partial_trace(state, entanglement_entropy_for).get_matrix())
    else:
        raise RuntimeError(f"Unsupported entanglement type: {entanglement_type}")


def main():
    parser = ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    parser.add_argument("r", type=int)
    parser.add_argument("a", type=int)
    parser.add_argument("ps", type=str, help="space separated floats")
    parser.add_argument("--entanglement-entropy", type=str, help="space separated ints")
    parser.add_argument("--measure-for", type=str, help="space separated ints")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    n, d, r, a = args.n, args.d, args.r, args.a
    ps = list(map(float, args.ps.split()))
    measure_for = None
    if args.measure_for is not None:
        measure_for = list(map(int, args.measure_for.split()))
    entanglement_entropy_args = {}
    if args.entanglement_entropy is not None:
        entanglement_entropy_args = {
            "entanglement_type": EntanglementType.ENTANGLEMENT_ENTROPY,
            "entanglement_entropy_for": list(
                map(int, args.entanglement_entropy.split())
            ),
        }

    output_dir = Path(__file__).parent / "output"
    output = (
        output_dir
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pt_n{n:02d}_d{d:02d}_a{a}_r{r:02d}"
    )

    with open("/dev/null" if args.output else output.with_suffix(".txt"), "w") as f:
        f.write(" ".join(map(repr, sys.argv)))

    with open(args.output or output.with_suffix(".tsv"), "w") as f:
        for p in ps:
            ds = [
                simulate(n, d, p, a, measure_for, **entanglement_entropy_args)
                for _ in range(r)
            ]
            av, std = np.average(ds), np.std(ds)
            print(f"{datetime.now()}\t{p=}\t{av}±{std}")
            f.write("\t".join(map(str, [p, av, std])) + "\n")


if __name__ == "__main__":
    main()
