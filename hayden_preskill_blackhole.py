from datetime import datetime
from pathlib import Path

import numpy as np
from argparse import ArgumentParser
from common import Randomizer, construct_lrc, simulate

def get_randomizer(kind: str) -> Randomizer:
    if kind == "haar":
        return lambda circuit, p, q: circuit.add_random_unitary_gate(range(p, q))
    if kind == "lrc":
        return construct_lrc
    raise RuntimeError(f"Unknown randomizer kind: {kind}")

def main():
    parser = ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("r", type=int)
    parser.add_argument("random_kind", choices=["haar", "lrc"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    n, k, r = args.n, args.k, args.r
    randomizer = get_randomizer(args.random_kind)
    output_path = args.output or Path(__file__).parent / "output" / \
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_hp_n{n:02d}_k{k:02d}_r{r:02d}_{args.random_kind}.tsv"
    print(f"Outputing to {output_path}")

    with open(output_path, "w") as f:
        for l in range(1, n+k):
            ds = [simulate(n, k, l, randomizer) for _ in range(r)]
            print(f"{ds=} {np.average(ds)=} {np.std(ds)=}")
            f.write("\t".join(map(str, [l, np.average(ds), np.std(ds)])) + "\n")

if __name__ == '__main__':
    main()
