from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np

from common import get_randomizer_getter, simulate

def main():
    parser = ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("r", type=int)
    parser.add_argument("random_kind", choices=["haar", "lrc"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--depth", type=int)
    args = parser.parse_args()

    kind = args.random_kind
    n, k, r = args.n, args.k, args.r
    get_randomizer = get_randomizer_getter(kind)

    if kind == "haar":
        randomizer_args = {}
    elif kind == "lrc":
        if args.depth is None:
            raise RuntimeError("Specify depth.")
        randomizer_args = {"depth": args.depth}
    else:
        raise RuntimeError(f"Unknown randomizer kind: {kind}")

    randomizer_info = "".join(f"_{k}{v}" for k, v in randomizer_args.items())
    output_path = args.output or Path(__file__).parent / "output" / \
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_hp_n{n:02d}_k{k:02d}_r{r:02d}_{kind}{randomizer_info}.tsv"
    print(f"Outputing to {output_path}")

    with open(output_path, "w") as f:
        for l in range(1, n+k):
            ds = [simulate(n, k, l, get_randomizer(**randomizer_args)) for _ in range(r)]
            print(f"{ds=} {np.average(ds)=} {np.std(ds)=}")
            f.write("\t".join(map(str, [l, np.average(ds), np.std(ds)])) + "\n")

if __name__ == '__main__':
    main()
