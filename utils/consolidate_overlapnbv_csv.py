## Uso puntual: junta los CSV por-objeto que dejo la corrida anterior de OverlapNBV
## (uno por objeto, dentro de su propia carpeta) en un solo CSV por split, y borra
## los individuales. Para corridas futuras, run_overlapnbv_one_object.py ya no genera
## un CSV por objeto -- acumula directo en el CSV del split.

import os
import sys
import glob
import pandas as pd


def consolidate_split(split_dir, csv_name):
    pattern = os.path.join(split_dir, "*", csv_name)
    per_object_csvs = sorted(glob.glob(pattern))
    if not per_object_csvs:
        print("{}: no hay CSVs por objeto que consolidar".format(split_dir))
        return

    frames = [pd.read_csv(p) for p in per_object_csvs]
    combined = pd.concat(frames, ignore_index=True)

    combined_path = os.path.join(split_dir, csv_name)
    tmp_path = combined_path + ".tmp"
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, combined_path)

    for p in per_object_csvs:
        os.remove(p)

    print("{}: {} objetos consolidados, {} filas totales -> {}".format(
        split_dir, len(per_object_csvs), len(combined), combined_path))


if __name__ == '__main__':
    csv_name = sys.argv[1] if len(sys.argv) > 1 else "NBV_OverlapNBV.csv"
    split_dirs = sys.argv[2:] if len(sys.argv) > 2 else [
        "/mnt/cc7c68c6-c81c-401d-91fd-04c40177514b/ShapeNetCore-archive/prepared/train",
        "/mnt/cc7c68c6-c81c-401d-91fd-04c40177514b/ShapeNetCore-archive/prepared/val",
        "/mnt/cc7c68c6-c81c-401d-91fd-04c40177514b/ShapeNetCore-archive/prepared/test_similar",
        "/mnt/cc7c68c6-c81c-401d-91fd-04c40177514b/ShapeNetCore-archive/prepared/test_novel",
    ]
    for split_dir in split_dirs:
        consolidate_split(split_dir, csv_name)
