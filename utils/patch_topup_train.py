## Parche puntual: completa "train" a la cantidad pedida sin reshufflear ni
## tocar val/test_similar/test_novel (que ya estan completos). Toma
## reemplazos del mismo pool "seen", en el mismo orden deterministico
## (misma seed), saltando cualquier modelo ya usado en train/val/test_similar.

import csv
import os
from utils.reconstructorParams import Params
from utils.prepare_shapenet_split import SEEN_CATEGORIES, build_pool, prepare_object
import random

CONFIG = "dataexample/paramsShapeNetSplit.yaml"


def main():
    params = Params(CONFIG)
    shapenet_root = params.getParameter("shapenet_root")
    output_root = params.getParameter("output_root")
    seed = params.getParameter("seed")
    n_train = params.getParameter("split_sizes")["train"]

    manifest_path = os.path.join(output_root, "split_manifest.csv")
    with open(manifest_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    used_seen = set()
    train_count = 0
    for r in rows:
        if r["split"] in ("train", "val", "test_similar"):
            used_seen.add((r["synset_id"], r["model_id"]))
        if r["split"] == "train":
            train_count += 1

    missing = n_train - train_count
    print("train tiene {} de {} pedidos, faltan {}".format(train_count, n_train, missing))
    if missing <= 0:
        print("nada que hacer")
        return

    rng = random.Random(seed)
    seen_pool = build_pool(shapenet_root, SEEN_CATEGORIES)
    rng.shuffle(seen_pool)

    split_dir = os.path.join(output_root, "train")
    new_rows = []
    added = 0
    for synset_id, model_id, category_name in seen_pool:
        if added >= missing:
            break
        if (synset_id, model_id) in used_seen:
            continue
        object_name, ok = prepare_object(shapenet_root, synset_id, model_id, split_dir)
        if ok:
            added += 1
            used_seen.add((synset_id, model_id))
            new_rows.append({
                "split": "train",
                "categoria": category_name,
                "synset_id": synset_id,
                "model_id": model_id,
                "carpeta": os.path.join("train", object_name),
            })
            print("agregado: {}".format(object_name))

    rows.extend(new_rows)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "categoria", "synset_id", "model_id", "carpeta"])
        writer.writeheader()
        writer.writerows(rows)

    print("agregados {} objetos nuevos a train. total train ahora: {}".format(added, train_count + added))
    print("manifiesto actualizado: {}".format(manifest_path))


if __name__ == "__main__":
    main()
