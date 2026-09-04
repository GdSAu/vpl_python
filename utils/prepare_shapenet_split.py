## Prepara splits de ShapeNetCore para usar con Reconstructor/NBVDE, siguiendo el
## protocolo del paper PC-NBV: 4000 train + 400 val + 400 test_similar muestreados
## de 8 categorias "vistas", y 400 test_novel muestreados de 8 categorias "no vistas".
##
## No copia nada: arma <output_root>/<split>/<synsetId>_<modelId>/{meshes,images}
## con symlinks hacia los archivos reales de ShapeNetCore.v2, para que
## Simulator/sceneLoader.py los lea con la misma convencion <objeto>/meshes/model.obj
## que ya usan todos los demas datasets del proyecto.

import os
import sys
import csv
import random
from utils.reconstructorParams import Params

SEEN_CATEGORIES = {
    "airplane": "02691156",
    "cabinet": "02933112",
    "car": "02958343",
    "chair": "03001627",
    "lamp": "03636649",
    "sofa": "04256520",
    "table": "04379243",
    "vessel": "04530566",
}

NOVEL_CATEGORIES = {
    "bus": "02924116",
    "bed": "02818832",
    "bookshelf": "02871439",
    "bench": "02828884",
    "guitar": "03467517",
    "motorbike": "03790512",
    "skateboard": "04225987",
    "pistol": "03948459",
}


def list_available_models(shapenet_root, synset_id):
    synset_dir = os.path.join(shapenet_root, synset_id)
    if not os.path.isdir(synset_dir):
        return []
    return sorted(m for m in os.listdir(synset_dir) if os.path.isdir(os.path.join(synset_dir, m)))


def build_pool(shapenet_root, categories):
    pool = []
    for name, synset_id in categories.items():
        models = list_available_models(shapenet_root, synset_id)
        if not models:
            print("ADVERTENCIA: no hay modelos para {} ({}) en {} -- puede que falte descargar".format(
                name, synset_id, shapenet_root))
        else:
            print("{} ({}): {} modelos disponibles".format(name, synset_id, len(models)))
        pool.extend((synset_id, model_id, name) for model_id in models)
    return pool


def _real_image_extension(path):
    # Some ShapeNet textures are mislabeled (a PNG saved as texture.jpg).
    # Open3D's OBJ+MTL reader aborts the whole process (not a catchable
    # Python exception) when it hits one, so mismatches are detected here
    # and corrected before Open3D ever sees them.
    try:
        with open(path, "rb") as f:
            header = f.read(8)
    except OSError:
        return None
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    return None


def prepare_object(shapenet_root, synset_id, model_id, split_dir):
    object_name = "{}_{}".format(synset_id, model_id)
    object_dir = os.path.join(split_dir, object_name)
    meshes_dir = os.path.join(object_dir, "meshes")
    if os.path.exists(meshes_dir):
        return object_name, True

    src_model_dir = os.path.join(shapenet_root, synset_id, model_id)
    src_obj = os.path.join(src_model_dir, "models", "model_normalized.obj")
    src_mtl = os.path.join(src_model_dir, "models", "model_normalized.mtl")
    src_images_dir = os.path.join(src_model_dir, "images")

    if not os.path.isfile(src_obj):
        print("ADVERTENCIA: falta el .obj de {} ({}), se salta".format(object_name, src_obj))
        return object_name, False

    os.makedirs(meshes_dir, exist_ok=True)
    os.symlink(os.path.abspath(src_obj), os.path.join(meshes_dir, "model.obj"))

    if not os.path.isfile(src_mtl):
        return object_name, True

    if not os.path.isdir(src_images_dir):
        # No texture files at all (flat-color materials only) -- Open3D's
        # OBJ+MTL parser needs the .mtl present for these, and there is no
        # texture file that could be mislabeled, so it's symlinked as-is.
        os.symlink(os.path.abspath(src_mtl), os.path.join(meshes_dir, "model_normalized.mtl"))
        return object_name, True

    # Has textures: symlink each image individually (instead of the whole
    # folder) so mislabeled files can be exposed under their real extension,
    # and write a small corrected copy of the .mtl referencing those names.
    renames = {}
    for fname in os.listdir(src_images_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            continue
        real_ext = _real_image_extension(os.path.join(src_images_dir, fname))
        if real_ext and real_ext != ext and not (ext == ".jpeg" and real_ext == ".jpg"):
            renames[fname] = os.path.splitext(fname)[0] + real_ext

    images_dst = os.path.join(object_dir, "images")
    os.makedirs(images_dst, exist_ok=True)
    for fname in os.listdir(src_images_dir):
        target_name = renames.get(fname, fname)
        dst_path = os.path.join(images_dst, target_name)
        if not os.path.exists(dst_path):
            os.symlink(os.path.abspath(os.path.join(src_images_dir, fname)), dst_path)

    with open(src_mtl, "r", errors="ignore") as f:
        mtl_text = f.read()
    for old_name, new_name in renames.items():
        mtl_text = mtl_text.replace(old_name, new_name)
    with open(os.path.join(meshes_dir, "model_normalized.mtl"), "w") as f:
        f.write(mtl_text)

    return object_name, True


def fill_split(shapenet_root, pool_iter, split_name, output_root, target, manifest_rows):
    # Draws from pool_iter (a shared iterator over an already-shuffled pool)
    # until `target` objects are actually prepared, pulling extra candidates
    # to replace any that turn out unusable (missing/corrupt .obj) instead of
    # silently under-filling the split.
    split_dir = os.path.join(output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    added = 0
    exhausted = False
    while added < target:
        try:
            synset_id, model_id, category_name = next(pool_iter)
        except StopIteration:
            exhausted = True
            break
        object_name, ok = prepare_object(shapenet_root, synset_id, model_id, split_dir)
        if ok:
            added += 1
            manifest_rows.append({
                "split": split_name,
                "categoria": category_name,
                "synset_id": synset_id,
                "model_id": model_id,
                "carpeta": os.path.join(split_name, object_name),
            })
    if exhausted:
        print("ADVERTENCIA: se agoto el pool antes de completar {} para {} (llegaron a {})".format(
            target, split_name, added))
    print("{}: {} objetos preparados en {}".format(split_name, added, split_dir))


def main(config_path):
    params = Params(config_path)
    shapenet_root = params.getParameter("shapenet_root")
    output_root = params.getParameter("output_root")
    seed = params.getParameter("seed")
    sizes = params.getParameter("split_sizes")

    os.makedirs(output_root, exist_ok=True)
    rng = random.Random(seed)

    seen_pool = build_pool(shapenet_root, SEEN_CATEGORIES)
    novel_pool = build_pool(shapenet_root, NOVEL_CATEGORIES)

    rng.shuffle(seen_pool)
    rng.shuffle(novel_pool)

    n_train = sizes["train"]
    n_val = sizes["val"]
    n_test_similar = sizes["test_similar"]
    n_test_novel = sizes["test_novel"]

    needed_seen = n_train + n_val + n_test_similar
    if len(seen_pool) < needed_seen:
        print("ADVERTENCIA: pool 'seen' tiene {} modelos disponibles, se pidieron {} (train+val+test_similar) -- "
              "se va a repartir menos de lo pedido".format(len(seen_pool), needed_seen))
    if len(novel_pool) < n_test_novel:
        print("ADVERTENCIA: pool 'novel' tiene {} modelos disponibles, se pidieron {} -- "
              "se va a repartir menos de lo pedido".format(len(novel_pool), n_test_novel))

    seen_iter = iter(seen_pool)
    novel_iter = iter(novel_pool)

    manifest_rows = []
    fill_split(shapenet_root, seen_iter, "train", output_root, n_train, manifest_rows)
    fill_split(shapenet_root, seen_iter, "val", output_root, n_val, manifest_rows)
    fill_split(shapenet_root, seen_iter, "test_similar", output_root, n_test_similar, manifest_rows)
    fill_split(shapenet_root, novel_iter, "test_novel", output_root, n_test_novel, manifest_rows)

    # El manifiesto se reescribe entero cada corrida (refleja la seleccion actual;
    # el muestreo con semilla fija es reproducible mientras el pool disponible no cambie).
    manifest_path = os.path.join(output_root, "split_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "categoria", "synset_id", "model_id", "carpeta"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Manifiesto: {}".format(manifest_path))


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "dataexample/paramsShapeNetSplit.yaml"
    main(config_path)
