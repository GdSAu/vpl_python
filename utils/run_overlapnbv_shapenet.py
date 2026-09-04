## Corre OverlapNBV (Reconstructor) sobre los splits de ShapeNet ya preparados con NBVDE.
##
## Cada objeto se lanza en su PROPIO proceso (run_overlapnbv_one_object.py) -- reusar un
## solo Reconstructor/renderer de Open3D para varios objetos en el mismo proceso
## segfaultea (verificado: falla incluso al segundo objeto, identicos entre si). El
## marcador de saltado (Point_cloud/<carpeta_metodo> ya existe) es el mismo que usa
## NBVDE.py, y absorbe tanto objetos ya terminados como los que murieron a medias por
## un crash -- si un objeto muere a medias, su proximo intento lo vuelve a marcar como
## "en progreso" y lo reprocesa desde cero (mismo tradeoff ya aceptado en NBVDE).

import os
import subprocess
import sys
from utils.reconstructorParams import Params


def run_split(folder_path, config_path, carpeta_metodo):
    folder_path = folder_path.rstrip("/") + "/"
    objects = sorted(os.listdir(folder_path))
    processed, skipped, failed = 0, 0, 0
    for object_name in objects:
        object_dir = os.path.join(folder_path, object_name)
        if not os.path.isdir(object_dir):
            continue
        marker_dir = os.path.join(object_dir, "Point_cloud", carpeta_metodo)
        if os.path.exists(marker_dir):
            skipped += 1
            continue

        result = subprocess.run(
            [sys.executable, "run_overlapnbv_one_object.py", config_path, folder_path, object_name]
        )
        if result.returncode != 0:
            print("ERROR en {} (codigo {})".format(object_name, result.returncode))
            failed += 1
        else:
            processed += 1

    print("{}: {} procesados, {} saltados, {} fallidos".format(folder_path, processed, skipped, failed))


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'dataexample/paramsOverlapNBV.yaml'
    params = Params(config_path)

    object_folders = params.getParameter("carpetas.objectFolder")
    if isinstance(object_folders, str):
        object_folders = [object_folders]
    carpeta_metodo = params.getParameter("carpetas.carpeta_metodo")

    for folder_path in object_folders:
        print("=== {} ===".format(folder_path))
        run_split(folder_path, config_path, carpeta_metodo)
