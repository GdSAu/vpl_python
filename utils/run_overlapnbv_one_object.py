## Corre OverlapNBV (Reconstructor) sobre UN solo objeto, en su propio proceso.
## Ver run_overlapnbv_shapenet.py: reusar un Reconstructor para varios objetos en el
## mismo proceso segfaultea (el renderer offscreen de Open3D no tolera crearse dos
## veces en el mismo proceso) -- cada objeto se corre aislado, y el orquestador
## tolera que este proceso muera y sigue con el siguiente.

import os
import sys
import pandas as pd
from Reconstructor.ReconstructorSimple import ReconstructorSimple


def main(config_path, folder_path, object_name):
    folder_path = folder_path.rstrip("/") + "/"
    r = ReconstructorSimple(config_path)
    r.direccion = folder_path
    r.objeto = object_name

    combined_path = os.path.join(folder_path, r.csv_name)
    backup_path = combined_path + ".bak"
    # runReconstruction() sobreescribe combined_path con SOLO las filas de este objeto
    # (comportamiento de Reconstructor, compartido con otros planificadores -- no se toca
    # ese codigo). Se respalda lo acumulado de objetos anteriores antes de correr, para no
    # perderlo si este objeto crashea a medias despues de que runReconstruction() ya
    # sobreescribio el archivo.
    if os.path.exists(combined_path):
        os.replace(combined_path, backup_path)

    r.runReconstruction()

    this_object_df = pd.read_csv(combined_path) if os.path.exists(combined_path) else None
    previous_df = pd.read_csv(backup_path) if os.path.exists(backup_path) else None

    if previous_df is not None and this_object_df is not None:
        merged_df = pd.concat([previous_df, this_object_df], ignore_index=True)
    else:
        merged_df = previous_df if previous_df is not None else this_object_df

    if merged_df is not None:
        tmp_path = combined_path + ".tmp"
        merged_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, combined_path)

    if os.path.exists(backup_path):
        os.remove(backup_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
