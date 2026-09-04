import sys
from utils.reconstructorParams import Params
from Viewplanner.NBVDE import NBVDatasetExtractor


def run_one(objects_dir, params):
    extractor = NBVDatasetExtractor(
        objects_dir=objects_dir,
        viewspace_path=params.getParameter("carpetas.viewspace"),
        gt_points=params.getParameter("variables.gt_points"),
        hpr_radius_factor=params.getParameter("variables.hpr_radius_factor"),
        use_extent=params.getParameter("simulation.use_extent"),
        fov=params.getParameter("camera.fov"),
        up=params.getParameter("camera.up"),
        img_W=params.getParameter("camera.img_W"),
        img_H=params.getParameter("camera.img_H"),
        umbral=params.getParameter("variables.umbral"),
        voxel_resolution=params.getParameter("variables.voxel_resolution"),
        voxel_voxdim=params.getParameter("variables.voxel_dim"),
        csv_name=params.getParameter("carpetas.csv_name"),
    )

    extractor.run(
        num_extractions=params.getParameter("variables.num_extractions"),
        max_scans=params.getParameter("variables.max_scans"),
    )


if __name__ == '__main__':
    params_file = sys.argv[1] if len(sys.argv) > 1 else 'dataexample/paramsNBVDE.yaml'
    params = Params(params_file)

    object_folders = params.getParameter("carpetas.objectFolder")
    if isinstance(object_folders, str):
        object_folders = [object_folders]

    for objects_dir in object_folders:
        print("=== {} ===".format(objects_dir))
        run_one(objects_dir, params)
