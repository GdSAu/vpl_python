## NBV Training Data Extraction
# Based on the paper PC-NBV, Algorithm 1: greedy, ground-truth-driven view
# selection used to generate per-view surface-coverage-gain labels.
#
# Candidate views are not re-rendered against the mesh: they are visibility
# subsets (Open3D hidden_point_removal) of a single uniformly sampled GT
# point cloud, so "new points" / "coverage gain" reduce to exact index-set
# operations over that one array instead of a nearest-neighbor distance.
#
# Once a round's NBV is chosen, the same view/accumulated-cloud subsets are
# additionally persisted in the same folder/file convention Reconstructor.py
# uses for its live planners (Point_cloud/RGB/Depth/Octree per method per
# iteration), plus a real RGB-D render, an incremental octree, a NerfStudio-
# ready pose JSON and a metrics CSV — written directly inside each object's
# own folder, same as Reconstructor.py does (no separate mirrored output tree).

import os
import numpy as np
import open3d as o3d
import pandas as pd
from Simulator.sceneLoader import SceneLoader
from Robotsensor.sensor import Sensor
from Partialmodel.partialModel import PMOctomapPy, PMPointCloudPy
from utils.utils_metrics import chamfer_distance, getCobertura
from utils.utils_save import save_nerfstudio_transforms


class NBVDatasetExtractor:
    def __init__(self, objects_dir, viewspace_path, gt_points=16384,
                 hpr_radius_factor=100, use_extent=False,
                 fov=45, up=(0, 1, 0), img_W=240, img_H=320,
                 umbral=0.0035, voxel_resolution=0.025, voxel_voxdim=32,
                 carpeta_metodo="NBVDE/", csv_name="NBVDE_metrics.csv"):
        self.objects_dir = objects_dir
        self.viewspace = np.loadtxt(viewspace_path)
        self.view_num = self.viewspace.shape[0]
        self.gt_points = gt_points
        self.hpr_radius_factor = hpr_radius_factor
        self.use_extent = use_extent
        self.fov = fov
        self.up = up
        self.img_W = img_W
        self.img_H = img_H
        self.umbral = umbral
        self.voxel_resolution = voxel_resolution
        self.voxel_voxdim = voxel_voxdim
        self.carpeta_metodo = carpeta_metodo if carpeta_metodo.endswith("/") else carpeta_metodo + "/"
        self.csv_name = csv_name
        self.random = np.random.default_rng()
        self.metrics_rows = []

    def __getVisibleIndices(self, gt_pc):
        """For each candidate view, the indices of gt_pc visible from it."""
        diameter = np.linalg.norm(np.asarray(gt_pc.get_max_bound()) - np.asarray(gt_pc.get_min_bound()))
        radius = diameter * self.hpr_radius_factor
        visible_idx = []
        for j in range(self.view_num):
            _, pt_map = gt_pc.hidden_point_removal(self.viewspace[j], radius)
            visible_idx.append(np.asarray(pt_map, dtype=np.int64))
        return visible_idx

    def extractObject(self, object_dir, num_extractions=1, max_scans=10):
        object_name = os.path.basename(os.path.normpath(object_dir))

        scene = SceneLoader(object_dir, floor=False, use_extent=self.use_extent)
        gt_pc = scene.mesh.sample_points_uniformly(number_of_points=self.gt_points)
        gt_points_array = np.asarray(gt_pc.points)
        visible_idx = self.__getVisibleIndices(gt_pc)
        cent = scene.mesh.get_center()

        np.savetxt(os.path.join(object_dir, "gt.xyz"), gt_points_array)

        render = scene.create_offrender_scene(img_W=self.img_W, img_H=self.img_H)
        sensor = Sensor(self.fov, self.up, self.img_W, self.img_H, raycast=False, render=render, scene=None)

        for ex_index in range(num_extractions):
            iter_str = "{}/".format(ex_index)
            point_cloud_method_dir = os.path.join(object_dir, "Point_cloud", self.carpeta_metodo)
            point_cloud_iter_dir = os.path.join(point_cloud_method_dir, iter_str)
            rgb_iter_dir = os.path.join(object_dir, "RGB", self.carpeta_metodo, iter_str)
            depth_iter_dir = os.path.join(object_dir, "Depth", self.carpeta_metodo, iter_str)
            octree_iter_dir = os.path.join(object_dir, "Octree", self.carpeta_metodo, iter_str)
            view_vectors_iter_dir = os.path.join(object_dir, "ViewVectors", self.carpeta_metodo, iter_str)
            for d in (point_cloud_iter_dir, rgb_iter_dir, depth_iter_dir, octree_iter_dir, view_vectors_iter_dir):
                os.makedirs(d, exist_ok=True)

            o3d.io.write_point_cloud(os.path.join(point_cloud_iter_dir, "cloud_gt.pcd"), gt_pc, write_ascii=True)

            partial_model = PMOctomapPy(self.voxel_resolution, self.voxel_voxdim)
            pc_accumulator = PMPointCloudPy(self.voxel_resolution)

            viewstate = np.zeros(self.view_num, dtype=np.int32)  # 0 unselected, 1 selected
            covered_mask = np.zeros(self.gt_points, dtype=bool)

            init_view = int(self.random.integers(0, self.view_num))
            viewstate[init_view] = 1
            covered_mask[visible_idx[init_view]] = True
            cur_cov = covered_mask.sum() / self.gt_points

            eyes_used = []

            for scan_index in range(max_scans):
                print("{} ex{}: coverage {:.4f} at scan {}".format(object_name, ex_index, cur_cov, scan_index))

                np.save(os.path.join(view_vectors_iter_dir, "{}_viewstate.npy".format(scan_index)), viewstate)

                target_value = np.zeros((self.view_num, 1))  # surface coverage gain for each view
                for j in range(self.view_num):
                    new_count = np.count_nonzero(~covered_mask[visible_idx[j]])
                    target_value[j, 0] = new_count / self.gt_points

                np.save(os.path.join(view_vectors_iter_dir, "{}_target_value.npy".format(scan_index)), target_value)

                best_view = int(np.argmax(target_value[:, 0]))
                gain = target_value[best_view, 0]
                print("{} ex{}: choose view {} add coverage {:.4f}".format(
                    object_name, ex_index, best_view, gain))

                eye = self.viewspace[best_view]

                # --- simulator-style capture for the chosen view (does not affect the pick above) ---
                rgb_path = os.path.join(rgb_iter_dir, "RGB_{}.png".format(scan_index))
                depth_path = os.path.join(depth_iter_dir, "D_{}.tiff".format(scan_index))
                sensor.saveRGBD(cent, eye, rgb_path, depth_path)

                view_cloud = o3d.geometry.PointCloud()
                view_cloud.points = o3d.utility.Vector3dVector(gt_points_array[visible_idx[best_view]])
                cloud_path = os.path.join(point_cloud_iter_dir, "cloud_{}.pcd".format(scan_index))
                o3d.io.write_point_cloud(cloud_path, view_cloud, write_ascii=True)

                octree_path = os.path.join(octree_iter_dir, "octree_{}.ot".format(scan_index))
                partial_model.updateWithScan(cloud_path, eye)
                partial_model.savePartialModel(octree_path)

                acc_pc_path = os.path.join(point_cloud_iter_dir, "{}_acc_pc.pcd".format(scan_index))
                pc_accumulator.updateWithScan(cloud_path)
                pc_accumulator.savePartialModel(acc_pc_path)

                cur_cov += gain
                viewstate[best_view] = 1
                covered_mask[visible_idx[best_view]] = True
                eyes_used.append(eye)

                acc_cloud = o3d.geometry.PointCloud()
                acc_cloud.points = o3d.utility.Vector3dVector(gt_points_array[covered_mask])
                o3d.io.write_point_cloud(os.path.join(point_cloud_iter_dir, "cloud_acc.pcd"), acc_cloud, write_ascii=True)

                CD = chamfer_distance(point_cloud_method_dir, iter_str)
                cov_real = getCobertura(point_cloud_method_dir, iter_str, scan_index, umbral=self.umbral)

                self.metrics_rows.append({
                    "id_objeto": object_name,
                    "extraccion": ex_index,
                    "scan_index": scan_index,
                    "vista_elegida": best_view,
                    "eye_x": eye[0], "eye_y": eye[1], "eye_z": eye[2],
                    "ganancia_hpr": gain,
                    "cobertura_hpr_acumulada": cur_cov,
                    "chamfer": CD,
                    "cobertura": cov_real,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "cloud_path": cloud_path,
                    "octree_path": octree_path,
                    "acc_pc_path": acc_pc_path,
                })

            save_nerfstudio_transforms(object_dir, eyes_used, cent, self.up, self.fov,
                                        self.carpeta_metodo, iter_str)

    def run(self, num_extractions=1, max_scans=10, max_objects=None):
        object_list = sorted(os.listdir(self.objects_dir))
        processed = 0
        for object_name in object_list:
            if max_objects is not None and processed >= max_objects:
                break
            object_dir = os.path.join(self.objects_dir, object_name)
            if not os.path.isdir(object_dir):
                continue
            marker_dir = os.path.join(object_dir, "Point_cloud", self.carpeta_metodo)
            if os.path.exists(marker_dir):
                print("skip " + marker_dir)
                continue
            # Marked as claimed before the risky mesh-load/render work below,
            # so a hard native crash (e.g. a corrupt texture aborting the
            # whole process, uncatchable in Python) still leaves this object
            # skippable on the next retry instead of crash-looping on it.
            os.makedirs(marker_dir, exist_ok=True)
            print("processing " + object_dir, flush=True)
            self.extractObject(object_dir, num_extractions=num_extractions, max_scans=max_scans)
            processed += 1

        if self.metrics_rows:
            dataframe = pd.DataFrame(self.metrics_rows)
            dataframe.to_csv(os.path.join(self.objects_dir, self.csv_name), index=False)
