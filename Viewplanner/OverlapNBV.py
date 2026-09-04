## Overlap-gated greedy NBV, ported from NBV_iterative/Experiment1.ipynb.
# Each round, evaluates every candidate view in `viewspace`, keeps only the
# ones whose visible surface overlaps enough with what's already been
# scanned (Traslape > thres1), and among those picks the one that adds the
# most coverage against the GT (Cobertura). Candidate visibility uses the
# same hidden_point_removal technique as NBVDE.py (index-set operations on
# one sampled GT cloud) instead of real raycasting -- Cobertura/Traslape
# both reduce to intersection/count over those index sets:
#   Cobertura(A, B, umbral)  == fraction of B explained by A
#   Traslape(z, P_acu, gap)  == Cobertura(z, P_acu, gap)  (already-covered explained by candidate)
#   coverage gain of picking j == fraction of GT newly explained by j (same as NBVDE's target_value)

## FROM Mendoza et al., "NBV-Net: una red neuronal convolucional 3D para predecir la siguiente mejor vista." 2018

import os
import numpy as np
import open3d as o3d
from Viewplanner.viewPlanner import NBVPlanner
from Partialmodel.partialModel import PMOctomapPy


class OverlapNBV(NBVPlanner):
    def __init__(self, robot_sensor, partial_model, viewspace_path, gt_cloud_path,
                 view_vectors_dir, point_cloud_dir, octree_dir,
                 hpr_radius_factor=100, thresh1=0.0, thresh2=0, voxel_resolution=0.025, voxel_voxdim=32):
        super().__init__(robot_sensor, partial_model)
        self.robot_sensor = robot_sensor
        self.partial_model = partial_model
        self.viewspace = np.loadtxt(viewspace_path)
        self.view_num = self.viewspace.shape[0]
        self.thresh1 = thresh1
        self.thresh2 = thresh2

        gt_pc = o3d.io.read_point_cloud(gt_cloud_path)  # el mismo cloud_gt.pcd que ya escribio Reconstructor
        self.gt_points_array = np.asarray(gt_pc.points)
        self.gt_points = len(self.gt_points_array)
        self.visible_idx = self.__getVisibleIndices(gt_pc, hpr_radius_factor)

        self.covered_mask = np.zeros(self.gt_points, dtype=bool)  # lo que este planificador ya cubrio
        self.viewstate = np.zeros(self.view_num, dtype=np.int32)  # 0 no elegida, 1 elegida
        self.scan_index = 0
        self.last_eye = None  # eye elegido la ultima vez (origin para el octree); None solo en la ronda 0

        self.view_vectors_dir = view_vectors_dir
        self.point_cloud_dir = point_cloud_dir  # Point_cloud/<metodo>/<iter>/, ya creado por Reconstructor._createFolder
        self.octree_dir = octree_dir            # Octree/<metodo>/<iter>/, idem
        os.makedirs(self.view_vectors_dir, exist_ok=True)
        os.makedirs(self.octree_dir, exist_ok=True)

        # Octree propio del planificador (Reconstructor.PM es PMPointCloudPy en este modo; no puede
        # ser los dos a la vez) -- mismo patron de doble tracking que ya usa NBVDE.py.
        self.octree_model = PMOctomapPy(voxel_resolution, voxel_voxdim)

    def __getVisibleIndices(self, gt_pc, hpr_radius_factor):
        """For each candidate view, the indices of gt_pc visible from it."""
        diameter = np.linalg.norm(np.asarray(gt_pc.get_max_bound()) - np.asarray(gt_pc.get_min_bound()))
        radius = diameter * hpr_radius_factor
        visible_idx = []
        for j in range(self.view_num):
            _, pt_map = gt_pc.hidden_point_removal(self.viewspace[j], radius)
            visible_idx.append(np.asarray(pt_map, dtype=np.int64))
        return visible_idx

    def __countNarfKeypoints(self, indices, k_neighbors=20, nms_radius_factor=2.0):
        """Cuenta keypoints tipo NARF (Steder et al.) sobre la interseccion z' inter Pacu,
        sin depender de PCL: valor de interes = variacion superficial local (PCA de la
        vecindad k-NN, razon del menor eigenvalor sobre la suma) + supresion de no-maximos
        en un radio proporcional al espaciado promedio de la nube."""
        if len(indices) < k_neighbors:
            return 0

        pts = self.gt_points_array[indices]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        tree = o3d.geometry.KDTreeFlann(pcd)

        n = len(pts)
        interest = np.zeros(n)
        mean_nn_dist = 0.0
        for i in range(n):
            k, idx, dist2 = tree.search_knn_vector_3d(pts[i], k_neighbors)
            if k < 3:
                continue
            neighborhood = pts[np.asarray(idx)]
            cov = np.cov(neighborhood.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 0, None)
            total = eigvals.sum()
            interest[i] = eigvals[0] / total if total > 0 else 0.0
            mean_nn_dist += np.sqrt(dist2[1]) if k > 1 else 0.0
        mean_nn_dist = mean_nn_dist / n if n > 0 else 0.0
        nms_radius = nms_radius_factor * mean_nn_dist

        keypoint_count = 0
        for i in range(n):
            if nms_radius > 0:
                _, idx, _ = tree.search_radius_vector_3d(pts[i], nms_radius)
            else:
                idx = [i]
            if interest[i] >= interest[np.asarray(idx)].max():
                keypoint_count += 1
        return keypoint_count

    def savePartialModel(self, file_name):
        super().savePartialModel()
        self.partial_model.savePartialModel(file_name)

    def updateWithScan(self, **kwargs):
        pc_file = kwargs["pointcloud"]
        self.partial_model.updateWithScan(pc_file)

        # Reconstructor ya guarda cloud_acc.pcd (un solo archivo, se sobreescribe cada ronda).
        # Acá agregamos un snapshot indexado por ronda del MISMO acumulado (self.partial_model es
        # el mismo PMPointCloudPy que Reconstructor ya actualizo), igual que {scan}_acc_pc.pcd en NBVDE.
        acc_path = os.path.join(self.point_cloud_dir, "{}_acc_pc.pcd".format(self.scan_index))
        self.partial_model.savePartialModel(acc_path)

        # Octree por ronda, igual que NBVDE. Se salta en la ronda 0: el escaneo de esa ronda es
        # eye_init (de poses.npy), que este planificador no elige y no tiene registrado como origin.
        if self.last_eye is not None:
            self.octree_model.updateWithScan(pc_file, self.last_eye)
            octree_path = os.path.join(self.octree_dir, "{}_octree.ot".format(self.scan_index))
            self.octree_model.savePartialModel(octree_path)

    def PlanNBV(self):
        super().PlanNBV()
        np.save(os.path.join(self.view_vectors_dir, "{}_viewstate.npy".format(self.scan_index)), self.viewstate)

        covered_count = self.covered_mask.sum()
        target_value = np.zeros((self.view_num, 1))
        for j in range(self.view_num):
            vis = self.visible_idx[j]
            # Traslape: fraccion de lo ya cubierto que explica esta candidata.
            overlap = np.count_nonzero(self.covered_mask[vis]) / covered_count if covered_count > 0 else 1.0
            if overlap > self.thresh1:
                # Segundo gate: keypoints NARF sobre z' inter Pacu (lo visible desde la
                # candidata que ya esta cubierto por el acumulado). Igual que el gate de
                # traslape arriba, se saltea cuando aun no hay nada acumulado (ronda 0):
                # la interseccion seria vacia para toda candidata, no un criterio real.
                narf_ok = True
                if covered_count > 0:
                    intersection = np.intersect1d(vis, np.where(self.covered_mask)[0], assume_unique=True)
                    narf_ok = self.__countNarfKeypoints(intersection) > self.thresh2
                if narf_ok:
                    # Cobertura incremental si se eligiera esta vista (= target_value de NBVDE).
                    target_value[j, 0] = np.count_nonzero(~self.covered_mask[vis]) / self.gt_points

        np.save(os.path.join(self.view_vectors_dir, "{}_target_value.npy".format(self.scan_index)), target_value)

        # max_inc en el pseudocodigo arranca en 0 y v* solo se mueve si "inc > max_inc" --
        # si ningun candidato paso ambos gates con ganancia > 0, target_value queda todo en
        # cero y argmax caeria en el indice 0 por defecto (una vista arbitraria, no elegida
        # de verdad). Replicando la semantica del pseudocodigo: v* se queda igual que la
        # ronda anterior en ese caso (no se marca ni se mueve nada nuevo).
        if target_value.max() > 0:
            best_view = int(np.argmax(target_value[:, 0]))
            self.viewstate[best_view] = 1
            self.covered_mask[self.visible_idx[best_view]] = True
            self.last_eye = self.viewspace[best_view]
        self.scan_index += 1
        return self.last_eye
