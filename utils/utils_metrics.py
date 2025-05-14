import numpy as np
import open3d as o3d

def chamfer_distance(dir_carpeta, itera):
    """This function measure the Chamfer distance between two point clouds,
     ground_truth and accumulated"""
    distance_1_to_2 = 0
    distance_2_to_1 = 0

    points1 = o3d.io.read_point_cloud( dir_carpeta + itera + "cloud_gt.pcd").voxel_down_sample(voxel_size=0.0035)
    points1 = np.asarray(points1.points)
    points2 = np.asarray(o3d.io.read_point_cloud( dir_carpeta + itera +  "cloud_acc.pcd").points)

    # Compute distance from each point in arr1 to arr2
    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1)**2, axis=1))
        min_distance = np.min(distances)
        distance_1_to_2 += min_distance

    # Compute distance from each point in arr2 to arr1
    for p2 in points2:
        distances = np.sqrt(np.sum((points1 - p2)**2, axis=1))
        min_distance = np.min(distances)
        distance_2_to_1 += min_distance

    return (distance_1_to_2 + distance_2_to_1) / (len(points1) + len(points2))


def  Get_surface_coverage_gain(direccion,i, itera, umbral=0.0037): # 0.0037 #P_t - P_t-1 < umbral
    
    pt = o3d.io.read_point_cloud(direccion + itera + "cloud_{}.pcd".format(i), remove_nan_points=True, remove_infinite_points = True)
    pt_1 = o3d.io.read_point_cloud(direccion + itera +  "cloud_{}.pcd".format(i-1), remove_nan_points=True, remove_infinite_points = True)
    dist = np.asarray(pt.compute_point_cloud_distance(pt_1))
    ind = np.where(dist < umbral)[0] # puntos de intersección
    return ind, dist

def stop_condition(direccion,ind, i, itera, umbral= 0.8):
    pt = o3d.io.read_point_cloud(direccion + itera +  "cloud_{}.pcd".format(i), remove_nan_points=True, remove_infinite_points = True)
    size = len(ind)/(np.asarray(pt.points).shape[0])
    if size < umbral:
        condicion = False
    else:
        condicion = True
    return condicion, size

def Get_cloud_distance(direccion, i, itera):

    if i > 0:
        ind, _ = Get_surface_coverage_gain(direccion, i, itera)
        condicion, coverage_gain = stop_condition(direccion,ind,i, itera)
    else: 
        condicion = False
        coverage_gain = 0.0

    return condicion, coverage_gain

def getCobertura(dir_carpeta, itera, i, umbral=0.0037):
    '''
    P_acu = nube acumulada
    W_obj = nube del objeto
    '''
    if i>0:
        points1 = o3d.io.read_point_cloud( dir_carpeta + itera +  "cloud_gt.pcd")
        points1 = points1.voxel_down_sample(voxel_size=umbral)
        points2 = o3d.io.read_point_cloud( dir_carpeta + itera +  "cloud_acc.pcd")
        dist = np.asarray(points1.compute_point_cloud_distance(points2)) #KD-Tree
        ind = np.where(dist < umbral)[0]
        percentage = (len(ind)/np.asarray(points1.points).shape[0])*100
    else:
        percentage = 0.0
    return percentage