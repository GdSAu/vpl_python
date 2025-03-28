import numpy as np
import octomap
import open3d as o3d
import trimesh

def getOccupancyProbs(octree, puntos, dim_arreglo= 29791):

    arreglo = np.full((dim_arreglo), 0.5)
    j = 0 
    for i in puntos:
        #Get occupancy probability given a position (x,y,z)
        node = octree.search(i, 0)
        # If the returned value is different than -1, indicates that a node is in that position
        if node is not None:
            try:
                probability = node.getOccupancy() #Can use getValues(), but extracts the log-odds
                arreglo[j] = probability 
            except:
                pass
        j += 1 
    return arreglo  

def Get_voxpoints(max_dim=.4, dim = 31):
  #BBOX min & max
  resolution = max_dim/dim
  aabb_min = np.asarray([-(max_dim/2),-(max_dim/2),0.0])
  aabb_max = np.asarray([(max_dim/2),(max_dim/2),max_dim])
  center = (aabb_min + aabb_max) / 2
  dimension = np.array([dim, dim, dim]) # Voxelization dimensions
  origin = center - dimension/2  * resolution
  #New BBox given the new resolution
  grid = np.full(dimension, -1, np.int32)
  transform = trimesh.transformations.scale_and_translate(
      scale=resolution, translate=origin
  )
  # Voxelgrid encoding (create grid)
  points = trimesh.voxel.VoxelGrid(encoding=grid, transform=transform).points # Voxel grid con los puntos de la nube
  puntos = np.asarray(points)
  return puntos

def visualize_octree_voxels(octree, occupied_threshold=0.5):
    """
    Visualize the octree as a voxel grid or colored point cloud
    
    Parameters:
    - octree: OctoMap OcTree object
    - occupied_threshold: Threshold for considering a voxel as occupied
    """
    # Extract points and occupancies
    puntos = Get_voxpoints()
    occupancies = getOccupancyProbs(octree, puntos)
    
    if puntos is None:
        print("Failed to extract points from octree")
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(puntos)
    # Optionally color points based on occupancy
    # Optionally color points based on occupancy
    if occupancies is not None:
        # Create RGBA colors
        rgba_colors = np.zeros((len(puntos), 4))
        
        # Define color logic
        for i, occupancy in enumerate(occupancies):
            if occupancy == 0.5:
                # Yellow color with transparency for 0.5 occupancy
                rgba_colors[i] = [1, 1, 0, 0.1]  # Yellow with 50% transparency
            elif occupancy > 0.5:
                # Blue color for 1.0 occupancy
                rgba_colors[i] = [0, 0, 1, 1]  # Solid blue
            else:
                # Original color mapping logic
                rgba_colors[i] = [1, 1, 1, 0]  # Solid blue
        
        # Set colors to point cloud
        pcd.colors = o3d.utility.Vector3dVector(rgba_colors[:, :3])
        
    # Visualize
    o3d.visualization.draw_geometries([pcd])