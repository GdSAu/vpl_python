import open3d as o3d
import numpy as np
import octomap
import trimesh
import matplotlib.pyplot as plt
import cv2

def scale_and_translate(mesh, scale_factor=0.4):
  max = mesh.get_max_bound()
  scale = scale_factor/np.max(max)
  mesh.scale(scale, center= mesh.get_center())#Scale mesh
  traslacion = - np.asarray(mesh.get_min_bound())[2]
  traslacion =  [0,0,traslacion]
  mesh.translate(traslacion) #translate mesh
  return mesh

def Get_RGBD(render, fov, center, eye, up, direccion, i, itera):
   ''' (render,fov,center,eye,up) -> RGB-D
      render : es el objeto de escena que contiene el objeto
      fov : vertical_field of view
      center: camera center (orientation, where the camera see)
      eye: camera eye (position)
      up: camera up vector ()
      direccion: root folder direction
      i = index in loop
      '''
   render.setup_camera(fov, center, eye, up)
   img = render.render_to_image()
   depth = render.render_to_depth_image()
   o3d.io.write_image(direccion +"RGB/"+ itera +"RGB_{}.png".format(i), img, 9)
   cv2.imwrite(direccion + "Depth/"+ itera +"D_{}.tiff".format(i), np.asarray(depth))

def Get_PointcloudGT (direccion, mesh, itera):
  #Creamos nube de puntos GT
  size = np.asarray(mesh.vertices).shape[0]
  mesh.compute_vertex_normals()
  p_gt = mesh.sample_points_uniformly(number_of_points=size*10)
  o3d.io.write_point_cloud(direccion + "Point_cloud/"+ itera +"cloud_gt.pcd", p_gt, write_ascii=True)# GT cloud 

def Nube_acumulada_filtrada(P_acu, z, voxel_size =0.0005):
    #P_acu,z -> P_acu
    Nube_acc = o3d.geometry.PointCloud()
    Nube_acc.points = o3d.utility.Vector3dVector(np.vstack((P_acu.points,z.points)))
    Nube_acc = Nube_acc.voxel_down_sample(voxel_size= voxel_size)
    return Nube_acc 

def Get_Pointcloud(scene, fov, center, eye, up, width, height, direccion, i,itera, save_acc = False):
  '''scene: scene object which contains the mesh
    fov: vertical field of view
    center: camera center (orientation, where the camera see)
    eye: camera eye (position)
    up: camera up vector ()
    width, height : width and height of the image 
    direccion: root folder direction
    i = index in loop
    save_acc = (boolean value) to save accumulated pointcloud
  '''

  # (scene, fov, center, eye, up, width, height) -> point cloud
  rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg= fov,
    center= center,
    eye= eye,
    up= up,
    width_px=width,
    height_px=height,
  )
  # We can directly pass the rays tensor to the cast_rays function.
  ans = scene.cast_rays(rays)
  hit = ans['t_hit'].isfinite()
  points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points.numpy())
  #print(pcd)
  o3d.io.write_point_cloud(direccion + "/Point_cloud/"+ itera +"cloud_{}.pcd".format(i), pcd, write_ascii=True)# cloud in t-time

  if save_acc == True:
    # if the poincloud is new it begins the cloud, else the new perception is added to the accumulated cloud
    if i == 0:
      Nube_acc = pcd
    else:
      p_c = o3d.io.read_point_cloud(direccion + "/Point_cloud/" + itera + "cloud_acc.pcd")
      #p_c.point["positions"] = o3d.core.concatenate((p_c.point["positions"], pcd.point["positions"]), 0)
      Nube_acc = o3d.geometry.PointCloud()
      Nube_acc.points = o3d.utility.Vector3dVector(Nube_acumulada_filtrada(p_c,pcd, voxel_size=0.0005).points)
      #print(Nube_acc)
    o3d.io.write_point_cloud(direccion + "/Point_cloud/"+ itera + "cloud_acc.pcd", Nube_acc, write_ascii=True)# accumulated cloud 

def Get_octree(octree, direccion, i, itera, origin, puntos, dim_arreglo):
  '''octree -> occupation prob [n,]
    octree: octree object
    direccion: root folder direction
    i = index in loop
    origin = camera eye (position)
    '''
  p_c = o3d.io.read_point_cloud(direccion + "/Point_cloud/"+ itera +"cloud_{}.pcd".format(i), remove_nan_points=True, remove_infinite_points = True)
  octree.insertPointCloud(
      pointcloud= np.asarray(p_c.points), 
      origin= np.asarray(origin), #Measurement origin
      maxrange=-1, # maximum range for how long individual beams are inserted
      )  
  #arreglo = np.ndarray([29791], dtype=float)
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
              #arreglo[j] = 0.5
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
