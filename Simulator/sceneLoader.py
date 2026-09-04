import os
import open3d as o3d
from utils.utils_raycast import SimplePointCloudRaycast
from utils.utils_o3d import scale_and_translate

class SceneLoader:
    '''
    This class load the meshes and textures of a model
    also an floor can be added
    returns an: 
        offrender scene - to extract depth and RGB images
        raycast scene - to extract pointclouds 
    '''

    def __init__(self, mesh_file, floor=True, scale=True, scale_factor=0.4, floor_size= 20, floor_depth=0.01, use_extent=False):
        self.mesh_file = mesh_file
        self.scale = scale
        self.floort = floor
        self.scale_factor = scale_factor
        self.floor_size = floor_size
        self.floor_depth = floor_depth
        self.use_extent = use_extent
        self.mesh, self.mesh_material = self.load_mesh()
        if self.floort == True:
            self.floor, self.floor_material = self.create_floor()

    def __content_matches_extension(self, path):
      # Some ShapeNet textures are mislabeled (e.g. a PNG saved as texture.jpg).
      # o3d.io.read_image() picks its decoder from the extension, and a real
      # libjpeg hitting PNG bytes aborts the whole process (not a catchable
      # Python exception) -- so candidates are validated by magic bytes first.
      try:
          with open(path, 'rb') as f:
              header = f.read(8)
      except OSError:
          return False
      ext = path.lower()
      if ext.endswith('.png'):
          return header.startswith(b'\x89PNG\r\n\x1a\n')
      if ext.endswith(('.jpg', '.jpeg')):
          return header.startswith(b'\xff\xd8\xff')
      return False

    def __find_texture(self):
      texture_path = self.mesh_file + '/meshes/texture.png'
      if os.path.exists(texture_path) and self.__content_matches_extension(texture_path):
          return texture_path
      images_dir = self.mesh_file + '/images'
      if os.path.isdir(images_dir):
          for fname in sorted(os.listdir(images_dir)):
              if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                  candidate = os.path.join(images_dir, fname)
                  if self.__content_matches_extension(candidate):
                      return candidate
      return None

    def load_mesh(self):
      mesh = o3d.io.read_triangle_mesh(self.mesh_file + '/meshes/model.obj', True)
      if self.scale == True:
        mesh = scale_and_translate(mesh, scale_factor=self.scale_factor, use_extent=self.use_extent)
      material = o3d.visualization.rendering.MaterialRecord()
      texture_path = self.__find_texture()
      if texture_path:
          material.albedo_img = o3d.io.read_image(texture_path)
      else:
          material.base_color = [0.7, 0.7, 0.7, 1.0]
      return mesh, material

    def create_floor(self):
        floor = o3d.geometry.TriangleMesh.create_box(width=self.floor_size, height=self.floor_size, depth=self.floor_depth)
        floor.translate([-(self.floor_size/2), -(self.floor_size/2), -0.01])  # Mover el piso para que esté centrado en el origen
        floor.paint_uniform_color([0.1, 0.1, 0.7])  # Pintar el piso de color azul
        material_floor = o3d.visualization.rendering.MaterialRecord()
        if os.path.exists('wood_floor_texture.png'):
            material_floor.albedo_img = o3d.io.read_image('wood_floor_texture.png')
        else:
            material_floor.base_color = [0.3, 0.3, 0.3, 1.0]
        return floor, material_floor

    def create_offrender_scene(self, img_W, img_H):
        if self.floort == True:
            render = o3d.visualization.rendering.OffscreenRenderer(width=img_W, height=img_H) #Linux only
            render.scene.add_geometry('mesh', self.mesh, self.mesh_material)
            render.scene.add_geometry('floor', self.floor, self.floor_material)
            return render
        else:
            render = o3d.visualization.rendering.OffscreenRenderer(width=img_W, height=img_H) #Linux only
            render.scene.add_geometry('mesh', self.mesh, self.mesh_material)
            return render
    
    def create_raycast_scene(self):
        if self.floort == True:
            mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            floor1 = o3d.t.geometry.TriangleMesh.from_legacy(self.floor)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh1)
            scene.add_triangles(floor1)
            return scene
        else: 
            mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh1)
            return scene

    def create_raycast_scene_pointcloud(self,img_W, img_H):
        scene = SimplePointCloudRaycast(self.mesh_file, width=img_W, height=img_H, fov=45, search_radius=0.1)
        return scene

    def get_scenes(self, img_W, img_H, onlyraycast=False):
        if onlyraycast == True:
            return self.create_raycast_scene_pointcloud(img_W, img_H)
        else:
            return self.create_offrender_scene(img_W, img_H), self.create_raycast_scene()