import open3d as o3d
import numpy as np
from utils.utils_o3d import scale_and_translate

class SimplePointCloudRaycast:
    def __init__(self, pcd_direction, width=640, height=480, fov=45, search_radius=0.01,needmesh=True):
        
        if needmesh == True: # we expect mesh direction
            self.__pc_from_mesh(pcd_direction)
        else: # we expect poincloud direction
            self.pcd = o3d.io.read_point_cloud(pcd_direction)
        
        self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        self.search_radius = search_radius
        self.points = np.asarray(self.pcd.points)
        self.width = width
        self.height = height
        self.fov = fov

    def __pc_from_mesh(self,direction):
        mesh = o3d.io.read_triangle_mesh(direction + '/meshes/model.obj', True)
        mesh = scale_and_translate(mesh)
        number_of_points = 16384 # according to PC-NBV
        mesh.compute_vertex_normals()
        self.pcd = mesh.sample_points_uniformly(number_of_points= number_of_points)

    def get_visible_points(self, center, eye, step_factor=1):
        """
        Obtiene solo los puntos visibles usando raycast
        
        Args:
            center: np.array([x, y, z]) - punto hacia donde mira la cámara
            eye: np.array([x, y, z]) - posición de la cámara
            width: ancho de la "imagen virtual" para densidad de rayos
            height: alto de la "imagen virtual" para densidad de rayos
            fov: campo de visión en grados
            step_factor: factor para reducir rayos (1=todos, 2=cada 2 píxeles, etc.)
            
        Returns:
            dict: información de puntos visibles
        """
        
        # Calcular vectores de la cámara
        forward = center - eye
        forward = forward / np.linalg.norm(forward)
        
        # Vector up (asumimos Y hacia arriba)
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Parámetros de la cámara
        aspect_ratio = self.width / self.height
        fov_rad = np.radians(self.fov)
        tan_half_fov = np.tan(fov_rad / 2)
        
        # Estructura para almacenar puntos visibles (sin colores)
        visible_points = {
            'indices': [],           # Índices de los puntos visibles
            'coordinates': [],       # Coordenadas 3D de los puntos
            'distances': [],         # Distancias desde la cámara
            'ray_directions': [],    # Direcciones de los rayos que los encontraron
        }
        
        rays_cast = 0
        hits_found = 0
        
        # Para cada píxel (con step_factor para optimización)
        for y in range(0, self.height, step_factor):
           
            for x in range(0, self.width, step_factor):
                rays_cast += 1
                
                # Convertir coordenadas de píxel a coordenadas normalizadas [-1, 1]
                ndc_x = (2 * x / self.width) - 1
                ndc_y = 1 - (2 * y / self.height)  # Invertir Y
                
                # Calcular dirección del rayo
                ray_dir = (ndc_x * aspect_ratio * tan_half_fov * right + 
                          ndc_y * tan_half_fov * up + 
                          forward)
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Hacer raycast
                hit_result = self._intersect_ray(eye, ray_dir)
                
                if hit_result['hit']:
                    hits_found += 1
                    point_idx = hit_result['point_idx']
                    
                    # Evitar duplicados (un punto puede ser visible desde múltiples rayos)
                    if point_idx not in visible_points['indices']:
                        visible_points['indices'].append(point_idx)
                        visible_points['coordinates'].append(self.points[point_idx])
                        visible_points['distances'].append(hit_result['distance'])
                        visible_points['ray_directions'].append(ray_dir.copy())
        
        visible_points['indices'] = np.array(visible_points['indices'])
        visible_points['coordinates'] = np.array(visible_points['coordinates'])
        visible_points['distances'] = np.array(visible_points['distances'])
        visible_points['ray_directions'] = np.array(visible_points['ray_directions'])
        
        return visible_points
    
    
    def _intersect_ray(self, origin, direction, max_distance=20.0):
        """Encuentra intersección de rayo con nube de puntos"""
        step_size = self.search_radius / 3
        num_steps = int(max_distance / step_size)
        
        for i in range(num_steps):
            sample_point = origin + i * step_size * direction
            
            # Buscar puntos cercanos
            [k, idx, distances] = self.kdtree.search_radius_vector_3d(
                sample_point, self.search_radius)
            
            if k > 0:
                # Encontró intersección
                closest_idx = idx[np.argmin(distances)]
                return {
                    'hit': True,
                    'distance': i * step_size,
                    'point_idx': closest_idx,
                    'point': self.points[closest_idx],
                    'hit_point': sample_point
                }
        
        return {'hit': False, 'distance': np.inf}
    
    def create_visible_pointcloud(self, center, eye, **kwargs):
        """
        Crea una nueva nube de puntos con solo los puntos visibles (sin colores)
        """
        
        visible_points = self.get_visible_points(center, eye, **kwargs)
        visible_pcd = o3d.geometry.PointCloud()
        
        if len(visible_points['coordinates']) > 0:
            visible_pcd.points = o3d.utility.Vector3dVector(visible_points['coordinates'])

        return visible_pcd