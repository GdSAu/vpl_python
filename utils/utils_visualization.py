import open3d as o3d
import numpy as np
from .utils_o3d import scale_and_translate

class Visualizar():
    def __init__(self, dir_carpeta):
        self.dir_carpeta = dir_carpeta if dir_carpeta.endswith('/') else dir_carpeta + '/'
        self.geometrias = [] # Lista para guardar todo lo que queremos ver

    def read_model(self):
        # Cargar malla
        self.mesh = o3d.io.read_triangle_mesh(self.dir_carpeta + 'meshes/model.obj', True)
        self.mesh.compute_vertex_normals() # Para que se vea bien sin materiales pro
        
        # Aplicar transformación
        self.mesh = scale_and_translate(self.mesh, scale_factor=0.39)
        self.geometrias.append(self.mesh)

    def add_cameras(self):
        param = o3d.io.read_pinhole_camera_trajectory(self.dir_carpeta + "trayectoria_camara.json")
        for i in range(len(param.parameters)):
            extrinsic = param.parameters[i].extrinsic
            intrinsic = param.parameters[i].intrinsic.intrinsic_matrix
            width = param.parameters[i].intrinsic.width
            height = param.parameters[i].intrinsic.height
            
            # Obtener partes de la cámara (lista de geometrías)
            camera_parts = draw_camera(intrinsic, extrinsic, width, height)
            
            # En el visualizador clásico, simplemente las añadimos a la lista
            if isinstance(camera_parts, list):
                self.geometrias.extend(camera_parts)
            else:
                self.geometrias.append(camera_parts)

    def show(self):
        o3d.visualization.draw_geometries(self.geometrias, 
                                          window_name="Visualización Poses",
                                          width=1024, height=768)

def runVisualization(ruta):
    vis = Visualizar(ruta)
    vis.read_model()
    vis.add_cameras()
    vis.show()


def draw_camera(I, E, w, h, scale=0.1, color=[0.8, 0.2, 0.8]):
    C_inv = np.linalg.inv(I)
    pix_points = np.array([[0,0,0], [0,0,1], [w,0,1], [0,h,1], [w,h,1]])
    points_cam = [(C_inv @ p) * scale for p in pix_points]
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.25)
    
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_cam),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color(color)
    E_inv = np.linalg.inv(E)
    
    axis.transform(E_inv)
    line_set.transform(E_inv)

    return [axis, line_set]

