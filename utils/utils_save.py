import numpy as np
import open3d as o3d
import json
import os
from .utils_o3d import camara

def save_camera_trayectory(direccion, eyes, cent, up, fov, width, height, method_name:str, iteration_name:str):
    # 1. Preparar la parte de Open3D (mantenemos compatibilidad)
    intrinsic_np = camara.calcular_matriz_intrinseca(fov, width, height)
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_np)
    
    # FOV en radianes para el formato Synthetic
    camera_angle_x = np.deg2rad(fov)
    frames_nerf = []
    params_list = []
    
    for i, eye_pos in enumerate(eyes):
        # A. Extrínseca de tu clase (Mundo a Cámara - OpenCV)
        ext = camara.calcular_matriz_extrinsecas(eye_pos, cent, up)
        
        # Guardar para Open3D
        p = o3d.camera.PinholeCameraParameters()
        p.extrinsic = ext
        p.intrinsic = cam_intrinsic
        params_list.append(p)
        
        # B. Para NeRF Blender/Synthetic (Cámara a Mundo - OpenGL)
        c2w = np.linalg.inv(ext)
        
        # Inversión de ejes Y y Z (OpenCV -> OpenGL)
        c2w_blender = c2w.copy()
        c2w_blender[0:3, 1:3] *= -1 
        
        frames_nerf.append({
            "file_path": f"./{iteration_name}RGB_{i}", # Siguiendo tu formato r_0, r_1...
            "rotation": 0.0, # Valor por defecto si no tienes rotación de cámara extra
            "transform_matrix": c2w_blender.tolist()
        })
        
    # 2. Guardar JSON en formato Synthetic de Blender
    transforms = {
        "camera_angle_x": float(camera_angle_x),
        "frames": frames_nerf
    }
    
    # Guardar ambos archivos
    ruta_carpetas = os.path.join(direccion, "RGB", method_name)
    iteration_name = iteration_name.rstrip('/')
    ruta_final = os.path.join(ruta_carpetas, f"{iteration_name}_transforms_nerf.json")
    print(ruta_final)
    trayectoria_o3d = o3d.camera.PinholeCameraTrajectory()
    trayectoria_o3d.parameters = params_list
    o3d.io.write_pinhole_camera_trajectory((os.path.join(ruta_carpetas, f"{iteration_name}_trayectory_camara.json")), trayectoria_o3d)
        
    try:
        with open(ruta_final, "w") as f:
            json.dump(transforms, f, indent=4)
        print(f"Successfully saved camera trajectory")
    except Exception as e:
        print(f"Error: {e}")

def save_nerfstudio_transforms(direccion, eyes, cent, up, fov, method_name: str, iteration_name: str):
    """Same extrinsics math as save_camera_trayectory (OpenCV world->camera,
    inverted + Y/Z-flipped into Blender/NeRF-synthetic camera->world), but
    written directly in the layout NerfStudio's BlenderDataParser expects:
    transforms_{train,val,test}.json living next to the RGB_{i}.png frames,
    with extension-less relative file_path. Avoids the extra conversion step
    (NBV_implicit's nerf_converter.py) that today's save_camera_trayectory needs.
    """
    camera_angle_x = np.deg2rad(fov)
    frames = []
    for i, eye_pos in enumerate(eyes):
        ext = camara.calcular_matriz_extrinsecas(eye_pos, cent, up)
        c2w = np.linalg.inv(ext)
        c2w_blender = c2w.copy()
        c2w_blender[0:3, 1:3] *= -1
        frames.append({
            "file_path": "./RGB_{}".format(i),
            "rotation": 0.0,
            "transform_matrix": c2w_blender.tolist()
        })

    transforms = {"camera_angle_x": float(camera_angle_x), "frames": frames}

    ruta_carpeta = os.path.join(direccion, "RGB", method_name.strip('/'), iteration_name.strip('/'))
    os.makedirs(ruta_carpeta, exist_ok=True)
    for split in ("train", "val", "test"):
        with open(os.path.join(ruta_carpeta, "transforms_{}.json".format(split)), "w") as f:
            json.dump(transforms, f, indent=4)
    return ruta_carpeta


def GuardarDS(ds,I,i,obj_name,v_ini,v, itera,cd, distance,cov):
    """
    ds: objeto para almacenar los datos
    I: ID de secuencia
    i:  iteración del objeto
    obj_name: nombre del objeto
    v_ini: pose inicial de proceso
    v: nbv
    max_inc: Incremento de la iteración
    P_acu: nube de puntos acumulada
    octree: Octree 
    occupancy_probs: probabilidades de la rejilla para almacenar en .npy
    dir_carpeta: Direccion a la carpeta del objeto
    itera: dirección de la carpeta de iteración
    Siter: numero de vistas por iteración
    cov: Metrica de cobertura
    """
    #Agregamos al DS los apuntadores
    ds["ID"].append(I)
    ds["id_objeto"].append(obj_name)
    ds["iteracion_objeto"].append(i)
    ds["pose_inicial"].append(v_ini)
    ds["nube_puntos"].append("/Point_cloud/"+ itera +"cloud_{}.pcd".format(i))
    ds["rejilla"].append("/Octree/"+ itera +"octree_{}.ot".format(i))
    ds["nbv"].append(v)
    if i == 0:
        ds["id_anterior"].append(None)
    else:
        ds["id_anterior"].append(I-1)
    if i < 10:
        ds["id_siguiente"].append(I+1)
    else:
        ds["id_siguiente"].append(None)
    ds["chamfer"].append(cd)
    ds["ganancia_cobertura"].append(distance)
    ds["cobertura"].append(cov)