import numpy as np

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