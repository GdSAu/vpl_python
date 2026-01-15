import numpy as np
import matplotlib.pyplot as plt


def plot_camera_with_target(T, target, axis_length=1.0):
    """
    T      : np.array (4,4) transform_matrix (Blender / NeRFstudio)
    target : np.array (3,) punto observado (look-at)
    """

    cam_pos = T[0:3, 3]

    # Ejes de la cámara desde la matriz
    x = T[0:3, 0]  # derecha
    y = T[0:3, 1]  # arriba
    z = T[0:3, 2]  # atrás (la cámara mira a -z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Ejes de la cámara
    ax.quiver(*cam_pos, *x, color='r', length=axis_length, normalize=True)
    ax.quiver(*cam_pos, *y, color='g', length=axis_length, normalize=True)
    ax.quiver(*cam_pos, *z, color='b', length=axis_length, normalize=True)

    # Cámara
    ax.scatter(*cam_pos, color='k', s=40)
    ax.text(*cam_pos, "Camera", color='k')

    # Punto de pose (look-at)
    ax.scatter(*target, color='m', s=60)
    ax.text(*target, "Target", color='m')

    # Vector de vista (cámara → target)
    view_vec = target - cam_pos
    ax.quiver(
        *cam_pos,
        *view_vec,
        color='m',
        linestyle='dashed',
        length=1.0,
        normalize=True
    )

    # Etiquetas
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    #ax.set_box_aspect([1,1,1])

    plt.show()


def plot_camera_from_transform(T, axis_length=1.0):
    """
    T : np.array shape (4,4) - transform_matrix (Blender / NeRFstudio)
    """

    cam_pos = T[0:3, 3]
    x = T[0:3, 0]
    y = T[0:3, 1]
    z = T[0:3, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(*cam_pos, *x, color='r', length=axis_length, normalize=True)
    ax.quiver(*cam_pos, *y, color='g', length=axis_length, normalize=True)
    ax.quiver(*cam_pos, *z, color='b', length=axis_length, normalize=True)

    ax.text(*(cam_pos + x), "x", color='r')
    ax.text(*(cam_pos + y), "y", color='g')
    ax.text(*(cam_pos + z), "z", color='b')

    ax.scatter(*cam_pos, color='k', s=30)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])

    plt.show()



def normalize(v):
    return v / np.linalg.norm(v)

def look_at(eye, target, up=np.array([0,1,0])):
    """
    Create a look-at view matrix.
    
    Parameters:
    eye (numpy.ndarray): The position of the camera.
    target (numpy.ndarray): The point the camera is looking at.
    up (numpy.ndarray): The up direction for the camera.
    
    Returns:
    numpy.ndarray: A 4x4 view matrix.
    """
    
    forward = normalize(target - eye)
    z = -forward                       # Blender: cámara mira a -Z
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)

    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T

def save_json(file_name,data,object_center, camera_angle_x=45):
    """
    Save camera poses and image paths to a JSON file in NeRFstudio format, one oject at the time.
    File_name : str nombre del archivo y direccion sin extension
    data      : list of tuples (img_name, cam_pos)
    camera_angle_x : float in degrees
    object_center : list of 3 floats
    Note: img_name should include relative path to images
    """
    import json

    frames = []

    for i, (img_name, cam_pos) in enumerate(data):
        T = look_at(cam_pos, object_center)

        frames.append({
            "file_path": f"{img_name}",
            "transform_matrix": T.tolist()
        })
    camera_angle_x = np.deg2rad(camera_angle_x) 
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }

    with open(file_name + ".json", "w") as f:
        json.dump(transforms, f, indent=4)