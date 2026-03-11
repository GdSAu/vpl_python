import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_random_rgb_colors(n):
    np.random.seed(0)
    return np.random.rand(n, 3)

def get_arrow_to_target(origin, target, length_scale=.1):
    origin = np.array(origin)
    direction = np.array(target) - origin
    return origin, direction * length_scale

def get_nbv_plot(id_objeto, csv_dir, l=15):
    """
    Para cada grupo de 15 vistas del objeto, genera un plot 3D con etiquetas reiniciadas.
    Los vectores apuntan al punto (0, 0, 0.2).
    """
    # Leer CSV y filtrar por objeto
    nbv = pd.read_csv(csv_dir)
    nbv_filtrado = nbv[nbv['id_objeto'] == id_objeto].copy()

    # Convertir columna 'nbv' a vectores
    nbv_filtrado["nbv"] = nbv_filtrado["nbv"].apply(
        lambda s: np.fromstring(s.strip('[]'), sep=' ')
    )

    target = np.array([0, 0, 0.2])
    total = len(nbv_filtrado)
    bloques = (total + l - 1) // l  # redondeo hacia arriba
    angulos = [9,11.5,15,22.5,45]
    for b in range(bloques):
        inicio = b * l
        fin = min((b + 1) * l, total)
        bloque_actual = nbv_filtrado.iloc[inicio:fin].reset_index(drop=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = generate_random_rgb_colors(len(bloque_actual))
        all_points = []

        for i in range(len(bloque_actual)):
            vista = bloque_actual['nbv'].values[i]
            origin, direction = get_arrow_to_target(vista, target, length_scale=0.1)

            all_points.append(origin)
            all_points.append(origin + direction)

            ax.quiver(
                origin[0], origin[1], origin[2],
                direction[0], direction[1], direction[2],
                color=colors[i], linewidth=1.5, arrow_length_ratio=0.3
            )
            label_pos = origin + direction * 1.1
            ax.text(label_pos[0], label_pos[1], label_pos[2], str(i), color='black', fontsize=10)

        # Punto objetivo
        ax.scatter([target[0]], [target[1]], [target[2]], color='red', s=50, label='Target (0,0,0.2)')
        ax.legend()

        # Ajustar límites
        all_points = np.array(all_points)
        min_bound = all_points.min(axis=0) - 0.1
        max_bound = all_points.max(axis=0) + 0.1
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        ax.set_zlim(min_bound[2], max_bound[2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('{} — Angulos {}°'.format(id_objeto,angulos[b]))
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()

def get_nbv_plot_merged( id_objeto, csv_dir, l=15):
    """
    Divide las vistas en bloques de 15 y los muestra en subplots 3D en una sola figura.
    """
    nbv = pd.read_csv(csv_dir)
    nbv_filtrado = nbv[nbv['id_objeto'] == id_objeto].copy()
    nbv_filtrado["nbv"] = nbv_filtrado["nbv"].apply(
        lambda s: np.fromstring(s.strip('[]'), sep=' ')
    )

    target = np.array([0, 0, 0.2])
    total = len(nbv_filtrado)
    bloques = (total + l - 1) // l

    # Calcular tamaño de cuadrícula
    cols = min(bloques, 3)
    rows = (bloques + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    angulos = [9,11.5,15,22.5,45]
    for b in range(bloques):
        inicio = b * l
        fin = min((b + 1) * l, total)
        bloque_actual = nbv_filtrado.iloc[inicio:fin].reset_index(drop=True)
        colors = generate_random_rgb_colors(len(bloque_actual))

        ax = fig.add_subplot(rows, cols, b + 1, projection='3d')
        all_points = []

        for i in range(len(bloque_actual)):
            vista = bloque_actual['nbv'].values[i]
            origin, direction = get_arrow_to_target(vista, target, length_scale=0.1)
            all_points.append(origin)
            all_points.append(origin + direction)

            ax.quiver(
                origin[0], origin[1], origin[2],
                direction[0], direction[1], direction[2],
                color=colors[i], linewidth=1.5, arrow_length_ratio=0.3
            )

            label_pos = origin + direction * 1.1
            ax.text(label_pos[0], label_pos[1], label_pos[2], str(i), color='black', fontsize=8)

        # Punto objetivo
        ax.scatter([target[0]], [target[1]], [target[2]], color='red', s=50)

        # Ajustar límites
        all_points = np.array(all_points)
        min_bound = all_points.min(axis=0) - 0.1
        max_bound = all_points.max(axis=0) + 0.1
        ax.set_xlim(min_bound[0], max_bound[0])
        ax.set_ylim(min_bound[1], max_bound[1])
        ax.set_zlim(min_bound[2], max_bound[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('{} — Angulos {}°'.format(id_objeto,angulos[b]), fontsize=10)
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()

