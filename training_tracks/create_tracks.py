import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import os
import csv


def create_centerline_from_waypoints(waypoints, track_dir, track_name, resolution, track_width_px=60):
    """
    Crea un archivo centerline.csv a partir de waypoints.

    Args:
        waypoints: Lista de puntos (x, y) en píxeles
        track_dir: Directorio de la pista
        track_name: Nombre de la pista
        resolution: Resolución metros/píxel
        track_width_px: Ancho de la pista en píxeles
    """
    # Convertir píxeles a metros (centrar en origen)
    centerline_points = []

    # Obtener dimensiones de la imagen de waypoints para centrar
    if waypoints:
        # Encontrar límites de los waypoints
        x_coords = [wp[0] for wp in waypoints[:-1]]  # Excluir último punto duplicado
        y_coords = [wp[1] for wp in waypoints[:-1]]

        # Centrar usando los límites de los waypoints
        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2

    # Convertir ancho de pista a metros
    track_width_m = track_width_px * resolution
    half_width_m = track_width_m / 2

    for i, (x_px, y_px) in enumerate(waypoints[:-1]):  # Excluir último punto duplicado
        # Convertir píxeles a metros (centrar en origen)
        x_m = (x_px - x_center) * resolution
        y_m = (y_px - y_center) * resolution

        # Agregar punto con coordenadas x,y y anchuras izquierda/derecha (sin s_m)
        centerline_points.append([x_m, y_m, half_width_m, half_width_m])

    # Guardar centerline.csv
    centerline_file = f"{track_dir}/{track_name}_centerline.csv"
    with open(centerline_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['# x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])  # Header sin s_m
        for point in centerline_points:
            writer.writerow([f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}", f"{point[3]:.6f}"])

    return centerline_file


def create_numpy_map(img, track_dir, track_name):
    """Crea archivo .npy a partir de la imagen."""
    # Convertir imagen a formato numpy (0 = obstáculo, 1 = libre)
    np_map = (img > 128).astype(np.float64)

    # Guardar como .npy
    npy_file = f"{track_dir}/{track_name}_map.npy"
    np.save(npy_file, np_map)

    return npy_file


def create_custom_track(track_name="custom_track", track_type="oval"):
    """
    Genera una nueva pista para F1TENTH gym.

    Args:
        track_name: Nombre de la pista
        track_type: Tipo de pista ('oval', 'figure8', 'straight', 'custom')
    """

    # Crear directorio de la pista
    track_dir = f"./tracks/{track_name}"
    os.makedirs(track_dir, exist_ok=True)

    if track_type == "oval":
        create_oval_track(track_dir, track_name)
    elif track_type == "figure8":
        create_figure8_track(track_dir, track_name)
    elif track_type == "straight":
        create_straight_track(track_dir, track_name)
    else:
        create_custom_geometric_track(track_dir, track_name)


def create_straight_track(track_dir, track_name):
    """Crea una pista semi rectangular con curvas alargadas de 115°"""

    width = 1000  # píxeles
    height = 600  # píxeles
    track_width = 60  # ancho de la pista en píxeles

    # Crear imagen en blanco (negro = obstáculo, blanco = pista)
    img = np.zeros((height, width), dtype=np.uint8)

    # Parámetros de la pista semi rectangular
    margin = 80  # Margen desde los bordes

    # Radio de curvatura para las curvas de 115°
    curve_radius = 120

    # Definir waypoints para la pista semi rectangular con curvas alargadas
    waypoints = []

    # Recta superior (izquierda a derecha)
    for x in range(margin, width - margin, 20):
        waypoints.append((x, margin + 50))

    # Curva derecha (115° aproximadamente) - más alargada
    import math
    angle_start = 0  # 0°
    angle_end = math.radians(115)  # 115° en radianes
    center_x = width - margin - curve_radius
    center_y = margin + 50 + curve_radius

    # Puntos de la curva derecha
    num_curve_points = 25
    for i in range(num_curve_points):
        angle = angle_start + (angle_end - angle_start) * i / (num_curve_points - 1)
        x = center_x + curve_radius * math.cos(angle)
        y = center_y + curve_radius * math.sin(angle)
        waypoints.append((int(x), int(y)))

    # Recta inferior (derecha a izquierda)
    for x in range(width - margin, margin, -20):
        waypoints.append((x, height - margin - 50))

    # Curva izquierda (115° aproximadamente) - más alargada
    center_x = margin + curve_radius
    center_y = height - margin - 50 - curve_radius
    angle_start = math.pi  # 180°
    angle_end = math.pi + math.radians(115)  # 180° + 115°

    # Puntos de la curva izquierda
    for i in range(num_curve_points):
        angle = angle_start + (angle_end - angle_start) * i / (num_curve_points - 1)
        x = center_x + curve_radius * math.cos(angle)
        y = center_y + curve_radius * math.sin(angle)
        waypoints.append((int(x), int(y)))

    # Cerrar el circuito conectando al primer punto
    waypoints.append(waypoints[0])

    # Crear la pista siguiendo los waypoints
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]

        # Crear segmento de pista entre waypoints
        create_track_segment(img, x1, y1, x2, y2, track_width)

    # Suavizar las transiciones entre segmentos
    from scipy import ndimage
    img = ndimage.gaussian_filter(img.astype(float), sigma=1.5)
    img = (img > 128).astype(np.uint8) * 255

    # Añadir bordes más definidos
    # Crear una máscara para reforzar los bordes de la pista
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = ndimage.binary_dilation(img > 0, kernel).astype(np.uint8) * 255
    img = img_dilated    # Guardar imagen
    Image.fromarray(img).save(f"{track_dir}/{track_name}_map.png")

    # Crear archivo YAML de configuración con centerline y numpy map
    create_track_yaml(track_dir, track_name, width, height, 0.04,
                      waypoints=waypoints, img=img, track_width_px=track_width)

    print(f"✅ Pista semi rectangular con curvas de 115° '{track_name}' creada en {track_dir}")


def create_oval_track(track_dir, track_name):
    """Crea una pista ovalada"""

    # Parámetros de la pista
    width = 800  # píxeles
    height = 600  # píxeles
    track_width = 60  # ancho de la pista en píxeles

    # Crear imagen en blanco (negro = obstáculo, blanco = pista)
    img = np.zeros((height, width), dtype=np.uint8)

    # Centro de la pista
    center_x, center_y = width // 2, height // 2

    # Radios del óvalo
    outer_radius_x = 300
    outer_radius_y = 200
    inner_radius_x = outer_radius_x - track_width
    inner_radius_y = outer_radius_y - track_width

    # Crear grid de coordenadas
    y, x = np.ogrid[:height, :width]

    # Ecuación del óvalo exterior e interior
    outer_oval = ((x - center_x) / outer_radius_x) ** 2 + ((y - center_y) / outer_radius_y) ** 2
    inner_oval = ((x - center_x) / inner_radius_x) ** 2 + ((y - center_y) / inner_radius_y) ** 2

    # La pista está entre los dos óvalos
    track_mask = (outer_oval <= 1) & (inner_oval >= 1)
    img[track_mask] = 255  # Blanco para la pista

    # Crear waypoints para la línea central del óvalo
    waypoints = []
    import math
    center_radius_x = (outer_radius_x + inner_radius_x) / 2
    center_radius_y = (outer_radius_y + inner_radius_y) / 2

    # Generar puntos a lo largo del óvalo central
    num_points = 100
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = center_x + center_radius_x * math.cos(angle)
        y = center_y + center_radius_y * math.sin(angle)
        waypoints.append((int(x), int(y)))

    # Cerrar el circuito
    waypoints.append(waypoints[0])

    # Guardar imagen
    Image.fromarray(img).save(f"{track_dir}/{track_name}_map.png")

    # Crear archivo YAML de configuración con waypoints y mapa numpy
    create_track_yaml(track_dir, track_name, width, height, 0.05,
                      waypoints=waypoints, img=img, track_width_px=track_width)

    print(f"✅ Pista oval '{track_name}' creada en {track_dir}")


def create_figure8_track(track_dir, track_name):
    """Crea una pista en forma de 8"""

    width = 800
    height = 600
    track_width = 40

    img = np.zeros((height, width), dtype=np.uint8)

    # Crear dos círculos que se intersectan
    center1_x, center1_y = width // 3, height // 3
    center2_x, center2_y = 2 * width // 3, 2 * height // 3
    radius = 150

    y, x = np.ogrid[:height, :width]

    # Círculo superior
    circle1_outer = (x - center1_x) ** 2 + (y - center1_y) ** 2
    circle1_inner = (x - center1_x) ** 2 + (y - center1_y) ** 2

    # Círculo inferior
    circle2_outer = (x - center2_x) ** 2 + (y - center2_y) ** 2
    circle2_inner = (x - center2_x) ** 2 + (y - center2_y) ** 2

    # Pista del primer círculo
    track1 = (circle1_outer <= radius ** 2) & (circle1_inner >= (radius - track_width) ** 2)

    # Pista del segundo círculo
    track2 = (circle2_outer <= radius ** 2) & (circle2_inner >= (radius - track_width) ** 2)

    # Conexión entre círculos (intersección)
    connection = track1 | track2

    img[connection] = 255

    # Crear waypoints para la figura 8
    waypoints = []
    import math

    # Radio de la línea central
    center_radius = radius - track_width // 2

    # Primera mitad del 8 (círculo superior)
    num_points_per_circle = 50
    for i in range(num_points_per_circle):
        angle = 2 * math.pi * i / num_points_per_circle
        x = center1_x + center_radius * math.cos(angle)
        y = center1_y + center_radius * math.sin(angle)
        waypoints.append((int(x), int(y)))

    # Segunda mitad del 8 (círculo inferior)
    for i in range(num_points_per_circle):
        angle = 2 * math.pi * i / num_points_per_circle
        x = center2_x + center_radius * math.cos(angle)
        y = center2_y + center_radius * math.sin(angle)
        waypoints.append((int(x), int(y)))

    # Cerrar el circuito
    waypoints.append(waypoints[0])

    # Guardar
    Image.fromarray(img).save(f"{track_dir}/{track_name}_map.png")
    create_track_yaml(track_dir, track_name, width, height, 0.05,
                      waypoints=waypoints, img=img, track_width_px=track_width)

    print(f"✅ Pista figura-8 '{track_name}' creada en {track_dir}")


def create_custom_geometric_track(track_dir, track_name):
    """Crea una pista con geometría personalizada usando waypoints"""

    width = 1000
    height = 800
    track_width = 50

    # Definir waypoints de la línea central
    waypoints = [
        (100, 400),   # Start
        (300, 200),   # Curva 1
        (500, 150),   # Recta
        (700, 300),   # Curva 2
        (850, 500),   # Curva 3
        (700, 650),   # Curva 4
        (400, 700),   # Recta inferior
        (200, 600),   # Curva 5
        (100, 400),   # Vuelta al inicio
    ]

    img = np.zeros((height, width), dtype=np.uint8)

    # Crear pista siguiendo los waypoints
    for i in range(len(waypoints) - 1):
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]

        # Crear línea entre waypoints
        create_track_segment(img, x1, y1, x2, y2, track_width)

    # Suavizar curvas
    from scipy import ndimage
    img = ndimage.gaussian_filter(img.astype(float), sigma=2)
    img = (img > 128).astype(np.uint8) * 255

    # Guardar
    Image.fromarray(img).save(f"{track_dir}/{track_name}_map.png")
    create_track_yaml(track_dir, track_name, width, height, 0.04,
                      waypoints=waypoints, img=img, track_width_px=track_width)

    print(f"✅ Pista personalizada '{track_name}' creada en {track_dir}")


def create_track_segment(img, x1, y1, x2, y2, track_width):
    """Crea un segmento de pista entre dos puntos"""

    # Calcular la dirección
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length == 0:
        return

    # Vector unitario perpendicular
    perp_x = -dy / length
    perp_y = dx / length

    # Crear polígono para el segmento
    half_width = track_width // 2

    # Coordenadas de los 4 vértices del rectángulo
    vertices = [
        (int(x1 + perp_x * half_width), int(y1 + perp_y * half_width)),
        (int(x1 - perp_x * half_width), int(y1 - perp_y * half_width)),
        (int(x2 - perp_x * half_width), int(y2 - perp_y * half_width)),
        (int(x2 + perp_x * half_width), int(y2 + perp_y * half_width)),
    ]

    # Rellenar el polígono
    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.polygon(vertices, fill=255)

    # Convertir de vuelta a numpy
    img[:] = np.array(pil_img)


def create_track_yaml(track_dir, track_name, width, height, resolution, waypoints=None, img=None, track_width_px=60):
    """Crea el archivo YAML de configuración de la pista y archivos adicionales."""

    config = {
        'image': f'{track_name}_map.png',
        'resolution': resolution,  # metros por píxel
        'origin': [-width * resolution / 2, -height * resolution / 2, 0.0],
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
        'negate': 0
    }

    with open(f"{track_dir}/{track_name}_map.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Crear centerline si se proporcionan waypoints
    if waypoints:
        create_centerline_from_waypoints(waypoints, track_dir, track_name, resolution, track_width_px)
        print(f"  📍 Centerline created: {track_name}_centerline.csv")

    # Crear archivo numpy si se proporciona imagen
    if img is not None:
        create_numpy_map(img, track_dir, track_name)
        print(f"  💾 Numpy map created: {track_name}_map.npy")

# Función principal para generar diferentes tipos de pistas


def generate_track_set():
    """Genera un conjunto de pistas de diferentes tipos"""

    print("🏁 Generando conjunto de pistas personalizadas...")

    # Crear diferentes tipos de pistas
    create_custom_track("oval_small", "oval")
    create_custom_track("figure8_track", "figure8")
    create_custom_track("semi_rectangular", "straight")
    create_custom_track("complex_circuit", "custom")

    print("✅ Todas las pistas generadas correctamente!")
    print("📁 Las pistas están disponibles en el directorio ./tracks/")


if __name__ == "__main__":
    generate_track_set()
