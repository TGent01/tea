import cv2
import os
import csv
import numpy as np

INPUT_DIR = "Dataset_TEA"
OUTPUT_DIR = "Frames_TEA"
LABELS_CSV = "labels_TEA.csv"   # no, categoria, video_id, url, duration_s, start_time, end_time, start_sec, end_sec
SEQUENCE_LENGTH = 40
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Margen opcional (en segundos) para tomar un poco de contexto antes/despues
# del intervalo etiquetado. Dejar en 0 si quieres usar exactamente el
# intervalo anotado en el CSV.
MARGIN_SEC = 0.0


def cargar_etiquetas(ruta_csv):
    """Carga el CSV de etiquetas y devuelve un dict: video_id -> (start_sec, end_sec)."""
    etiquetas = {}
    with open(ruta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            video_id = fila["video_id"].strip()
            start_sec = float(fila["start_sec"])
            end_sec = float(fila["end_sec"])
            etiquetas[video_id] = (start_sec, end_sec)
    return etiquetas


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

etiquetas = cargar_etiquetas(LABELS_CSV)
print(f"Etiquetas cargadas: {len(etiquetas)} videos con ventana de comportamiento.")

categorias = os.listdir(INPUT_DIR)

for categoria in categorias:
    ruta_categoria = os.path.join(INPUT_DIR, categoria)

    if not os.path.isdir(ruta_categoria):
        continue

    print(f"--- Procesando categoría: {categoria} ---")

    ruta_salida_categoria = os.path.join(OUTPUT_DIR, categoria)
    if not os.path.exists(ruta_salida_categoria):
        os.makedirs(ruta_salida_categoria)

    videos = os.listdir(ruta_categoria)

    for video in videos:
        ruta_video = os.path.join(ruta_categoria, video)
        nombre_video = video.split(".")[0]

        ruta_salida_video = os.path.join(ruta_salida_categoria, nombre_video)
        if not os.path.exists(ruta_salida_video):
            os.makedirs(ruta_salida_video)

        cap = cv2.VideoCapture(ruta_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # --- Determinar el rango de frames a muestrear ---
        if nombre_video in etiquetas and fps and fps > 0:
            start_sec, end_sec = etiquetas[nombre_video]
            start_sec = max(0.0, start_sec - MARGIN_SEC)
            end_sec = end_sec + MARGIN_SEC

            frame_inicio = int(round(start_sec * fps))
            frame_fin = int(round(end_sec * fps))

            # Asegurar límites válidos dentro del video
            frame_inicio = max(0, min(frame_inicio, total_frames - 1))
            frame_fin = max(frame_inicio, min(frame_fin, total_frames - 1))

            print(f"  {video}: usando ventana {start_sec:.1f}s-{end_sec:.1f}s "
                  f"(frames {frame_inicio}-{frame_fin} de {total_frames}, fps={fps:.2f})")
        else:
            # Fallback: si el video no tiene etiqueta o no se pudo leer el fps,
            # se usa el video completo como antes.
            frame_inicio = 0
            frame_fin = max(0, total_frames - 1)
            print(f"  {video}: sin etiqueta en {LABELS_CSV} (o fps inválido), "
                  f"usando el video completo.")

        intervalos = np.linspace(frame_inicio, frame_fin, SEQUENCE_LENGTH, dtype=int)

        frames_guardados = []  # guarda los arrays ya redimensionados, en orden
        for frame_idx in intervalos:
            frame_resized = None

            # Intentar leer el frame exacto y, si falla, retroceder unos
            # pocos frames (hasta 5) antes de darlo por perdido. Esto cubre
            # el caso típico de que el último frame de una ventana corta no
            # se pueda decodificar exactamente.
            for offset in range(0, 6):
                idx_intento = max(0, int(frame_idx) - offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_intento)
                ret, frame = cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    break

            if frame_resized is not None:
                frames_guardados.append(frame_resized)
            elif frames_guardados:
                # Ni con reintentos se pudo leer: repetir el último frame
                # válido para no perder la posición en la secuencia.
                frames_guardados.append(frames_guardados[-1])

        # Si aun así faltan frames al final (p.ej. el primer frame del video
        # tampoco se pudo leer), rellenar repitiendo el último disponible.
        while frames_guardados and len(frames_guardados) < SEQUENCE_LENGTH:
            frames_guardados.append(frames_guardados[-1])

        for frame_count, frame_resized in enumerate(frames_guardados):
            nombre_frame = f"frame_{frame_count:02d}.jpg"
            ruta_frame = os.path.join(ruta_salida_video, nombre_frame)
            cv2.imwrite(ruta_frame, frame_resized)

        cap.release()
        if len(frames_guardados) < SEQUENCE_LENGTH:
            print(f"Video {video}: ¡ADVERTENCIA! solo se pudieron obtener "
                  f"{len(frames_guardados)} frames (no se pudo leer ningún frame de la ventana).")
        else:
            print(f"Video {video} procesado: {len(frames_guardados)} frames extraídos.")
print("¡Proceso de extracción finalizado con éxito!")
