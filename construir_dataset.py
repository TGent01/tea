import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



FRAMES_DIR = "Frames_TEA"
SEQUENCE_LENGTH = 40
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Orden fijo de clases: importante para que el mapeo índice -> clase
# sea siempre el mismo entre distintas ejecuciones.
CLASES = sorted(
    d for d in os.listdir(FRAMES_DIR)
    if os.path.isdir(os.path.join(FRAMES_DIR, d))
)
print(f"Clases encontradas: {CLASES}")


def cargar_secuencia(ruta_video_frames):
    """Carga los SEQUENCE_LENGTH frames de una carpeta de video como un array
    (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3) normalizado en [0, 1]."""
    archivos = sorted(os.listdir(ruta_video_frames))  # frame_00.jpg, frame_01.jpg, ...
    frames = []
    for archivo in archivos:
        ruta_frame = os.path.join(ruta_video_frames, archivo)
        img = cv2.imread(ruta_frame)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        frames.append(img)

    if len(frames) != SEQUENCE_LENGTH:
        # Video con menos frames de los esperados (video muy corto, error de
        # extracción, etc.). Se descarta para no romper la forma del tensor.
        return None

    return np.array(frames, dtype=np.float32)


def construir_dataset():
    features = []
    labels = []
    videos_descartados = []

    for idx_clase, categoria in enumerate(CLASES):
        ruta_categoria = os.path.join(FRAMES_DIR, categoria)
        videos = sorted(os.listdir(ruta_categoria))

        for video in videos:
            ruta_video_frames = os.path.join(ruta_categoria, video)
            if not os.path.isdir(ruta_video_frames):
                continue

            secuencia = cargar_secuencia(ruta_video_frames)
            if secuencia is None:
                videos_descartados.append(f"{categoria}/{video}")
                continue

            features.append(secuencia)
            labels.append(idx_clase)

    if videos_descartados:
        print(f"\nAVISO: se descartaron {len(videos_descartados)} videos "
              f"por no tener {SEQUENCE_LENGTH} frames:")
        for v in videos_descartados:
            print(f"  - {v}")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print(f"\nDataset construido: X={X.shape}, y={y.shape}")
    for idx_clase, categoria in enumerate(CLASES):
        print(f"  {categoria}: {(y == idx_clase).sum()} videos")

    return X, y


if __name__ == "__main__":
    X, y = construir_dataset()

    y_onehot = to_categorical(y, num_classes=len(CLASES))

    # stratify=y (no y_onehot) para mantener la proporción de clases en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=0.25,
        random_state=42,
        stratify=y,
        shuffle=True,
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    os.makedirs("dataset_listo", exist_ok=True)
    np.save("dataset_listo/X_train.npy", X_train)
    np.save("dataset_listo/X_test.npy", X_test)
    np.save("dataset_listo/y_train.npy", y_train)
    np.save("dataset_listo/y_test.npy", y_test)
    with open("dataset_listo/clases.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(CLASES))

    print("\nGuardado en dataset_listo/ (X_train.npy, X_test.npy, y_train.npy, "
          "y_test.npy, clases.txt)")
