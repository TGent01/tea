import numpy as np
from keras.models import Model
from keras.layers import Input, TimeDistributed, Resizing, Rescaling, Lambda
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

SEQUENCE_LENGTH = 40
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CANALES = 3
RESNET_INPUT_SIZE = 224


def crear_extractor_resnet_congelado():
    """Solo la parte SIN pesos entrenables: resize + preprocesamiento +
    ResNet50 congelado. No incluye la capa de proyección (esa sí tiene
    pesos entrenables y se entrena después, rápido, sobre estas
    características ya calculadas)."""
    entrada = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES))
    x = TimeDistributed(Resizing(RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))(entrada)
    x = TimeDistributed(Rescaling(255.0))(x)
    x = Lambda(preprocess_input, output_shape=lambda s: s)(x)

    resnet_base = ResNet50(include_top=False, weights="imagenet", pooling="avg")
    resnet_base.trainable = False

    salida = TimeDistributed(resnet_base)(x)  # (SEQUENCE_LENGTH, 2048)
    return Model(inputs=entrada, outputs=salida, name="ResNet50_congelado")


def main():
    print("Cargando ResNet50 preentrenado (puede tardar la primera vez, descarga los pesos)...")
    extractor = crear_extractor_resnet_congelado()

    for split in ["train", "test"]:
        print(f"\nProcesando split '{split}'...")
        X = np.load(f"dataset_listo/X_{split}.npy")
        print(f"  Entrada: {X.shape}")

        # Procesar de a pocos videos por vez para no acumular todo en memoria
        features = []
        LOTE = 4
        for i in range(0, len(X), LOTE):
            lote = X[i:i + LOTE]
            feats = extractor.predict(lote, verbose=0)
            features.append(feats)
            print(f"  {min(i + LOTE, len(X))}/{len(X)} videos procesados")

        features = np.concatenate(features, axis=0)
        print(f"  Salida: {features.shape}")
        np.save(f"dataset_listo/X_{split}_resnet_features.npy", features)

    print("\nListo. Características guardadas como:")
    print("  dataset_listo/X_train_resnet_features.npy")
    print("  dataset_listo/X_test_resnet_features.npy")
    print("(las etiquetas y_train.npy / y_test.npy / clases.txt son las mismas de siempre)")


if __name__ == "__main__":
    main()
