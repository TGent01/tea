import numpy as np
from cnn_extractor import crear_cnn_extractor, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES

resultados = []


def reportar(id_caso, nombre, ok, detalle=""):
    estado = "PASA" if ok else "FALLA"
    print(f"[{estado}] {id_caso} - {nombre}" + (f": {detalle}" if detalle else ""))
    resultados.append((id_caso, nombre, ok))


def tc_cnn_01_forma_entrada_salida(extractor):
    entrada = np.random.rand(1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES).astype("float32")
    salida = extractor.predict(entrada, verbose=0)
    ok = salida.shape == (1, SEQUENCE_LENGTH, 64)
    reportar("TC-CNN-01", "Forma de entrada/salida", ok, f"forma obtenida: {salida.shape}")



def tc_cnn_03_sin_nan_ni_inf(extractor):
    entrada = np.random.rand(2, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES).astype("float32")
    salida = extractor.predict(entrada, verbose=0)
    ok = not (np.isnan(salida).any() or np.isinf(salida).any())
    reportar("TC-CNN-03", "Ausencia de NaN/Inf", ok)


def tc_cnn_04_consistencia_frame_repetido(extractor):
    frame = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, CANALES).astype("float32")
    secuencia = np.random.rand(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES).astype("float32")
    secuencia[0] = frame
    secuencia[20] = frame
    entrada = np.expand_dims(secuencia, axis=0)

    # Desactivar dropout (modo inferencia) para que la comparación sea determinística
    salida = extractor(entrada, training=False).numpy()

    vector_0 = salida[0, 0]
    vector_20 = salida[0, 20]
    ok = np.allclose(vector_0, vector_20, atol=1e-5)
    reportar("TC-CNN-04", "Consistencia de pesos compartidos", ok)


def tc_cnn_05_frames_degenerados(extractor):
    entrada_negra = np.zeros((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES), dtype="float32")
    entrada_blanca = np.ones((1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES), dtype="float32")
    try:
        salida_negra = extractor.predict(entrada_negra, verbose=0)
        salida_blanca = extractor.predict(entrada_blanca, verbose=0)
        ok = (
            salida_negra.shape == (1, SEQUENCE_LENGTH, 64)
            and salida_blanca.shape == (1, SEQUENCE_LENGTH, 64)
        )
    except Exception as e:
        ok = False
        reportar("TC-CNN-05", "Robustez ante frames degenerados", ok, f"excepción: {e}")
        return
    reportar("TC-CNN-05", "Robustez ante frames degenerados", ok)


def main():
    print("Ejecutando casos de prueba del submodelo CNN...\n")
    extractor = crear_cnn_extractor()

    tc_cnn_01_forma_entrada_salida(extractor)
    tc_cnn_03_sin_nan_ni_inf(extractor)
    tc_cnn_04_consistencia_frame_repetido(extractor)
    tc_cnn_05_frames_degenerados(extractor)

    print("\n--- Resumen ---")
    total = len(resultados)
    pasados = sum(1 for _, _, ok in resultados if ok)
    for id_caso, nombre, ok in resultados:
        print(f"  {'✔' if ok else '✘'} {id_caso}: {nombre}")
    print(f"\n{pasados}/{total} casos pasaron.")


if __name__ == "__main__":
    main()
