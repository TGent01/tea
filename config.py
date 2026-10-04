# Configuración centralizada de hiperparámetros para el entrenamiento del
# modelo CNN-LSTM (detección de patrones de motricidad en TEA).
#
# Ajustar los valores acá en vez de editarlos sueltos dentro de
# entrenar_modelo.py / cnn_extractor.py / modelo_lstm.py. Cada corrida con una
# configuración distinta queda documentada simplemente guardando una copia de
# este archivo (por ejemplo config_v1.py, config_v2.py...), lo cual sirve
# como registro del ajuste de hiperparámetros para la sección de resultados.

CONFIG = {
    # --- Datos ---
    'sequence_length': 40,       # frames por video (debe coincidir con extraer_frame.py)
    'image_size': 64,            # alto/ancho de cada frame en píxeles
    'test_size': 0.2,            # proporción de validación dentro del set de entrenamiento
    'seed': 42,                  # semilla fija, para que una misma corrida sea reproducible

    # --- Entrenamiento ---
    'batch_size': 8,
    'learning_rate': 5e-4,
    'clipnorm': 1.0,             # recorte de gradiente, protección extra con este learning rate
    'epochs': 100,
    'early_stopping_patience': 15,

    # --- Augmentation ---
    # Solo espejo horizontal por ahora (ver entrenar_modelo.py). Si se agregan
    # más tipos de aumento (brillo, rotación), sus parámetros van acá.
    'augmentation_multiplier': 2,   # 2 = original + espejo horizontal
    'horizontal_flip': True,

    # --- Regularización (arquitectura) ---
    # Configuración "liviana": la que dio mejores resultados empíricos con
    # este dataset (ver comparación contra la versión con BatchNormalization
    # en entrenar_multiples_semillas.py).
    'usar_batch_norm': False,
    'dropout_cnn': 0.3,
    'dropout_lstm': 0.3,
    'l2_reg': 0.0,                # 0.0 = sin regularización L2
}
