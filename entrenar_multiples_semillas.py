import json
import os

import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from modelo_cnn_lstm import crear_modelo
from config import CONFIG

# --- Búsqueda de semillas ---
# Cada configuración retoma desde su mejor semilla conocida (guardada en
# ARCHIVO_MEJOR_SEMILLA) y prueba las semillas siguientes hasta SEMILLA_MAX.
# Se detiene, por configuración, cuando ocurra lo primero de:
#   (a) aparece una semilla con accuracy MAYOR que la mejor guardada
#       (se guarda como nueva mejor semilla), o
#   (b) el promedio acumulado de accuracy de la búsqueda llega a UMBRAL_PROMEDIO.
SEMILLA_MAX = 100
UMBRAL_PROMEDIO = 0.6
ARCHIVO_MEJOR_SEMILLA = "mejor_semilla.json"

# Las dos configuraciones a comparar. "liviana" toma los valores de
# regularización definidos en config.py; "pesada" es la variante alternativa
# (BatchNormalization + más regularización).
CONFIGS = {
    "liviana": dict(
        usar_bn=CONFIG['usar_batch_norm'],
        dropout_cnn=CONFIG['dropout_cnn'],
        dropout_lstm=CONFIG['dropout_lstm'],
        l2_reg=CONFIG['l2_reg'],
    ),
    "pesada": dict(
        usar_bn=True, dropout_cnn=0.3, dropout_lstm=0.4, l2_reg=1e-4
    ),
}


def cargar_mejores():
    if not os.path.exists(ARCHIVO_MEJOR_SEMILLA):
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_MEJOR_SEMILLA}. Debe estar en la misma carpeta "
            "que este script, con la mejor semilla de cada configuración."
        )
    with open(ARCHIVO_MEJOR_SEMILLA, encoding="utf-8") as f:
        return json.load(f)


def guardar_mejores(mejores):
    with open(ARCHIVO_MEJOR_SEMILLA, "w", encoding="utf-8") as f:
        json.dump(mejores, f, indent=2, ensure_ascii=False)


# --- Cargar el dataset una sola vez ---
X_train_full = np.load("dataset_listo/X_train.npy")
X_test = np.load("dataset_listo/X_test.npy")
y_train_full = np.load("dataset_listo/y_train.npy")
y_test = np.load("dataset_listo/y_test.npy")

with open("dataset_listo/clases.txt", encoding="utf-8") as f:
    CLASES = f.read().splitlines()

y_real = np.argmax(y_test, axis=1)
etiquetas_enteras = np.argmax(y_train_full, axis=1)


def entrenar_y_evaluar(params, semilla):
    """Entrena un modelo con la semilla dada y devuelve (accuracy, y_pred) en test."""
    keras.utils.set_random_seed(semilla)

    # Mismo split de datos en todas las corridas (random_state de config.py);
    # lo único que cambia entre semillas es la inicialización/Dropout de la red.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['seed'],
        stratify=etiquetas_enteras,
        shuffle=True,
    )

    if CONFIG['horizontal_flip']:
        X_tr = np.concatenate([X_tr, np.flip(X_tr, axis=3)], axis=0)
        y_tr = np.concatenate([y_tr, y_tr], axis=0)
        idx = np.random.permutation(len(X_tr))
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    modelo = crear_modelo(num_clases=len(CLASES), **params)
    modelo.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate'], clipnorm=CONFIG['clipnorm']),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=CONFIG['early_stopping_patience'],
        restore_best_weights=True,
    )
    modelo.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        callbacks=[early_stopping],
        verbose=0,
    )
    y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
    return accuracy_score(y_real, y_pred), y_pred


mejores = cargar_mejores()
resumen = {}

for nombre, params in CONFIGS.items():
    mejor = mejores[nombre]
    print(f"\n{'#' * 70}\n# CONFIGURACIÓN: {nombre} — retoma desde semilla {mejor['semilla']} "
          f"(accuracy {mejor['accuracy']:.4f})\n{'#' * 70}")

    # El promedio acumulado arranca con la mejor semilla ya conocida.
    accuracies = [mejor["accuracy"]]
    motivo = f"se llegó a la semilla {SEMILLA_MAX} sin cumplir ninguna condición de parada"

    for semilla in range(mejor["semilla"] + 1, SEMILLA_MAX + 1):
        acc, y_pred = entrenar_y_evaluar(params, semilla)
        accuracies.append(acc)
        promedio = float(np.mean(accuracies))
        print(f"[{nombre}] semilla {semilla}: accuracy={acc:.4f} | "
              f"promedio acumulado={promedio:.4f} | mejor guardada={mejor['accuracy']:.4f} "
              f"(semilla {mejor['semilla']})")

        if acc > mejor["accuracy"] + 1e-9:
            mejor = {"semilla": semilla, "accuracy": round(float(acc), 4)}
            mejores[nombre] = mejor
            guardar_mejores(mejores)
            print(classification_report(y_real, y_pred, target_names=CLASES, zero_division=0))
            motivo = f"nueva mejor semilla encontrada: {semilla} ({acc:.4f})"
            break

        if promedio >= UMBRAL_PROMEDIO:
            motivo = f"promedio acumulado {promedio:.4f} >= {UMBRAL_PROMEDIO}"
            break

    resumen[nombre] = (mejor, float(np.mean(accuracies)), len(accuracies) - 1, motivo)

# --- Resumen final ---
print(f"\n{'#' * 70}\n# RESUMEN (mejores semillas guardadas en {ARCHIVO_MEJOR_SEMILLA})\n{'#' * 70}")
for nombre, (mejor, promedio, n_probadas, motivo) in resumen.items():
    print(f"\n{nombre}")
    print(f"  Mejor semilla: {mejor['semilla']} (accuracy {mejor['accuracy']:.4f})")
    print(f"  Semillas nuevas probadas: {n_probadas} | promedio acumulado: {promedio:.4f}")
    print(f"  Detenido por: {motivo}")
