import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from modelo_cnn_lstm import crear_modelo

SEMILLAS = [5,6,7,8]

# Las dos configuraciones a comparar. "liviana" es la que dio 67% en una
# corrida suelta; "pesada" es la que colapsó a resultados degenerados.
# Esta vez las comparamos de forma controlada: mismas semillas, mismo split
# de datos, para saber cuál es mejor EN PROMEDIO, no en una corrida suelta.
CONFIGS = {
    "liviana (sin BN, dropout 0.3, sin L2)": dict(
        usar_bn=False, dropout_cnn=0.3, dropout_lstm=0.3, l2_reg=0.0
    ),
    "pesada (con BN, dropout 0.3/0.4, L2 1e-4)": dict(
        usar_bn=True, dropout_cnn=0.3, dropout_lstm=0.4, l2_reg=1e-4
    ),
}

# --- Cargar el dataset una sola vez ---
X_train_full = np.load("dataset_listo/X_train.npy")
X_test = np.load("dataset_listo/X_test.npy")
y_train_full = np.load("dataset_listo/y_train.npy")
y_test = np.load("dataset_listo/y_test.npy")

with open("dataset_listo/clases.txt", encoding="utf-8") as f:
    CLASES = f.read().splitlines()

y_real = np.argmax(y_test, axis=1)
etiquetas_enteras = np.argmax(y_train_full, axis=1)

resultados_por_config = {}

for nombre_config, params in CONFIGS.items():
    print(f"\n{'#' * 70}\n# CONFIGURACIÓN: {nombre_config}\n{'#' * 70}")
    accuracies = []

    for semilla in SEMILLAS:
        print(f"\n{'=' * 60}\n{nombre_config} — semilla {semilla}\n{'=' * 60}")
        keras.utils.set_random_seed(semilla)

        # Mismo random_state en el split de datos (no es la semilla del
        # modelo) para que todas las corridas usen la misma partición, y la
        # única diferencia real sea la inicialización de la red.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=etiquetas_enteras,
            shuffle=True,
        )

        # Aumento de datos: espejo horizontal (solo entrenamiento)
        X_tr = np.concatenate([X_tr, np.flip(X_tr, axis=3)], axis=0)
        y_tr = np.concatenate([y_tr, y_tr], axis=0)
        idx = np.random.permutation(len(X_tr))
        X_tr, y_tr = X_tr[idx], y_tr[idx]

        modelo = crear_modelo(num_clases=len(CLASES), **params)
        modelo.compile(
            optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

        modelo.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=8,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0,
        )

        y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_real, y_pred)
        accuracies.append(acc)
        print(f"Semilla {semilla}: accuracy en test = {acc:.4f}")
        print(classification_report(y_real, y_pred, target_names=CLASES, zero_division=0))

    resultados_por_config[nombre_config] = np.array(accuracies)

# --- Resumen comparativo final ---
print(f"\n{'#' * 70}\n# RESUMEN COMPARATIVO ({len(SEMILLAS)} semillas por configuración)\n{'#' * 70}")
for nombre_config, accs in resultados_por_config.items():
    print(f"\n{nombre_config}")
    for semilla, acc in zip(SEMILLAS, accs):
        print(f"  Semilla {semilla}: {acc:.4f}")
    print(f"  --> Promedio: {accs.mean():.4f} ± {accs.std():.4f}  "
          f"(mín {accs.min():.4f}, máx {accs.max():.4f})")
