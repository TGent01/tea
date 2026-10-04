import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from modelo_cnn_lstm import crear_modelo
from config import CONFIG

SEMILLAS = [0, 1, 2, 3, 4]

# Las dos configuraciones a comparar. "liviana" toma los valores de
# regularización definidos en config.py (la configuración que se usa en
# entrenar_modelo.py); "pesada" es una variante alternativa explícita, para
# comparar de forma controlada si vale la pena BatchNormalization + más
# regularización con este dataset.
CONFIGS = {
    "liviana (según config.py)": dict(
        usar_bn=CONFIG['usar_batch_norm'],
        dropout_cnn=CONFIG['dropout_cnn'],
        dropout_lstm=CONFIG['dropout_lstm'],
        l2_reg=CONFIG['l2_reg'],
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

        # Mismo random_state en el split de datos (el de config.py, no la
        # semilla del modelo) para que todas las corridas usen la misma
        # partición, y la única diferencia real sea la inicialización de la red.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['seed'],
            stratify=etiquetas_enteras,
            shuffle=True,
        )

        # Aumento de datos: espejo horizontal (solo entrenamiento), según config.py
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
