import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from modelo_lstm import crear_lstm_clasificador

SEED = 42
keras.utils.set_random_seed(SEED)

FEATURE_DIM_PROYECCION = 64  # mismo tamaño que usa la CNN propia, para comparar en igualdad de condiciones

# --- Cargar las características ya precalculadas por extraer_features_resnet.py ---
# (no los píxeles crudos -- eso sería recalcular ResNet50 en cada época)
X_train_full = np.load("dataset_listo/X_train_resnet_features.npy")  # (n, 40, 2048)
X_test = np.load("dataset_listo/X_test_resnet_features.npy")
y_train_full = np.load("dataset_listo/y_train.npy")
y_test = np.load("dataset_listo/y_test.npy")

with open("dataset_listo/clases.txt", encoding="utf-8") as f:
    CLASES = f.read().splitlines()

print(f"Clases: {CLASES}")
print(f"Train (total): {X_train_full.shape}, Test: {X_test.shape}")

# --- Split estratificado train/validación (igual que entrenar_modelo.py) ---
etiquetas_enteras = np.argmax(y_train_full, axis=1)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=etiquetas_enteras,
    shuffle=True,
)

# --- Aumento de datos ---
# Nota: acá NO podemos hacer espejo horizontal sobre las características ya
# extraídas (invertir un vector de 2048 valores no equivale a invertir la
# imagen). El aumento de datos, para esta variante, debería aplicarse antes
# de extraer las características (en extraer_features_resnet.py). Se omite
# por ahora para tener un primer resultado rápido; es una diferencia real
# entre esta comparación y el pipeline de la CNN propia, que sí lo usa.
print(f"Entrenamiento: {X_tr.shape}, Validación: {X_val.shape} (sin aumento de datos en esta variante)")
for i, c in enumerate(CLASES):
    print(f"  {c}: {sum(np.argmax(y_tr, axis=1) == i)} train, "
          f"{sum(np.argmax(y_val, axis=1) == i)} val")

# --- Construir el modelo: proyección (entrenable) + LSTM (sin cambios) ---
entrada = Input(shape=(40, 2048), name="features_resnet")
x = TimeDistributed(Dense(FEATURE_DIM_PROYECCION, activation="relu"), name="proyeccion_64")(entrada)

clasificador = crear_lstm_clasificador(num_clases=len(CLASES))  # espera (40, 64), igual que con la CNN propia
salida = clasificador(x)

modelo = Model(inputs=entrada, outputs=salida, name="ResNet_LSTM_TEA")
modelo.compile(
    optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
modelo.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# Esto ahora sí es rápido: no hay ningún ResNet50 corriendo dentro del loop
# de entrenamiento, solo la proyección + LSTM sobre vectores ya calculados.
historial = modelo.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    shuffle=True,
    callbacks=[early_stopping],
)

y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)
y_real = np.argmax(y_test, axis=1)

print("\n=== Reporte de clasificación (ResNet50 + LSTM) ===")
print(classification_report(y_real, y_pred, target_names=CLASES, zero_division=0))
print("=== Matriz de confusión ===")
print(confusion_matrix(y_real, y_pred))

modelo.save("modelo_resnet_lstm_TEA.keras")
print("\nModelo guardado como modelo_resnet_lstm_TEA.keras")
