import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from modelo_cnn_lstm import crear_modelo

# --- Semilla fija para reproducibilidad ---
# Sin esto, cada corrida inicializa los pesos de la red y aplica Dropout de
# forma distinta, lo que en un dataset tan chico puede producir resultados
# muy distintos entre corridas (incluso con el mismo código). Fijar la
# semilla no elimina la sensibilidad del modelo a la inicialización, pero sí
# hace que una misma corrida sea repetible, para poder comparar cambios de
# forma confiable.
SEED = 42
keras.utils.set_random_seed(SEED)

# --- Cargar el dataset ya preparado por construir_dataset.py ---
X_train_full = np.load("dataset_listo/X_train.npy")
X_test = np.load("dataset_listo/X_test.npy")
y_train_full = np.load("dataset_listo/y_train.npy")
y_test = np.load("dataset_listo/y_test.npy")

with open("dataset_listo/clases.txt", encoding="utf-8") as f:
    CLASES = f.read().splitlines()

print(f"Clases: {CLASES}")
print(f"Train (total): {X_train_full.shape}, Test: {X_test.shape}")

# --- Split explícito y estratificado de entrenamiento/validación ---
# (en vez de validation_split, que no mezcla los datos antes de partir y
# puede dejar una validación desbalanceada con un dataset tan chico)
etiquetas_enteras = np.argmax(y_train_full, axis=1)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=etiquetas_enteras,
    shuffle=True,
)

print(f"Entrenamiento: {X_tr.shape}, Validación: {X_val.shape}")
for i, c in enumerate(CLASES):
    print(f"  {c}: {sum(np.argmax(y_tr, axis=1) == i)} train, "
          f"{sum(np.argmax(y_val, axis=1) == i)} val")

# --- Aumento de datos: espejo horizontal (solo en entrenamiento) ---
# Se aplica DESPUÉS del split para no filtrar información hacia validación
# (un video y su espejo son casi idénticos; si uno cae en train y el otro en
# validación, la validación deja de ser honesta). Duplica el tamaño efectivo
# del set de entrenamiento, lo cual ayuda a mitigar el sobreajuste observado
# con un dataset tan chico.
X_tr_espejo = np.flip(X_tr, axis=3)  # eje 3 = ancho de la imagen
X_tr = np.concatenate([X_tr, X_tr_espejo], axis=0)
y_tr = np.concatenate([y_tr, y_tr], axis=0)

indices_mezclados = np.random.permutation(len(X_tr))
X_tr = X_tr[indices_mezclados]
y_tr = y_tr[indices_mezclados]

print(f"Entrenamiento tras aumento (espejo horizontal): {X_tr.shape}")

# --- Construir y compilar el modelo ---
modelo = crear_modelo(num_clases=len(CLASES))
modelo.compile(
    optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
modelo.summary()

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
)
checkpoint = ModelCheckpoint(
    "mejor_modelo_TEA.keras",
    monitor="val_accuracy",
    save_best_only=True,
)

# --- Entrenamiento ---
historial = modelo.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,  # vuelve a subir un poco: ahora hay el doble de datos de entrenamiento
    shuffle=True,
    callbacks=[early_stopping, checkpoint],
)

# --- Diagnóstico: curva de entrenamiento completa, época por época ---
print("\n=== Curva de entrenamiento ===")
print(f"{'Época':>5} {'loss':>8} {'acc':>8} {'val_loss':>10} {'val_acc':>9}")
for i in range(len(historial.history["loss"])):
    print(f"{i+1:>5} "
          f"{historial.history['loss'][i]:>8.4f} "
          f"{historial.history['accuracy'][i]:>8.4f} "
          f"{historial.history['val_loss'][i]:>10.4f} "
          f"{historial.history['val_accuracy'][i]:>9.4f}")

# --- Evaluación sobre el conjunto de prueba ---
y_pred_prob = modelo.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_real = np.argmax(y_test, axis=1)

print("\n=== Reporte de clasificación (precisión, recall, F1-score) ===")
print(classification_report(y_real, y_pred, target_names=CLASES, zero_division=0))

print("=== Matriz de confusión ===")
print(confusion_matrix(y_real, y_pred))

modelo.save("modelo_final_TEA.keras")
print("\nModelo guardado como modelo_final_TEA.keras")
