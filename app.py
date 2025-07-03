# cnn_fashion.py
# ---------------------------------------------------------------
# 1) IMPORTS
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# 2) CONFIGURACIÓN GENERAL
DATA_FILE = "fashion_dataset.csv"          # nombre del CSV
TEST_SIZE  = 0.20
RANDOM_SEED = 42
EPOCHS = 20
BATCH  = 16

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------------------------------------------------------
# 3) CARGA ROBUSTA DEL CSV
def robust_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df

if not Path(DATA_FILE).is_file():
    sys.exit(f"[ERROR] No se encontró {DATA_FILE} en {os.getcwd()}")

df = robust_read_csv(DATA_FILE)
if len(df) < 100:
    sys.exit(f"[ERROR] El archivo parece incompleto. Solo tiene {len(df)} filas.")

print(f"Cargadas {df.shape[0]} filas y {df.shape[1]} columnas")

# ---------------------------------------------------------------
# 4) CONVERTIR Y RE-MAPEAR ETIQUETAS A RANGO CONTIGUO
df["label"] = pd.to_numeric(df["label"], errors="coerce")
unique_labels = sorted(df["label"].dropna().unique())
label_map = {orig: idx for idx, orig in enumerate(unique_labels)}
df["label"] = df["label"].map(label_map).astype(int)
print("Mapeo etiquetas aplicado:", label_map)
print("Distribución de clases (antes de duplicar):\n", df["label"].value_counts())

# ---------------------------------------------------------------
# 5) ASEGURAR ≥ 2 EJEMPLOS POR CLASE
min_count = df["label"].value_counts().min()
if min_count < 2:
    rare_labels = df["label"].value_counts()[df["label"].value_counts() < 2].index
    for lbl in rare_labels:
        row = df[df["label"] == lbl].iloc[0]
        while (df["label"] == lbl).sum() < 2:
            df = pd.concat([df, row.to_frame().T], ignore_index=True)
    print("Clases duplicadas para garantizar ≥2 muestras.")

print("Distribución de clases (después de duplicar):\n", df["label"].value_counts())

# ---------------------------------------------------------------
# 6) ONE-HOT ENCODING
cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "label"]
df = pd.get_dummies(df, columns=cat_cols)

# ---------------------------------------------------------------
# 7) PREPARAR FEATURES Y LABELS
X = df.drop("label", axis=1).values.astype("float32")
y = df["label"].values.astype("int32")
X /= X.max() if X.max() != 0 else 1.0
X = X.reshape(X.shape[0], X.shape[1], 1)

# ---------------------------------------------------------------
# 8) SPLIT TRAIN / TEST
stratify_param = y if np.min(np.bincount(y)) >= 2 else None
if stratify_param is None:
    print("[AVISO] Estratificación desactivada (clase con <2 muestras).")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=stratify_param
)

# ---------------------------------------------------------------
# 9) MODELO CNN 1-D
num_features = X.shape[1]
num_classes  = len(label_map)

model = Sequential([
    Conv1D(32, 3, activation="relu", input_shape=(num_features, 1)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation="relu"),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------
# 10) ENTRENAMIENTO
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=2
)

# ---------------------------------------------------------------
# 11) EVALUACIÓN
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# ---------------------------------------------------------------
# 12) MATRIZ DE CONFUSIÓN
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# ---------------------------------------------------------------
# 13) CURVAS DE ENTRENAMIENTO
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
# ---------------------------------------------------------------
