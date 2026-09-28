import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)
print(" TensorFlow configurado")
BASE      = r"C:\Users\Cris\Desktop\breast_cancer\dataset"
train_dir = BASE + r"\train"
val_dir   = BASE + r"\val"
test_dir  = BASE + r"\test"

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range         = 10,
    zoom_range             = 0.1,
    horizontal_flip        = True
)
val_datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(150,150),
    batch_size=32, class_mode='binary'
)
val_data = val_datagen.flow_from_directory(
    val_dir, target_size=(150,150),
    batch_size=32, class_mode='binary'
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(150,150),
    batch_size=32, class_mode='binary',
    shuffle=False  # importante para la matriz de confusión
)
base_model = VGG16(
    weights     = 'imagenet',
    include_top = False,
    input_shape = (150, 150, 3)
)
base_model.trainable = False
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss      = 'binary_crossentropy',
    metrics   = ['acc']
)
model.summary()
checkpoint = ModelCheckpoint(
    filepath       = 'mejor_modelo.keras',
    monitor        = 'val_acc',
    save_best_only = True,
    mode           = 'max',
    verbose        = 1
)
early_stop = EarlyStopping(
    monitor              = 'val_loss',
    patience             = 5,
    restore_best_weights = True,
    verbose              = 1
)

print("\n FASE 1: Entrenando nuevas capas")
history1 = model.fit(
    train_data,
    epochs          = 25,
    validation_data = val_data,
    callbacks       = [checkpoint, early_stop]
)

print("\n FASE 2: Fine tuning últimas 4 capas VGG16")
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000085),
    loss      = 'binary_crossentropy',
    metrics   = ['acc']
)
reduce_lr = ReduceLROnPlateau(
    monitor  = 'val_loss',
    factor   = 0.5,
    patience = 2,
    min_lr   = 1e-7,
    verbose  = 1
)
early_stop_f2 = EarlyStopping(
    monitor              = 'val_loss',
    patience             = 5,
    restore_best_weights = True,
    verbose              = 1
)
history2 = model.fit(
    train_data,
    epochs          = 30,
    validation_data = val_data,
    callbacks       = [checkpoint, reduce_lr, early_stop_f2]
)

acc      = history1.history['acc']      + history2.history['acc']
val_acc  = history1.history['val_acc']  + history2.history['val_acc']
loss     = history1.history['loss']     + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

fin_fase1 = len(history1.history['acc']) - 1

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(acc,     label='Train')
plt.plot(val_acc, label='Validación')
plt.axvline(x=fin_fase1, color='gray', linestyle='--', label='Inicio fine tuning')
plt.title('Accuracy por época')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,     label='Train')
plt.plot(val_loss, label='Validación')
plt.axvline(x=fin_fase1, color='gray', linestyle='--', label='Inicio fine tuning')
plt.title('Loss por época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\n Evaluando mejor modelo ")
mejor = load_model('mejor_modelo.keras')
loss_test, acc_test = mejor.evaluate(test_data)
print(f"\nAccuracy en test: {acc_test:.4f}")
print(f"Loss en test:     {loss_test:.4f}")

test_data.reset()
y_pred      = mejor.predict(test_data)
y_pred_bin  = (y_pred > 0.5).astype(int).flatten()
y_real      = test_data.classes

cm = confusion_matrix(y_real, y_pred_bin)
tn, fp, fn, tp = cm.ravel()

sensibilidad  = tp / (tp + fn)   
especificidad = tn / (tn + fp)   
precision     = tp / (tp + fp)
f1            = 2 * (precision * sensibilidad) / (precision + sensibilidad)

print(f"\n── Métricas generales ─────────")
print(f"Sensibilidad (recall maligno): {sensibilidad:.4f} ({sensibilidad*100:.1f}%)")
print(f"Especificidad (recall benigno):{especificidad:.4f} ({especificidad*100:.1f}%)")
print(f"Precisión:                     {precision:.4f} ({precision*100:.1f}%)")
print(f"F1-Score:                      {f1:.4f}")
print(f"\nVP (Verdaderos Positivos):  {tp}")
print(f"VN (Verdaderos Negativos):  {tn}")
print(f"FP (Falsos Positivos):      {fp}")
print(f"FN (Falsos Negativos):      {fn}")

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.show()

print("\n── Reporte completo ─────────────")
print(classification_report(y_real, y_pred_bin,
      target_names=['Benigno', 'Maligno']))