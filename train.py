import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

# CREATION DU DOSSIER MODELS
os.makedirs("models", exist_ok=True)

# CHARGEMENT DATASET
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "images",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "images",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

CLASS_NAMES = train_ds.class_names
NB_CLASSES = len(CLASS_NAMES)
print("Nombre de classes :", NB_CLASSES)

# PREPROCESSING
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# MODELE
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NB_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()


# PHASE 1 : ENTRAINEMENT DE LA TETE (backbone gele)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# Evaluation phase 1
loss, accuracy = model.evaluate(val_ds)
print(f"Accuracy phase 1 (tete seule) : {accuracy*100:.2f}%")


# PHASE 2 : FINE-TUNING DU BACKBONE

print("\n Fine-tuning des dernières couches de MobileNetV2...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# IMPORTANT : nouvel EarlyStopping avec patience plus haute
# pour laisser le temps au fine-tuning de progresser
early_stop_finetune = EarlyStopping(
    monitor="val_accuracy",    # on surveille accuracy, pas loss
    patience=5,                # plus de patience qu'en phase 1
    restore_best_weights=True
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,                 # plus d'epochs pour que ça progresse
    callbacks=[early_stop_finetune]
)

# EVALUATION FINALE
loss, accuracy = model.evaluate(val_ds)
print(f"Accuracy finale apres fine-tuning : {accuracy*100:.2f}%")

# SAUVEGARDE
model.save("models/food_classifier.keras")
with open("models/classes.pkl", "wb") as f:
    pickle.dump(CLASS_NAMES, f)
print("Modèle sauvegardé avec succès")