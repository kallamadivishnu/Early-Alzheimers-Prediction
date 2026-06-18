import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# -----------------------------
# PATHS
# -----------------------------
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "alzheimer_model.keras")

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 25

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest"
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = test_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(weights))

print(class_weights)

# -----------------------------
# BUILD MODEL
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = True

# Freeze first layers, fine-tune the last layers
for layer in base_model.layers[:-40]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

output = Dense(
    4,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

model.load_weights(MODEL_PATH)

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(f"\nTest Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
# -----------------------------
# SAVE ACCURACY GRAPH
# -----------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy.png"
    )
)

plt.close()

# -----------------------------
# SAVE LOSS GRAPH
# -----------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "loss.png"
    )
)

plt.close()

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

test_data.reset()

predictions = model.predict(
    test_data,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_data.classes

labels = list(test_data.class_indices.keys())

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(8,8))

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.colorbar()

ticks = np.arange(len(labels))

plt.xticks(
    ticks,
    labels,
    rotation=45
)

plt.yticks(
    ticks,
    labels
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i,j],
            ha="center",
            color="white" if cm[i,j] > cm.max()/2 else "black"
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=labels
)

print(report)

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

print("\n========================================")
print("Training Completed Successfully")
print("========================================")
print("Model Saved      :", MODEL_PATH)
print("Accuracy Graph   : results/accuracy.png")
print("Loss Graph       : results/loss.png")
print("Confusion Matrix : results/confusion_matrix.png")
print("Classification   : results/classification_report.txt")
print("========================================")