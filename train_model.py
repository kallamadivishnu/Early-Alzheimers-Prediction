import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "alzheimer_model.keras"
)

CLASS_FILE = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

IMG_SIZE = (224,224)

BATCH_SIZE = 32

EPOCHS = 30

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------
# DATA GENERATORS
# -------------------------------------------------------

train_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    zoom_range=0.20,

    width_shift_range=0.15,

    height_shift_range=0.15,

    shear_range=0.15,

    horizontal_flip=True,

    brightness_range=[0.85,1.15],

    fill_mode="nearest"

)

validation_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

test_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------

train_data = train_generator.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True,

    seed=SEED

)

val_data = validation_generator.flow_from_directory(

    VAL_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

test_data = test_generator.flow_from_directory(

    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

# -------------------------------------------------------
# SAVE CLASS NAMES
# -------------------------------------------------------

class_names = list(train_data.class_indices.keys())

print("\nClass Order Used By Model:")

print(class_names)

with open(CLASS_FILE,"w") as f:

    json.dump(class_names,f,indent=4)

# -------------------------------------------------------
# CLASS WEIGHTS
# -------------------------------------------------------

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_data.classes),

    y=train_data.classes

)

class_weights = dict(enumerate(weights))

print("\nClass Weights")

print(class_weights)

# -------------------------------------------------------
# BUILD EFFICIENTNET MODEL
# -------------------------------------------------------

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the backbone initially
base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.4)(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(0.3)(x)

outputs = Dense(
    train_data.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=outputs
)

# -------------------------------------------------------
# COMPILE MODEL
# -------------------------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)

print("\nModel Summary\n")

model.summary()

# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------

checkpoint = ModelCheckpoint(

    filepath=MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    save_weights_only=False,

    mode="max",

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

# -------------------------------------------------------
# STAGE 1 TRAINING
# -------------------------------------------------------

print("\nStarting Stage 1 Training...\n")

history_stage1 = model.fit(

    train_data,

    validation_data=val_data,

    epochs=10,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]

)
# -------------------------------------------------------
# STAGE 2 - FINE TUNING
# -------------------------------------------------------

print("\nStarting Stage 2 Fine-Tuning...\n")

# Unfreeze the last 30 layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a lower learning rate
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

history_stage2 = model.fit(

    train_data,

    validation_data=val_data,

    epochs=20,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]

)

# -------------------------------------------------------
# LOAD BEST MODEL
# -------------------------------------------------------

print("\nLoading Best Saved Model...\n")

model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------------------------------
# TEST EVALUATION
# -------------------------------------------------------

print("\nEvaluating on Test Dataset...\n")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print("\n========================================")
print(f"Test Accuracy : {test_accuracy*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
print("========================================")

# -------------------------------------------------------
# PREDICTIONS
# -------------------------------------------------------

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

labels = list(train_data.class_indices.keys())

# -------------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------------

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=labels
)

print("\nClassification Report\n")
print(report)

with open(
    os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

# -------------------------------------------------------
# ACCURACY GRAPH
# -------------------------------------------------------

train_acc = (
    history_stage1.history["accuracy"] +
    history_stage2.history["accuracy"]
)

val_acc = (
    history_stage1.history["val_accuracy"] +
    history_stage2.history["val_accuracy"]
)

plt.figure(figsize=(8,6))

plt.plot(train_acc, linewidth=2, label="Training Accuracy")
plt.plot(val_acc, linewidth=2, label="Validation Accuracy")

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

# -------------------------------------------------------
# LOSS GRAPH
# -------------------------------------------------------

train_loss = (
    history_stage1.history["loss"] +
    history_stage2.history["loss"]
)

val_loss = (
    history_stage1.history["val_loss"] +
    history_stage2.history["val_loss"]
)

plt.figure(figsize=(8,6))

plt.plot(train_loss, linewidth=2, label="Training Loss")
plt.plot(val_loss, linewidth=2, label="Validation Loss")

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

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(8,8))

plt.imshow(cm, cmap="Blues")

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
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max()/2 else "black"
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

# -------------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------------

print("\n==========================================")
print("Training Completed Successfully")
print("==========================================")
print("Model Saved        :", MODEL_PATH)
print("Class File Saved   :", CLASS_FILE)
print("Accuracy Graph     : results/accuracy.png")
print("Loss Graph         : results/loss.png")
print("Confusion Matrix   : results/confusion_matrix.png")
print("Classification Rep : results/classification_report.txt")
print("==========================================")

