import os
import shutil
import random
from pathlib import Path

random.seed(42)

AUGMENTED_DIR = "dataset/augmented"
ORIGINAL_DIR = "dataset/original"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)
# ---------------------------------------
# DELETE OLD FOLDERS
# ---------------------------------------

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if os.path.exists(folder):
        shutil.rmtree(folder)

# ---------------------------------------
# CREATE NEW FOLDERS
# ---------------------------------------

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# ---------------------------------------
# GET CLASSES
# ---------------------------------------

classes = sorted([
    d.name
    for d in Path(ORIGINAL_DIR).iterdir()
    if d.is_dir()
])

print("\nClasses Found:")

for cls in classes:
    print(cls)

# ---------------------------------------
# CREATE TRAIN DATASET FROM AUGMENTED
# ---------------------------------------

for cls in classes:

    train_source = os.path.join(AUGMENTED_DIR, cls)

    train_images = [
        img for img in os.listdir(train_source)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    os.makedirs(
        os.path.join(TRAIN_DIR, cls),
        exist_ok=True
    )

    for img in train_images:

        shutil.copy(
            os.path.join(train_source, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    print(f"Training Images ({cls}) : {len(train_images)}")