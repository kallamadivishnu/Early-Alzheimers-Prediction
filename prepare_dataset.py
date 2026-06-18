import os
import shutil
import random
from pathlib import Path

# -----------------------------
# SETTINGS
# -----------------------------
random.seed(42)

SOURCE_DIR = "dataset/original"

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# -----------------------------
# CREATE FOLDERS
# -----------------------------
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)


classes = [d.name for d in Path(SOURCE_DIR).iterdir() if d.is_dir()]

print("\nFound Classes:")
print(classes)

for cls in classes:

    src = os.path.join(SOURCE_DIR, cls)

    images = [
        img for img in os.listdir(src)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_SPLIT)
    val_count = int(total * VAL_SPLIT)

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(VAL_DIR, cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(f"\n{cls}")
    print(f"Total Images : {total}")
    print(f"Train : {len(train_imgs)}")
    print(f"Validation : {len(val_imgs)}")
    print(f"Test : {len(test_imgs)}")

print("\n===================================")
print("Dataset preparation completed!")
print("===================================")