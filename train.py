"""
Retrain the Chest X-Ray Pneumonia Detection Model
with Class Weighting to fix majority-class bias.

This script:
1. Loads the dataset (only NORMAL and PNEUMONIA classes)
2. Computes balanced class weights using sklearn
3. Builds the same CNN architecture as the original notebook
4. Trains with class_weight to penalize mistakes on the minority class
5. Evaluates using classification_report and roc_auc_score
6. Saves the new, unbiased model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ──────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "muhammadrehan00", "chest-xray-dataset", "versions", "1"
)

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR   = os.path.join(DATASET_PATH, "val")
TEST_DIR  = os.path.join(DATASET_PATH, "test")

# We only want NORMAL and PNEUMONIA (ignore tuberculosis)
# image_dataset_from_directory will pick up subdirectories alphabetically.
# The dataset has: normal, pneumonia, tuberculosis
# We need to filter to only normal and pneumonia.

IMG_HEIGHT = 180
IMG_WIDTH  = 180
BATCH_SIZE = 32
EPOCHS     = 10

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model", "xray_model.hdf5")

# ──────────────────────────────────────────────
# 2. Build filtered datasets (NORMAL & PNEUMONIA only)
# ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading dataset...")
print("=" * 60)

# Since the dataset has 3 classes but we only want 2,
# we create symlink directories with just normal and pneumonia.
import tempfile, shutil

def create_filtered_dir(source_dir, temp_base):
    """Create a temporary directory containing only 'normal' and 'pneumonia' subdirs."""
    filtered = os.path.join(temp_base, os.path.basename(source_dir))
    os.makedirs(filtered, exist_ok=True)
    for cls in ["normal", "pneumonia"]:
        src = os.path.join(source_dir, cls)
        dst = os.path.join(filtered, cls)
        if os.path.exists(src) and not os.path.exists(dst):
            # Use symlink for speed (no copy)
            try:
                os.symlink(src, dst, target_is_directory=True)
            except OSError:
                # Fallback: use junction on Windows if symlink fails
                import subprocess
                subprocess.run(["cmd", "/c", "mklink", "/J", dst, src],
                               capture_output=True, check=True)
    return filtered

temp_base = tempfile.mkdtemp(prefix="xray_train_")
print(f"Temporary filtered directory: {temp_base}")

filtered_train = create_filtered_dir(TRAIN_DIR, temp_base)
filtered_val   = create_filtered_dir(VAL_DIR, temp_base)
filtered_test  = create_filtered_dir(TEST_DIR, temp_base)

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    filtered_train,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    filtered_val,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    filtered_test,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"\nClass names: {class_names}")
print(f"  Index 0 = {class_names[0]}")
print(f"  Index 1 = {class_names[1]}")

# ──────────────────────────────────────────────
# 3. Compute Class Weights
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Computing class weights...")
print("=" * 60)

# Extract all labels from the training dataset
train_labels = []
for _, labels in train_ds:
    train_labels.extend(labels.numpy())
train_labels = np.array(train_labels)

unique_classes = np.unique(train_labels)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_labels
)
class_weight_dict = dict(zip(unique_classes.astype(int), class_weights_array))

print(f"\nTraining samples per class:")
for cls_idx in unique_classes:
    count = np.sum(train_labels == cls_idx)
    print(f"  {class_names[int(cls_idx)]}: {count} images")

print(f"\nComputed class weights:")
for cls_idx, weight in class_weight_dict.items():
    print(f"  {class_names[cls_idx]}: {weight:.4f}")

# ──────────────────────────────────────────────
# 4. Performance optimization
# ──────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ──────────────────────────────────────────────
# 5. Build the Model (same architecture as original notebook)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Building model...")
print("=" * 60)

num_classes = len(class_names)

# Data augmentation layer for the Normal class (helps with imbalance)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# ──────────────────────────────────────────────
# 6. Train with Class Weights
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Training with class weights...")
print("=" * 60)
print(f"  Epochs: {EPOCHS}")
print(f"  Class weights: {class_weight_dict}")
print()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict  # <-- THE KEY FIX
)

# ──────────────────────────────────────────────
# 7. Evaluate on Test Set
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Evaluating model on test set...")
print("=" * 60)

# Collect predictions
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    probs = tf.nn.softmax(preds).numpy()
    y_true.extend(labels.numpy())
    y_pred_probs.extend(probs)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
print("--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(f"  {'':>12s} Pred {class_names[0]:>10s}  Pred {class_names[1]:>10s}")
for i, row in enumerate(cm):
    print(f"  True {class_names[i]:>10s}  {row[0]:>10d}  {row[1]:>10d}")

# AUC-ROC (for the positive class = pneumonia)
try:
    auc = roc_auc_score(y_true, y_pred_probs[:, 1])
    print(f"\n  AUC-ROC: {auc:.4f}")
except Exception as e:
    print(f"\n  AUC-ROC calculation failed: {e}")

# Normal class recall (the critical metric)
normal_idx = class_names.index("normal")
normal_correct = cm[normal_idx][normal_idx]
normal_total   = cm[normal_idx].sum()
print(f"\n  *** Normal Recall: {normal_correct}/{normal_total} = {normal_correct/normal_total:.2%} ***")

pneumonia_idx = class_names.index("pneumonia")
pneumonia_correct = cm[pneumonia_idx][pneumonia_idx]
pneumonia_total   = cm[pneumonia_idx].sum()
print(f"  *** Pneumonia Recall: {pneumonia_correct}/{pneumonia_total} = {pneumonia_correct/pneumonia_total:.2%} ***")

# ──────────────────────────────────────────────
# 8. Save the new model
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving model...")
print("=" * 60)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"  Model saved to: {MODEL_SAVE_PATH}")

# Cleanup temp directory
try:
    shutil.rmtree(temp_base)
    print(f"  Cleaned up temp directory: {temp_base}")
except:
    print(f"  Note: Could not clean up {temp_base}")

print("\n" + "=" * 60)
print("DONE! The model has been retrained with balanced class weights.")
print("Restart your Streamlit app to use the new model.")
print("=" * 60)
