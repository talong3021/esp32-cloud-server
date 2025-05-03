import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

# Set image size and dataset root
IMAGE_SIZE = (64, 64)
DATASET_PATH = r"C:\Users\QCU\Desktop\plant_model_training\eggplant_dataset\Eggplant Disease Recognition Dataset"

X = []
y = []

print("[INFO] Scanning folders and loading images...")
print(f"[DEBUG] Dataset path: {DATASET_PATH}")  # Debugging the dataset path

# Loop over "Augmented Image", "Original Images", and "Negative Samples"
for folder in ["Augmented Images", "Original Images", "Negative Samples"]:
    full_path = os.path.join(DATASET_PATH, folder)
    print(f"[DEBUG] Checking folder: {full_path}")  # Debugging the folder path
    if not os.path.isdir(full_path):
        print(f"[WARNING] Folder not found: {full_path}")
        continue

    # Loop through each category (label)
    for label in os.listdir(full_path):
        class_dir = os.path.join(full_path, label)
        print(f"[DEBUG] Checking category folder: {class_dir}")  # Debugging category folder
        if not os.path.isdir(class_dir):
            continue

        # If folder is "Negative Samples", assign a fixed label
        if folder == "Negative Samples":
            label = "Not an Eggplant"

        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                img_path = os.path.join(class_dir, img_file)
                try:
                    print(f"[DEBUG] Found image: {img_path}")
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMAGE_SIZE)
                    X.append(np.array(img).flatten())
                    y.append(label)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

# Check if any images were loaded
if len(X) == 0 or len(y) == 0:
    print("[ERROR] No images found in the dataset. Please check the dataset path and structure.")
    exit()

print(f"[INFO] Loaded {len(X)} images from all categories.")

# Split the dataset into training and testing sets
print("[INFO] Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[INFO] Training set size: {len(X_train)}, Testing set size: {len(X_test)}")

# Train a RandomForestClassifier
print("[INFO] Training the RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("[INFO] Model training complete.")

# Evaluate the model
print("[INFO] Evaluating the model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
print("[INFO] Saving model as 'eggplant_model.joblib'...")
dump(model, "eggplant_model.joblib")
print("[DONE] Model saved successfully.")