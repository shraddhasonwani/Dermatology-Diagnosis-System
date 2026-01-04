import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.saving import save_model

# Load metadata
print("Loading metadata...")
df = pd.read_csv("d:/Skin Gaurd--AI/HAM10000_metadata.csv")
print(f"Metadata loaded. Total entries: {len(df)}")

# Set image directory
IMAGE_DIR = "d:/Skin Gaurd--AI/image_data"

data = []
labels = []
missing = 0
print("Preprocessing images...")
for idx, image_id in enumerate(df["image_id"]):
    image_path = os.path.join(IMAGE_DIR, image_id + ".jpg")

    if not os.path.exists(image_path):
        missing += 1
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Couldn't read image {image_path}")
        missing += 1
        continue

    img = cv2.resize(img, (64, 64))
    data.append(img)

    label = df[df["image_id"] == image_id]["dx"].values[0]
    labels.append(label)

    if idx % 500 == 0:
        print(f"Processed {idx} images...")

print(f"Images processed: {len(data)}, Missing images: {missing}")

# Normalize and encode
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded, num_classes=7)

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42
)

# Build CNN model
print("Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training the model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
save_model(model, "skin_model.keras")
print("Model saved as 'skin_model.keras'")
