from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

model = load_model("skin_model.h5")

image_folder = r"d:\Skin Gaurd--AI"
available_files = os.listdir(image_folder)

test_image_name = None
for file in available_files:
    if file.lower().endswith(".jpg"):
        test_image_name = file
        break

if test_image_name is None:
    raise FileNotFoundError("No .jpg image found in the folder.")

image_path = os.path.join(image_folder, test_image_name)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

img = cv2.resize(img, (64, 64))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
predicted_class_index = np.argmax(pred)

class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
predicted_class_label = class_labels[predicted_class_index]

confidence = np.max(pred) * 100

print(f"Predicted class: {predicted_class_label} (index {predicted_class_index}) with {confidence:.2f}% confidence")
