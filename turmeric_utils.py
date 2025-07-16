import os
import numpy as np
import cv2

def load_custom_dataset(root_path, image_size=(64, 64)):
    images = []
    labels = []
    class_names = sorted(os.listdir(root_path))
    for label_index, class_name in enumerate(class_names):
        class_folder = os.path.join(root_path, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    labels.append(label_index)
    return np.array(images), np.array(labels), class_names

def patchify(image, patch_size=8):
    patches = []
    h, w, _ = image.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    return np.array(patches)

def simple_transformer_encoder(patches):
    return np.mean(patches, axis=0)

def train_classifier(images, labels):
    features = []
    for img in images:
        patches = patchify(img)
        encoded = simple_transformer_encoder(patches)
        features.append(encoded)

    class_features = {}
    for i in range(len(labels)):
        label = labels[i]
        class_features.setdefault(label, []).append(features[i])
    for label in class_features:
        class_features[label] = np.mean(class_features[label], axis=0)
    return class_features

def predict_image(img_path, model, label_names, threshold=1000):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    patches = patchify(img)
    encoded = simple_transformer_encoder(patches)

    min_dist = float('inf')
    predicted_label = None
    for label in model:
        dist = np.linalg.norm(encoded - model[label])
        if dist < min_dist:
            min_dist = dist
            predicted_label = label

    if min_dist > threshold:
        return "Uncertain / Unknown"
    return label_names[predicted_label]

def segment_disease(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 50, 50])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def suggest_remedy(label):
    remedies = {
        "Rhizome Disease Root": """<b>Remedy for Rhizome Rot Disease:</b><br>
<ul>
  <li><b>Cultural Practices:</b>
    <ul>
      <li>Use well-drained soils; avoid waterlogging.</li>
      <li>Rotate crops with non-hosts (e.g., cereals).</li>
      <li>Use healthy, disease-free seed rhizomes.</li>
      <li>Maintain proper plant spacing and field aeration.</li>
    </ul>
  </li>
  <li><b>Chemical Control:</b>
    <ul>
      <li>Treat rhizomes with Copper oxychloride (0.25%) or Mancozeb (0.3%) for 30 mins.</li>
      <li>Drench affected areas with Metalaxyl + Mancozeb (0.25%).</li>
    </ul>
  </li>
  <li><b>Biological Control:</b>
    <ul>
      <li>Apply Trichoderma spp. or Pseudomonas fluorescens.</li>
      <li>Use neem cake enriched with bio-agents in the soil.</li>
    </ul>
  </li>
</ul>""",

        "Leaf Blotch": """<b>Remedy for Leaf Spot Disease:</b><br>
<ul>
  <li><b>Cultural Practices:</b>
    <ul>
      <li>Maintain proper spacing and airflow between plants.</li>
      <li>Remove and burn infected leaves.</li>
      <li>Use clean, certified planting materials.</li>
    </ul>
  </li>
  <li><b>Chemical Control:</b>
    <ul>
      <li>Spray Mancozeb (0.2%) or Carbendazim (0.1%) every 10–15 days.</li>
      <li>Use Chlorothalonil or Propiconazole for severe infections.</li>
    </ul>
  </li>
  <li><b>Biological Control:</b>
    <ul>
      <li>Spray with Trichoderma viride or Pseudomonas fluorescens.</li>
      <li>Apply neem oil (2%) regularly as a preventive measure.</li>
    </ul>
  </li>
</ul>""",

        "Dry Leaf": """<b>Remedy:</b><br>Ensure consistent watering schedule.<br>Avoid excessive sunlight and check for underlying fungal causes if drying persists."""
    }
    return remedies.get(label, "No remedy needed. Plant is healthy!")
