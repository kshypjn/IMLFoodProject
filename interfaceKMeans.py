import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import cv2
import torch
from torchvision import transforms
from torch import nn

# PyTorch Model for MultiLabel Inference
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_logits=False):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        logits = self.fc2(x)
        probabilities = torch.sigmoid(logits)

        if return_logits:
            return logits
        return probabilities


# Utility Functions
def preprocess_image(image):
    """Flatten and convert the image to float32."""
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    reshaped = image.reshape((-1, 3))
    return reshaped


def segment_using_kmeans(image, k=3):
    """Perform KMeans segmentation."""
    reshaped = preprocess_image(image)
    kmeans = KMeans(n_clusters=k, random_state=22)
    kmeans.fit(reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_flat = centers[labels].astype("uint8")
    segmented = segmented_flat.reshape(image.shape)
    return segmented, labels.reshape(image.shape[:2])


def extract_objects_from_clusters(labels, num_clusters, min_size, max_boxes, image_shape):
    """Extract bounding boxes around clusters."""
    bounding_boxes = []
    image_height, image_width = image_shape[:2]
    for cluster in range(num_clusters):
        mask = (labels == cluster).astype("uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area >= min_size and w < image_width * 0.98 and h < image_height * 0.98:
                bounding_boxes.append((x, y, w, h))
    return bounding_boxes[:max_boxes]


def predict_label(roi, model, class_names):
    """Predict label for a given ROI using TensorFlow model."""
    roi_resized = cv2.resize(roi, (224, 224))
    roi_resized = np.expand_dims(roi_resized, axis=0)
    prediction = model.predict(roi_resized)
    label_index = np.argmax(prediction)
    return class_names[label_index]


def run_inference_ensemble(image, model_paths):
    """Perform ensemble inference with PyTorch models."""
    confidence_scores = {}
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    for class_name, model_path in model_paths.items():
        model = CustomCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)
            confidence_scores[class_name] = probabilities.item()

    max_class = max(confidence_scores, key=confidence_scores.get)
    max_confidence = confidence_scores[max_class]
    return max_class, max_confidence


# Streamlit Interface
st.title("Food Image Segmentation and Classification")
st.sidebar.header("Settings")

# Sidebar Inputs
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
model_choice = st.sidebar.selectbox(
    "Choose a Model", ["MultiClass Model", "MultiLabel Model (Ensemble)", "Pretrained ResNet"]
)
k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)
min_size = st.sidebar.slider("Minimum Size of Objects", min_value=5000, max_value=20000, value=9000)
max_boxes = st.sidebar.slider("Maximum Number of Boxes", min_value=1, max_value=10, value=5)

if uploaded_image:

    image = Image.open(uploaded_image).convert("RGB")  
    image = np.array(image)  
    segmented, labels = segment_using_kmeans(image, k=k)
    bounding_boxes = extract_objects_from_clusters(labels, k, min_size, max_boxes, image.shape)
    annotated_image = image.copy()

    if model_choice == "MultiClass Model":
        model_path = "epoch_100_cnn_indian_food.h5"
        model = tf.keras.models.load_model(model_path)

        class_names = {
            0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 5: 'dal_makhani',
            6: 'dhokla', 7: 'fried_rice', 8: 'idli', 9: 'jalebi', 10: 'kaathi_rolls',
            11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa', 14: 'momos', 15: 'paani_puri',
            16: 'pakode', 17: 'pav_bhaji', 18: 'pizza', 19: 'samosa'
        }

        for x, y, w, h in bounding_boxes:
            roi = image[y:y + h, x:x + w]
            label = predict_label(roi, model, class_names)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 216, 0), 3)
            cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    elif model_choice == "MultiLabel Model (Ensemble)":
        model_paths = {
            'jalebi': 'model_jalebi.pth',
            'samosa': 'model_samosa.pth',
            'pakode': 'model_pakode.pth',
            'chapati': 'model_chapati.pth',
            'chai': 'model_chai.pth',
            'dal_makhani': 'model_dal_makhani.pth',
            'kulfi': 'model_kulfi.pth',
            'paani_puri': 'model_paani_puri.pth',
            'momos': 'model_momos.pth',
            'kadai_paneer': 'model_kadai_paneer.pth',
            'dhokla': 'model_dhokla.pth',
            'idli': 'model_idli.pth',
            'chole_bhature': 'model_chole_bhature.pth',
            'pav_bhaji': 'model_pav_bhaji.pth',
            'burger': 'model_burger.pth',
            'kaathi_rolls': 'model_kaathi_rolls.pth',
            'masala_dosa': 'model_masala_dosa.pth',
            'butter_naan': 'model_butter_naan.pth',
            'pizza': 'model_pizza.pth',
            'fried_rice': 'model_fried_rice.pth'
        }

        pil_image = Image.fromarray(image)
        max_class, max_confidence = run_inference_ensemble(pil_image, model_paths)

        for x, y, w, h in bounding_boxes:
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 216, 0), 3)
            cv2.putText(annotated_image, f"{max_class} ({max_confidence:.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    elif model_choice == "Pretrained ResNet":
        model_path = "resnet_model_epoch_20.h5"
        model = tf.keras.models.load_model(model_path)

        class_names = {
            0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 5: 'dal_makhani',
            6: 'dhokla', 7: 'fried_rice', 8: 'idli', 9: 'jalebi', 10: 'kaathi_rolls',
            11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa', 14: 'momos', 15: 'paani_puri',
            16: 'pakode', 17: 'pav_bhaji', 18: 'pizza', 19: 'samosa'
        }

        for x, y, w, h in bounding_boxes:
            roi = image[y:y + h, x:x + w]
            label = predict_label(roi, model, class_names)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 216, 0), 3)
            cv2.putText(annotated_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display Results
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(segmented.astype("uint8"), caption="Segmented Image", use_column_width=True)
    st.image(annotated_image, caption="Labeled Image", use_column_width=True)
