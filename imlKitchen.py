import streamlit as st
import cv2
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.cluster import KMeans
from io import BytesIO
import requests

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

def segment_using_kmeans(image, k=3):
    """K-means image segmentation"""
    reshaped = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=22)
    kmeans.fit(reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_flat = centers[labels].astype("uint8")
    segmented = segmented_flat.reshape(image.shape)
    return segmented, labels.reshape(image.shape[:2])

def compute_iou(box1, box2):
    """Compute Intersection over Union for bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def extract_objects_from_clusters(labels, num_clusters, min_size=8999, max_boxes=3, image_shape=None, iou_threshold=0.3):
    """Extract bounding boxes from image clusters"""
    bounding_boxes = []
    image_height, image_width = image_shape[:2]

    for cluster in range(num_clusters):
        mask = (labels == cluster).astype("uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if (
                cv2.contourArea(contour) >= min_size
                and area >= min_size
                and w < image_width * 0.98
                and h < image_height * 0.98
            ):
                bounding_boxes.append((x, y, w, h))

    filtered_boxes = []
    while bounding_boxes:
        chosen_box = bounding_boxes.pop(0)
        filtered_boxes.append(chosen_box)
        bounding_boxes = [box for box in bounding_boxes if compute_iou(chosen_box, box) < iou_threshold]

    return filtered_boxes[:max_boxes]

def run_inference_ensemble(uploaded_image, bounding_boxes, model_paths):
    """Perform ensemble inference for uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    confidence_scores = {}

    for box_idx, (x, y, w, h) in enumerate(bounding_boxes):
        roi = uploaded_image.crop((x, y, x + w, y + h))
        roi_tensor = transform(roi).unsqueeze(0)

        class_confidences = {}
        for class_name, model_path in model_paths.items():
            model = CustomCNN()
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()

            with torch.no_grad():
                logits = model(roi_tensor, return_logits=True)
                probabilities = torch.sigmoid(logits).item()
                class_confidences[class_name] = probabilities

        max_class = max(class_confidences, key=class_confidences.get)
        max_confidence = class_confidences[max_class]
        confidence_scores[box_idx] = (max_class, max_confidence)

    return confidence_scores

def load_multiclass_model(model_path):
    """Load multiclass model with predefined class names"""
    model = tf.keras.models.load_model(model_path)
    class_names = {
        0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 
        5: 'dal_makhani', 6: 'dhokla', 7: 'fried_rice', 8: 'idli', 9: 'jalebi', 
        10: 'kaathi_rolls', 11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa', 
        14: 'momos', 15: 'paani_puri', 16: 'pakode', 17: 'pav_bhaji', 
        18: 'pizza', 19: 'samosa'
    }
    return model, class_names

def main():
    st.set_page_config(
        page_title="IML Kitchen", 
        page_icon="🍽️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        color: #FF6B35;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size:24px !important;
        color: #4A4E69;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background-color: #4A4E69;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and Subtitle
    st.markdown('<p class="big-font">🍽️ IML Kitchen</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Food Image Classification System</p>', unsafe_allow_html=True)

    # Model and Image Upload Configuration
    st.header("🔧 Model Configuration")
    model_choice = st.selectbox(
        "Select Classification Approach",
        ["Multi-Class Model", "Multi-Label Model (Ensemble)", "Pretrained ResNet"]
    )
    
    st.header("📸 Image Upload")
    uploaded_image = st.file_uploader(
        "Upload High-Resolution Image", 
        type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        accept_multiple_files=False
    )

    # Processing and Display
    if uploaded_image:
        uploaded_image = Image.open(uploaded_image)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_image, use_column_width=True)
        
        # Convert image for processing
        image_np = np.array(uploaded_image)
        
        # Segmentation
        segmented, labels = segment_using_kmeans(image_np, k=3)
        
        with col2:
            st.subheader("Segmented Image")
            st.image(segmented, use_column_width=True)
        
        # Extract bounding boxes
        bounding_boxes = extract_objects_from_clusters(
            labels, num_clusters=3, min_size=9000, max_boxes=5, image_shape=image_np.shape
        )

        # Prepare labeled image
        labeled_image = image_np.copy()
        
        if model_choice == "Multi-Label Model (Ensemble)":
            model_paths = {
                'jalebi': 'model_jalebi.pth', 
                'samosa': 'model_samosa.pth',
                'pakode': 'model_pakode.pth',
                # Add other model paths as needed
            }
            confidence_scores = run_inference_ensemble(uploaded_image, bounding_boxes, model_paths)

            # Draw bounding boxes and labels
            for idx, (x, y, w, h) in enumerate(bounding_boxes):
                max_class, max_confidence = confidence_scores[idx]
                cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(labeled_image, f"{max_class} ({max_confidence:.2f})", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        elif model_choice == "Multi-Class Model":
            model, class_names = load_multiclass_model('epoch_100_cnn_indian_food.h5')
            
            for x, y, w, h in bounding_boxes:
                roi = cv2.resize(labeled_image[y:y+h, x:x+w], (224, 224))
                roi_expanded = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi_expanded)
                label_index = np.argmax(prediction)
                label = class_names[label_index]
                
                cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(labeled_image, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        elif model_choice == "Pretrained ResNet":
            model, class_names = load_multiclass_model('resnet_model_epoch_20.h5')
            
            for x, y, w, h in bounding_boxes:
                roi = cv2.resize(labeled_image[y:y+h, x:x+w], (224, 224))
                roi_expanded = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi_expanded)
                label_index = np.argmax(prediction)
                label = class_names[label_index]
                
                cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(labeled_image, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        with col3:
            st.subheader("Labeled Image")
            st.image(labeled_image, use_column_width=True)

if __name__ == "__main__":
    main()