import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pickle
from torch.utils.data import Dataset

import torch.nn as nn

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

def preprocess_image(image):
    reshaped = image.reshape((-1, 3))
    return reshaped


def segment_using_kmeans(image, k=3):
    reshaped = preprocess_image(image)
    kmeans = KMeans(n_clusters=k, random_state=22)
    kmeans.fit(reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_flat = centers[labels].astype("uint8")
    segmented = segmented_flat.reshape(image.shape)
    return segmented, labels.reshape(image.shape[:2])


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1, box2: Bounding boxes in the format (x, y, w, h).
    Returns:
        IoU: Intersection over Union score.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)


    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)


    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection


    return intersection / union if union > 0 else 0


def non_maximum_suppression(boxes, iou_threshold=0.3):
    
    if not boxes:
        return []

    # Sort boxes by area (largest first)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    filtered_boxes = []

    while boxes:
        # Take the box with the largest area
        chosen_box = boxes.pop(0)
        filtered_boxes.append(chosen_box)

        # Remove boxes with high IoU overlap
        boxes = [box for box in boxes if compute_iou(chosen_box, box) < iou_threshold]

    return filtered_boxes


def extract_objects_from_clusters(labels, num_clusters, min_size=8999, max_boxes=3, image_shape=None, iou_threshold=0.3):
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

    
    filtered_boxes = non_maximum_suppression(bounding_boxes, iou_threshold)


    print(f"Found {len(filtered_boxes)} valid bounding boxes (max allowed: {max_boxes})")


    return filtered_boxes[:max_boxes]


def visualize_and_label_segmentation(original, segmented, bounding_boxes, model, class_names, output_path):
    result = original.copy()

    for x, y, w, h in bounding_boxes:

        roi = result[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (224, 224))
        roi_resized = np.expand_dims(roi_resized, axis=0)


        prediction = model.predict(roi_resized)
        label_index = np.argmax(prediction)
        label = class_names[label_index]


        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 216, 0), 3) 
        label_y = max(y - 20, 20)  
        cv2.putText(result, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(132)
    plt.title('Segmented Image')
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(133)
    plt.title('Labeled Segmentation')
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Annotated image saved to {output_path}")

    # Open the annotated image
    annotated_image = Image.open(output_path)
    annotated_image.show()


def process_single_image(image_path, model, class_names, output_dir, k=5, min_size=5000, max_boxes=5):
    try:
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image from {image_path}")
            return None, None

        segmented, labels = segment_using_kmeans(image, k=k)
        bounding_boxes = extract_objects_from_clusters(
            labels, k, min_size=min_size, max_boxes=max_boxes, image_shape=image.shape, iou_threshold=0.4
        )

        output_filename = os.path.join(output_dir, f"segmentation_{os.path.basename(image_path)}")
        visualize_and_label_segmentation(image, segmented, bounding_boxes, model, class_names, output_filename)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
def get_confidence_scores(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        logits = model(image) 
        probabilities = torch.sigmoid(logits)  
        confidence = probabilities.item() 
    return confidence
def run_inference_ensemble(image_path, bounding_boxes, model_paths):
    """
    Perform ensemble inference for each ROI in bounding boxes using PyTorch models.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    confidence_scores = {}  # Store results for each bounding box

    # Read image
    image = Image.open(image_path).convert("RGB")

    for box_idx, (x, y, w, h) in enumerate(bounding_boxes):
        # Crop ROI from the image
        roi = image.crop((x, y, x + w, y + h))
        roi_tensor = transform(roi).unsqueeze(0)  # Transform ROI for model input

        # Perform inference for each class
        class_confidences = {}
        for class_name, model_path in model_paths.items():
            model = CustomCNN()
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()

            with torch.no_grad():
                logits = model(roi_tensor)
                probabilities = torch.sigmoid(logits).item()
                class_confidences[class_name] = probabilities

        # Find the class with the highest confidence
        max_class = max(class_confidences, key=class_confidences.get)
        max_confidence = class_confidences[max_class]
        confidence_scores[box_idx] = (max_class, max_confidence)

    return confidence_scores

# def run_inference_ensemble(image_path):
#     confidence_scores = {}
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),  
#         transforms.ToTensor(),
#     ])
#     model_paths = {
#         'jalebi': 'model_jalebi.pth',
#         'samosa': 'model_samosa.pth',
#         'pakode': 'model_pakode.pth',
#         'chapati': 'model_chapati.pth',
#         'chai': 'model_chai.pth',
#         'dal_makhani': 'model_dal_makhani.pth',
#         'kulfi': 'model_kulfi.pth',
#         'paani_puri': 'model_paani_puri.pth',
#         'momos': 'model_momos.pth',
#         'kadai_paneer': 'model_kadai_paneer.pth',
#         'dhokla': 'model_dhokla.pth',
#         'idli': 'model_idli.pth',
#         'chole_bhature': 'model_chole_bhature.pth',
#         'pav_bhaji': 'model_pav_bhaji.pth',
#         'burger': 'model_burger.pth',
#         'kaathi_rolls': 'model_kaathi_rolls.pth',
#         'masala_dosa': 'model_masala_dosa.pth',
#         'butter_naan': 'model_butter_naan.pth',
#         'pizza': 'model_pizza.pth',
#         'fried_rice': 'model_fried_rice.pth'
#     }

#     for class_name, model_path in model_paths.items():
#         model = CustomCNN() 
#         model.load_state_dict(torch.load(model_path))
#         model.eval()

#         confidence = get_confidence_scores(image_path, model, transform)
#         confidence_scores[class_name] = confidence  # Store the confidence score

#     max_class = max(confidence_scores, key=confidence_scores.get)
#     max_confidence = confidence_scores[max_class]

#     # Annotate the image with the predicted label and bounding box
#     image = cv2.imread(image_path)
#     if image is not None:
#         height, width, _ = image.shape
#         # Draw a bounding box around the entire image
#         cv2.rectangle(image, (0, 0), (width, height), (0, 255, 0), 3)
#         # Put the label on the bounding box
#         cv2.putText(image, f"{max_class} ({max_confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#         annotated_image_path = f"annotated_{os.path.basename(image_path)}"
#         cv2.imwrite(annotated_image_path, image)
#         print(f"Annotated image saved to {annotated_image_path}")

#         # Open the annotated image
#         annotated_image = Image.open(annotated_image_path)
#         annotated_image.show()

#     return max_class

def main():
    image_path = "/Users/kashyapj/Desktop/ashoka/IML/ProjectTry/images/jalebisamosa.png"
    

    model_choice = input("Choose a Model (1: MultiClass Model, 2: MultiLabel Model (Ensemble), 3: Pretrained ResNet): ")

    if model_choice == "1":
        output_dir = "MultiClass"  
        model_path = "epoch_100_cnn_indian_food.h5"
        model = tf.keras.models.load_model(model_path)
        class_names = {
            0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 5: 'dal_makhani',
            6: 'dhokla', 7: 'fried_rice', 8: 'idli', 9: 'jalebi', 10: 'kaathi_rolls',
            11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa', 14: 'momos', 15: 'paani_puri',
            16: 'pakode', 17: 'pav_bhaji', 18: 'pizza', 19: 'samosa'
        }
        process_single_image(image_path, model, class_names, output_dir, k=3, min_size=9000, max_boxes=5)

    elif model_choice == "2":
        output_dir = "MultiLabel"
        os.makedirs(output_dir, exist_ok=True)

        # Segment image and extract bounding boxes
        image = cv2.imread(image_path)
        segmented, labels = segment_using_kmeans(image, k=3)
        bounding_boxes = extract_objects_from_clusters(
            labels, num_clusters=3, min_size=9000, max_boxes=5, image_shape=image.shape
        )
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

        confidence_scores = run_inference_ensemble(image_path, bounding_boxes, model_paths)


        for idx, (x, y, w, h) in enumerate(bounding_boxes):
            max_class, max_confidence = confidence_scores[idx]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(image, f"{max_class} ({max_confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save and display the annotated image
        output_image_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, image)
        print(f"Annotated image saved to {output_image_path}")

        # Display annotated image
        annotated_image = Image.open(output_image_path)
        annotated_image.show()

    elif model_choice == "3":
        output_dir = "PreTrained"  
        model_path = "resnet_model_epoch_20.h5"
        model = tf.keras.models.load_model(model_path)
        class_names = {
            0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 5: 'dal_makhani',
            6: 'dhokla', 7: 'fried_rice', 8: 'idli', 9: 'jalebi', 10: 'kaathi_rolls',
            11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa', 14: 'momos', 15: 'paani_puri',
            16: 'pakode', 17: 'pav_bhaji', 18: 'pizza', 19: 'samosa'
        }
        process_single_image(image_path, model, class_names, output_dir, k=3, min_size=8900, max_boxes=5)

    else:
        print("Invalid choice. Please select a valid model option.")
        exit()



if __name__ == "__main__":
    main()