import cv2
import os
import torch
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from skimage.feature import local_binary_pattern
from sklearn.cluster import AgglomerativeClustering

# Custom modules (keep same)
from .detector import TreeDetector
from .density import calculate_density
from .statistics import compute_spacing
from .heatmap import generate_density_heatmap
from .ndvi import compute_ndvi_proxy
from .visualization import draw_detections
from .health import compute_forest_health
from .report import generate_forest_report

# -------------------- DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Preprocessing --------------------
def resize_and_pad(image, target_size=1024):
    h, w, _ = image.shape
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    square_img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
    return square_img

def gamma_correction(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def green_mask(image, lower_hsv=(30,40,40), upper_hsv=(95,255,255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def preprocess_image(image_path, target_size=1024):
    image = cv2.imread(image_path)
    square_img = resize_and_pad(image, target_size)
    gamma_img = gamma_correction(square_img, gamma=1.2)
    masked_img = green_mask(gamma_img)
    temp_path = "temp_preprocessed.jpg"
    cv2.imwrite(temp_path, masked_img)
    return masked_img, temp_path

# -------------------- Tree Filtering --------------------
def filter_non_green_trees(image, predictions, green_thresh=50):
    keep_idx = []
    for i, row in predictions.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size == 0: continue
        avg_green = np.mean(crop[:,:,1])
        if avg_green > green_thresh:
            keep_idx.append(i)
    return predictions.iloc[keep_idx].reset_index(drop=True)

# -------------------- Initialize Models --------------------
detector = TreeDetector()
weights = ResNet50_Weights.DEFAULT
cnn_model = resnet50(weights=weights)
cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])
cnn_model.to(device)
cnn_model.eval()
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------- Species Clustering --------------------
def extract_tree_crops(image, predictions):
    crops = []
    for _, row in predictions.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        crop = image[ymin:ymax, xmin:xmax]
        if crop.size>0:
            crops.append(crop)
    return crops

def extract_tree_features(crops, lbp_points=8, lbp_radius=1, hsv_bins=16):
    features = []
    for crop in crops:
        h, w = crop.shape[:2]
        if h == 0 or w == 0: 
            features.append(np.zeros(50))
            continue
        # HSV histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_hist = np.histogram(hsv[:,:,0], bins=hsv_bins, range=(0,180), density=True)[0]
        s_hist = np.histogram(hsv[:,:,1], bins=hsv_bins, range=(0,255), density=True)[0]
        v_hist = np.histogram(hsv[:,:,2], bins=hsv_bins, range=(0,255), density=True)[0]
        # LBP histogram
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=lbp_points+2, range=(0, lbp_points+2), density=True)
        # Area & aspect ratio
        area = h*w
        aspect = w/h
        feat = np.concatenate([h_hist, s_hist, v_hist, lbp_hist, [area/1e4, aspect]])
        features.append(feat)
    return np.array(features)

def cluster_species(features, n_clusters=5):
    if len(features) == 0:
        return []
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(features)
    return labels

def assign_species_names(labels, species_list=None):
    if species_list is None:
        species_list = ["TEAK","ROTH","SASAD","ROSEWOOD","MALABAR KINO"]
    unique_labels = np.unique(labels)
    label_to_name = {lbl: species_list[i % len(species_list)] for i,lbl in enumerate(unique_labels)}
    named_labels = [label_to_name[lbl] for lbl in labels]
    return named_labels, label_to_name

def annotate_species_clusters(image, predictions, named_labels, label_to_name):
    n_labels = len(label_to_name)
    cmap = plt.cm.get_cmap('tab10', max(n_labels,10))
    annotated = image.copy()
    for i, row in predictions.iterrows():
        if i >= len(named_labels): break
        xmin, ymin, xmax, ymax = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        species_name = named_labels[i]
        label = list(label_to_name.keys())[list(label_to_name.values()).index(species_name)]
        color = tuple(int(255*c) for c in cmap(label%10)[:3])
        cv2.rectangle(annotated, (xmin,ymin),(xmax,ymax), color,2)
        #cv2.putText(annotated, species_name, (xmin,ymin-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    return annotated

def count_trees_per_species(named_labels):
    counter = Counter(named_labels)
    return dict(counter)

def classify_and_annotate_species(image, predictions, n_clusters=5, species_list=None):
    crops = extract_tree_crops(image, predictions)
    features = extract_tree_features(crops)
    labels = cluster_species(features, n_clusters=n_clusters)
    named_labels, label_to_name = assign_species_names(labels, species_list)
    annotated = annotate_species_clusters(image, predictions, named_labels, label_to_name)
    species_count = count_trees_per_species(named_labels)
    return annotated, named_labels, species_count

# -------------------- Biodiversity Index --------------------
def compute_biodiversity_index(labels):
    labels = np.array(labels)
    if len(labels)==0: return 0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts/counts.sum()
    shannon_index = -np.sum(probs*np.log(probs))
    return float(shannon_index)

# -------------------- Main Pipeline --------------------
def analyze_forest(image_path, resolution=0.5):
    image, temp_path = preprocess_image(image_path)
    height, width, _ = image.shape
    predictions = detector.detect(temp_path)
    predictions = filter_non_green_trees(image, predictions)
    tree_count = len(predictions)
    area_km2, density = calculate_density(tree_count, width, height, resolution)
    spacing = compute_spacing(predictions)
    heatmap = generate_density_heatmap(image, predictions)
    ndvi_map, ndvi_raw = compute_ndvi_proxy(image)
    annotated = draw_detections(image, predictions)
    health_score = compute_forest_health(ndvi_raw, density)

    # --- Species Clustering ---
    species_annotated, named_labels, species_count = classify_and_annotate_species(
        image, predictions, n_clusters=5
    )
    print("Tree count per species:", species_count)
    biodiversity_index = compute_biodiversity_index(named_labels)

    # --- Save outputs ---
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/tree_density_heatmap.png", heatmap)
    cv2.imwrite("outputs/vegetation_health_map.png", ndvi_map)
    cv2.imwrite("outputs/tree_detections.png", annotated)
    cv2.imwrite("outputs/tree_species_clusters.png", species_annotated)

    # --- Result dictionary ---
    result = {
        "tree_count": tree_count,
        "area_km2": area_km2,
        "tree_density": density,
        "avg_tree_spacing": float(spacing),
        "forest_health_score": health_score,
        "ndvi_mean": float(np.mean(ndvi_raw)),
        "tree_clusters": list(named_labels),
        "biodiversity_index": biodiversity_index,
        "species_count": species_count
    }

    ai_report = generate_forest_report(result)
    result["analysis_report"] = ai_report if ai_report else None

    result["outputs"] = {
        "detections":"outputs/tree_detections.png",
        "density_heatmap":"outputs/tree_density_heatmap.png",
        "ndvi_map":"outputs/vegetation_health_map.png",
        "species_clusters":"outputs/tree_species_clusters.png"
    }

    return result
