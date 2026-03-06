import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from .detector import TreeDetector
from .density import calculate_density
from .statistics import compute_spacing
from .heatmap import generate_density_heatmap
from .ndvi import compute_ndvi_proxy
from .visualization import draw_detections
from .health import compute_forest_health
from .report import generate_forest_report


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

    square_img = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return square_img


def gamma_correction(img, gamma=1.2):

    invGamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(img, table)


def green_mask(image,
               lower_hsv=(30, 40, 40),
               upper_hsv=(95, 255, 255)):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv),
        np.array(upper_hsv)
    )

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


# -------------------- Post-detection filtering --------------------

def filter_non_green_trees(image, predictions, green_thresh=50):

    keep_idx = []

    for i, row in predictions.iterrows():

        xmin, ymin, xmax, ymax = map(
            int,
            [row.xmin, row.ymin, row.xmax, row.ymax]
        )

        crop = image[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            continue

        avg_green = np.mean(crop[:, :, 1])

        if avg_green > green_thresh:
            keep_idx.append(i)

    return predictions.iloc[keep_idx].reset_index(drop=True)


# -------------------- Initialize Models --------------------

detector = TreeDetector()

weights = ResNet50_Weights.DEFAULT

cnn_model = resnet50(weights=weights)

cnn_model = torch.nn.Sequential(
    *list(cnn_model.children())[:-1]
)

cnn_model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------- Helper Functions --------------------

def crop_and_save_trees(image, predictions, output_dir="tree_crops"):

    os.makedirs(output_dir, exist_ok=True)

    crop_paths = []

    for i, row in predictions.iterrows():

        xmin, ymin, xmax, ymax = map(
            int,
            [row.xmin, row.ymin, row.xmax, row.ymax]
        )

        crop = image[ymin:ymax, xmin:xmax]

        crop_path = os.path.join(
            output_dir,
            f"tree_{i}.png"
        )

        cv2.imwrite(crop_path, crop)

        crop_paths.append(crop_path)

    return crop_paths


def get_embeddings(crop_paths):

    embeddings = []

    for path in crop_paths:

        img = Image.open(path).convert("RGB")

        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():

            feat = cnn_model(tensor).squeeze().numpy()

        embeddings.append(feat)

    return np.array(embeddings)


def cluster_trees(embeddings, n_clusters=5):

    if len(embeddings) == 0:
        return []

    n_clusters = min(n_clusters, len(embeddings))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = kmeans.fit_predict(embeddings)

    return labels


def annotate_species_clusters(image, predictions, labels, n_clusters):

    if len(labels) == 0:
        return image.copy()

    colors = [
        tuple(int(c * 255) for c in plt.cm.tab10(i)[:3])
        for i in range(n_clusters)
    ]

    annotated = image.copy()

    for i, row in predictions.iterrows():

        xmin, ymin, xmax, ymax = map(
            int,
            [row.xmin, row.ymin, row.xmax, row.ymax]
        )

        color = colors[labels[i] % n_clusters]

        cv2.rectangle(
            annotated,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2
        )

        cv2.putText(
            annotated,
            f"C{labels[i]}",
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return annotated


# -------------------- Main Pipeline --------------------

def analyze_forest(image_path, resolution=0.5, n_clusters=5):

    image, temp_path = preprocess_image(image_path)

    height, width, _ = image.shape

    predictions = detector.detect(temp_path)

    predictions = filter_non_green_trees(image, predictions)

    tree_count = len(predictions)

    area_km2, density = calculate_density(
        tree_count,
        width,
        height,
        resolution
    )

    spacing = compute_spacing(predictions)

    heatmap = generate_density_heatmap(image, predictions)

    ndvi_map, ndvi_raw = compute_ndvi_proxy(image)

    annotated = draw_detections(image, predictions)

    health_score = compute_forest_health(ndvi_raw, density)

    crop_paths = crop_and_save_trees(image, predictions)

    embeddings = get_embeddings(crop_paths)

    labels = cluster_trees(embeddings, n_clusters)

    species_annotated = annotate_species_clusters(
        image,
        predictions,
        labels,
        min(n_clusters, len(labels))
    )

    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite("outputs/tree_density_heatmap.png", heatmap)
    cv2.imwrite("outputs/vegetation_health_map.png", ndvi_map)
    cv2.imwrite("outputs/tree_detections.png", annotated)
    cv2.imwrite("outputs/tree_species_clusters.png", species_annotated)

    # ---------------- Metrics dictionary ----------------

    result = {

        "tree_count": tree_count,
        "area_km2": area_km2,
        "tree_density": density,
        "avg_tree_spacing": float(spacing),
        "forest_health_score": health_score,
        "ndvi_mean": float(np.mean(ndvi_raw)),
        "tree_clusters": list(labels)
    }

    # ---------------- AI ecological report ----------------

    ai_report = generate_forest_report(result)

    if ai_report:
        result["analysis_report"] = ai_report
    else:
        result["analysis_report"] = None

    result["outputs"] = {

        "detections": "outputs/tree_detections.png",
        "density_heatmap": "outputs/tree_density_heatmap.png",
        "ndvi_map": "outputs/vegetation_health_map.png",
        "species_clusters": "outputs/tree_species_clusters.png"

    }

    return result