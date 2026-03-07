# EcoVision-AI

EcoVision-AI is an AI-powered forest health analytics platform that
analyzes aerial or satellite forest imagery to extract ecological
insights such as tree count, density, biodiversity, and environmental
risk indicators.

Built by **Team Shooting Stars**, EcoVision-AI uses computer vision,
deep learning, and ecological metrics to help monitor forests and
support environmental decision-making.

------------------------------------------------------------------------

# Overview

EcoVision-AI takes a **top-down image of a forest** (from satellite,
drone, or aerial imagery) and automatically analyzes it using AI models
to generate detailed forest health metrics and visualizations.

The system detects individual trees, analyzes vegetation health,
estimates biodiversity, and generates ecological insights.

------------------------------------------------------------------------

# Features

## Tree Detection

-   Detects trees using a trained **YOLO object detection model**
-   Draws bounding boxes around detected trees

## Forest Analytics

EcoVision-AI automatically calculates:

-   Tree Count
-   Tree Density (trees/km²)
-   Average Tree Spacing
-   Forest Health Score

## Vegetation Health Analysis

-   NDVI-based vegetation proxy analysis
-   Generates vegetation health heatmaps

## Species Detection

-   Uses **deep CNN embeddings (ResNet50)** to extract tree features
-   Clusters trees into species groups using **K-Means clustering**
-   Produces species-labeled visualizations

## Environmental Risk Indicators

EcoVision-AI provides insight into potential risks such as:

-   Deforestation risk
-   Wildfire risk
-   Biodiversity loss risk
-   Disease spread risk

## Visual Outputs

The system generates multiple visualizations:

-   Tree detection image
-   Tree density heatmap
-   Vegetation health map (NDVI proxy)
-   Species cluster visualization

## AI Ecological Report

An optional **AI-generated forest analysis report** provides:

-   Forest health assessment
-   Key ecological observations
-   Environmental risk indicators
-   Recommended management actions

------------------------------------------------------------------------

# Demo Dashboard

EcoVision-AI includes a **Streamlit web dashboard** where users can
upload images and view results interactively.

Dashboard features:

-   Image upload
-   Detection visualization
-   Forest analytics metrics
-   Biodiversity visualization
-   AI ecological report

------------------------------------------------------------------------

# System Architecture

Input Forest Image\
↓\
Image Preprocessing (Gamma correction + Green masking)\
↓\
Tree Detection (YOLO)\
↓\
Tree Filtering (Remove non-vegetation detections)\
↓\
Analytics Pipeline\
• Tree Count\
• Tree Density\
• Avg Tree Spacing\
• NDVI Proxy\
• Density Heatmap\
↓\
Species Detection \
↓\
Forest Health Metrics\
↓\
Visualization + AI Ecological Report

------------------------------------------------------------------------

# Technologies Used

## AI / Machine Learning

-   YOLO (Ultralytics) : RoboFlow (https://app.roboflow.com/moksh-gandhi/trees-vs3lw-hydtc/models/trees-vs3lw-hydtc/1)
-   Feature Extraction
-   Unsupervised (Agglomeritive Clustering)
  

## Data Science

-   NumPy
-   Pandas
-   Scikit-Learn
-   SciPy

## Computer Vision

-   OpenCV
-   PIL
-   TorchVision

## Visualization

-   Matplotlib

## Web Interface

-   Streamlit

## AI Report Generation

-   OpenAI API

------------------------------------------------------------------------

# Project Structure

```
EcoVision-AI
│
├── app.py                     # Streamlit dashboard
│
├── tree_module
│   ├── analyze.py             # Main forest analysis pipeline
│   ├── detector.py            # YOLO tree detection
│   ├── density.py             # Tree density calculations
│   ├── statistics.py          # Tree spacing metrics
│   ├── heatmap.py             # Density heatmap generation
│   ├── ndvi.py                # Vegetation health analysis
│   ├── visualization.py       # Detection visualization
│   ├── health.py              # Forest health scoring
│   └── report.py              # AI ecological report
│
├── models
│   └── tree_detector.pt       # Trained YOLO model
│
├── outputs
│   ├── tree_detections.png
│   ├── tree_density_heatmap.png
│   ├── vegetation_health_map.png
│   └── tree_species_clusters.png
│
└── README.md
```

------------------------------------------------------------------------

# Installation

## 1. Clone the repository

git clone https://github.com/mokshgandhi/EcoVision-AI cd
EcoVision-AI

## 2. Install dependencies

pip install -r requirements.txt

Example dependencies:

torch torchvision opencv-python numpy pandas scikit-learn scipy
matplotlib streamlit ultralytics openai Pillow

## 3. Add the trained model

Place the trained YOLO model in:

models/tree_detector.pt

## 4. (Optional) Enable AI ecological reports

Add your OpenAI API key in:

tree_module/report.py

api_key = "YOUR_OPENAI_API_KEY"

------------------------------------------------------------------------

# Running the Application

Launch the Streamlit dashboard:

streamlit run app.py

Then open in browser:

http://localhost:8501

Upload a **satellite or drone forest image** to begin analysis.

------------------------------------------------------------------------

# Example Outputs

EcoVision-AI generates:

* Tree detection maps
* Density heatmaps
* Vegetation health maps
* Species cluster visualization
* Forest analytics dashboard
* AI ecological report

------------------------------------------------------------------------

# Potential Applications

-   Forest conservation
-   Biodiversity monitoring
-   Deforestation detection
-   Climate change research
-   Wildlife habitat assessment
-   Environmental risk analysis

------------------------------------------------------------------------

# Team

Developed by:

**Team Shooting Stars**
* Team Leader : Yashvi Dalsaniya
* Team Member : Tirth Tandel
* Team Member : Moksh Gandhi

Project: **EcoVision-AI**

