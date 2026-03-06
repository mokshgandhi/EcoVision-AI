import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tree_module.analyze import analyze_forest
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

h1 {
    color: #22c55e;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background: #111827;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 EcoVision AI")
st.caption("AI Powered Forest Monitoring & Analytics")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Forest Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    cv2.imwrite("temp.jpg", image)

    with st.spinner("Running AI Forest Analysis..."):
        # Use modified analyze_forest (outputs colored cluster image)
        result = analyze_forest("temp.jpg", n_clusters=5)

    st.divider()

    st.subheader("AI Detection Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("outputs/tree_detections.png", caption="Tree Detection")

    with col2:
        st.image("outputs/tree_density_heatmap.png", caption="Tree Density Heatmap")

    with col3:
        st.image("outputs/vegetation_health_map.png", caption="Vegetation Health (NDVI)")

    st.divider()

    st.subheader("Forest Analytics Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🌳 Tree Count",
        result["tree_count"]
    )

    c2.metric(
        "📊 Tree Density",
        f'{result["tree_density"]:.2f} trees/km²'
    )

    c3.metric(
        "🌱 Forest Health Score",
        result["forest_health_score"]
    )

    c4.metric(
        "📏 Avg Tree Spacing",
        f'{result["avg_tree_spacing"]:.2f} m'
    )

    st.divider()

    st.subheader("Species Classification")

    # Load the cluster-colored image
    species_img = cv2.imread("outputs/tree_species_clusters.png")
    species_img = cv2.cvtColor(species_img, cv2.COLOR_BGR2RGB)
    st.image(species_img, caption="Species / Cluster Visualization")

    # Create a legend for cluster colors
    n_clusters = len(set(result["tree_clusters"]))
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    legend_html = "<div style='display:flex; gap:10px; flex-wrap: wrap;'>"
    for i in range(n_clusters):
        color = tuple(int(255*c) for c in cmap(i)[:3])
        hex_color = '#%02x%02x%02x' % color
        legend_html += f"<div style='display:flex; align-items:center; gap:5px;'><div style='width:20px; height:20px; background:{hex_color}; border:1px solid #000'></div>Species {i}</div>"
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)
