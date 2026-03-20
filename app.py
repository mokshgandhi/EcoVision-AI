import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tree_module.analyze import analyze_forest

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🌿",
    layout="wide"
)

# ---------------- FOREST THEME CSS ---------------- #
st.markdown("""
<style>
*{
color:white;
}
.stApp {
    background:
    radial-gradient(circle at 20% 20%, #0f3d1e 0%, transparent 40%),
    radial-gradient(circle at 80% 30%, #14532d 0%, transparent 35%),
    radial-gradient(circle at 40% 80%, #064e3b 0%, transparent 35%),
    linear-gradient(180deg,#02140a,#041b0d,#052411,#03170d);
    color:white;
}

/* Upload box */
section[data-testid="stFileUploader"] {
    background: linear-gradient(135deg,#052e16,#064e3b);
    padding:35px;
    border-radius:18px;
    border:2px dashed #4ade80;
    box-shadow:0 12px 40px rgba(0,0,0,0.5);
}

/* Image cards */
.image-card {
    background: rgba(12,40,25,0.75);
    padding:12px;
    border-radius:16px;
    border:1px solid rgba(34,197,94,0.25);
    backdrop-filter: blur(10px);
    box-shadow:0 10px 30px rgba(0,0,0,0.35);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(16,46,28,0.85);
    border-radius:14px;
    padding:18px;
    border:1px solid rgba(34,197,94,0.3);
    box-shadow:0 8px 25px rgba(0,0,0,0.4);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#052411,#03170d);
    color: white;
}

/* Footer */
.footer {
    text-align:center;
    margin-top:40px;
    padding:12px;
    font-size:12px;
    opacity:0.6;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.title("EcoVision AI 🌿")
    st.caption("Forest Monitoring & Environmental Analytics")
    st.markdown("---")
    st.markdown("### Built By")
    st.write("**Team Shooting Stars**")
    st.markdown("---")
    st.markdown("### Features")
    st.write("• Tree Detection")
    st.write("• Density Heatmaps")
    st.write("• Vegetation Health (NDVI)")
    st.write("• Forest Analytics")
    st.markdown("---")
    st.info(
        "Upload satellite or drone images of forests to analyze vegetation and tree distribution."
    )

# ---------------- PAGE TITLE ---------------- #
st.title("Forest Analysis Dashboard")
st.write("")

# ---------------- FILE UPLOADER ---------------- #
uploaded_files = st.file_uploader(
    "Upload Forest Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# ---------------- PROCESS IMAGES ---------------- #
if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        with st.expander(f"Analysis for: {uploaded_file.name}", expanded=(i == 0)):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            temp_path = f"temp_{i}.jpg"
            cv2.imwrite(temp_path, image)

            with st.spinner("Running AI Forest Analysis..."):
                result = analyze_forest(temp_path)

            # ---------------- AI Detection Results ---------------- #
            st.subheader("AI Detection Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image("outputs/tree_detections.png", caption="Tree Detection")
            with col2:
                st.image("outputs/tree_density_heatmap.png", caption="Tree Density Heatmap")
            with col3:
                st.image("outputs/vegetation_health_map.png", caption="Vegetation Health (NDVI)")

            st.divider()

            # ---------------- Analytics Dashboard ---------------- #
            st.subheader("Forest Analytics Dashboard")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Tree Count", result["tree_count"])
            with c2:
                st.metric("Tree Density (trees/km²)", f'{result["tree_density"]:.2f}')
            with c3:
                st.metric("Forest Health Score", result["forest_health_score"])
            with c4:
                st.metric("Avg Tree Spacing", f'{result["avg_tree_spacing"]:.2f} m')

            st.divider()

            # ---------------- AI Report ---------------- #
            st.subheader("AI Forest Intelligence Report")
            if result.get("analysis_report"):
                st.success(result["analysis_report"])
            else:
                st.info(
                    "AI report generation is disabled. "
                    "Add your OpenAI API key to enable ecological insights."
                )

            # ---------------- Species Classification ---------------- #
            st.subheader("Species Classification")

            # Load images
            species_img = cv2.imread("outputs/tree_species_clusters.png")
            species_img = cv2.cvtColor(species_img, cv2.COLOR_BGR2RGB)
            orig_img = cv2.imread(temp_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            # Resize original image to match species-clustered image
            species_h, species_w, _ = species_img.shape
            orig_resized = cv2.resize(orig_img, (species_w, species_h))

            # Display side by side
            col_orig, col_species = st.columns(2)
            with col_orig:
                st.image(orig_resized, caption="Original Input Image (Resized)")
            with col_species:
                st.image(species_img, caption="Species / Cluster Visualization")

            # Species legend with counts
            species_counts = result.get("species_count", {})
            unique_species = list(species_counts.keys())
            n_clusters = len(unique_species)
            cmap = plt.get_cmap('tab10', n_clusters)

            legend_html = "<div style='display:flex; gap:15px; flex-wrap: wrap; margin-top:10px;'>"
            for i, species_name in enumerate(unique_species):
                count = species_counts[species_name]
                hex_color = result["labels"][species_name]
                print(hex_color)
                legend_html += (
                    f"<div style='display:flex; align-items:center; gap:5px;'>"
                    f"<div style='width:20px; height:20px; background: rgb{hex_color}; border:1px solid #000'></div>"
                    f"{species_name} ({count})"
                    f"</div>"
                )
            legend_html += "</div>"

            st.markdown(legend_html, unsafe_allow_html=True)

# ---------------- FOOTER CREDIT ---------------- #
st.markdown("""
<div class='footer'>
EcoVision AI • Created by <b>Team Shooting Stars</b>
</div>
""", unsafe_allow_html=True)
