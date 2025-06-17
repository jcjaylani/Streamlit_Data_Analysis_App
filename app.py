import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration for the Streamlit app
st.set_page_config(page_title="Modular Data Explorer", layout="wide")


# Custom CSS to style the logo and center the title
st.markdown("""
    <style>
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: -50px; /* pull logo closer to the top */
            margin-bottom: 10px;
        }
        .centered-title {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Define a helper function for displaying the logo
def display_logo(image_path):
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(image_path, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Define the main tabs of the interface
tabs = st.tabs([
    "Upload & Preview",
    "Data Profiling",
    "Clustering",
    "Imputation Comparison",
    "Outliers",
    "ML Automation",
    "AI Review"
])


# Import modules for each component
import data_loader
import profiling
import clustering
import outliers
import ml_pipeline
import ai_review

# 1. Upload & Preview Tab
with tabs[0]:
    display_logo("Data_Image_1.png")
    st.markdown('<h1 class="centered-title">DataCortex</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="centered-title">An App for Data Scientists</h2>', unsafe_allow_html=True)

with tabs[0]:
    st.subheader("Upload & Preview")
    uploaded_file = st.file_uploader(
        "Upload a data file (CSV, Excel, JSON, or PDF)", 
        type=["csv", "xls", "xlsx", "json", "pdf"]
    )
    if uploaded_file:
        df, error = data_loader.load_file(uploaded_file)
        if error:
            st.error(error)
        else:
            # Store original and a copy for cleaning in session state
            st.session_state['df'] = df
            st.session_state['df_cleaned'] = df.copy()
            st.success(f"File loaded successfully! Shape: {df.shape}")
            st.write("First 5 rows of the dataset:")
            st.dataframe(df.head())
    else:
        st.info("Please upload a data file to get started.")

# 2. Data Profiling & Cleaning Tab
with tabs[1]:
    st.subheader("Data Profiling & Cleaning")
    if 'df' not in st.session_state:
        st.info("Upload a dataset first to see profiling information.")
    else:
        # Show profiling info and provide missing-value handling options
        profiling.show_profiling(st.session_state['df'], st.session_state['df_cleaned'])

# 3. Clustering Analysis Tab
with tabs[2]:
    st.subheader("Clustering Analysis")
    if 'df_cleaned' not in st.session_state:
        st.info("Upload and prepare a dataset to perform clustering.")
    else:
        clustering.show_clustering_analysis(st.session_state['df_cleaned'])

# 4. Imputation Comparison Tab
with tabs[3]:
    st.subheader("Imputation Method Comparison")
    if 'df' not in st.session_state:
        st.info("Please upload a dataset to compare imputation methods.")
    else:
        # Compare Mean vs KNN vs Clustering imputation (requires cluster-based impute to be done)
        clustering.show_imputation_comparison(st.session_state['df'], st.session_state.get('df_cleaned'))

# 5. Outlier Detection & Handling Tab
with tabs[4]:
    st.subheader("Outlier Detection & Handling")
    if 'df_cleaned' not in st.session_state:
        st.info("Please upload and clean a dataset to handle outliers.")
    else:
        outliers.show_outlier_handling(st.session_state['df_cleaned'])

# 6. ML Automation & Tuning Tab
with tabs[5]:
    st.subheader("ML Automation & Model Tuning")
    if 'df_cleaned' not in st.session_state:
        st.info("Please upload and prepare a dataset first.")
    else:
        ml_pipeline.show_ml_pipeline(st.session_state['df_cleaned'])

# 7. AI Review Tab
with tabs[6]:
    st.subheader("AI-Powered Data & Model Review")
    if 'df' not in st.session_state:
        st.info("Please upload a dataset to generate AI insights.")
    else:
        ai_review.show_ai_review(st.session_state['df'], st.session_state.get('ml_results'))
