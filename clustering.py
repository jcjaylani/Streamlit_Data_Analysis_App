import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Attempt to import Hopkins statistic function
try:
    from pyclustertend import hopkins
except ImportError:
    hopkins = None

def show_clustering_analysis(df):
    # Use only numeric columns for clustering analysis
    numeric_data = df.select_dtypes(include='number').dropna()
    if numeric_data.shape[0] < 2 or numeric_data.shape[1] == 0:
        st.warning("Not enough numeric data to perform clustering analysis.")
        return
    
    # Standardize numeric data for clustering
    scaled = StandardScaler().fit_transform(numeric_data)
    # Allow user to select number of clusters k
    k = st.slider("Select number of clusters (K) for KMeans:", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaled)
    sil_score = silhouette_score(scaled, labels)
    
    # Calculate Hopkins statistic if available
    hop_score = None
    if hopkins:
        try:
            hop_score = hopkins(scaled, numeric_data.shape[0])
        except Exception:
            hop_score = None
    
    # Display clustering tendency metrics
    col1, col2 = st.columns(2)
    with col1:
        if hop_score is not None:
            st.metric("Hopkins Statistic", f"{hop_score:.4f}")
            if hop_score > 0.75:
                st.success("Strong clustering tendency (clusters likely present).")
            elif hop_score > 0.5:
                st.info("Moderate clustering tendency.")
            else:
                st.warning("Weak clustering tendency (data is fairly random).")
        else:
            st.info("Hopkins statistic not available (pyclustertend not installed).")
    with col2:
        st.metric("Silhouette Score", f"{sil_score:.4f}")
        if sil_score > 0.70:
            st.success("Clusters are well-separated.")
        elif sil_score > 0.50:
            st.info("Moderate cluster separation.")
        else:
            st.warning("Poor cluster separation.")
    
    # Visualize clusters in 2D using PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels.astype(str)
    st.markdown("### Cluster Visualization (PCA projection)")
    st.scatter_chart(pca_df, x='PC1', y='PC2', color='Cluster')
    
    # Offer clustering-based imputation if there are missing values in the data
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        st.markdown("### Clustering-Based Missing Value Imputation")
        st.write(f"⚠️ The dataset still has **{int(total_missing)}** missing values. You can apply clustering-based imputation:")
               
        
        
        
        if st.button(f"Apply Clustering-Based Imputation (k={k})"):
            df_clusters = df.copy()
            numeric_cols = df_clusters.select_dtypes(include='number').columns
            # Temporarily fill numeric missing values with mean for clustering
            df_temp = df_clusters[numeric_cols].copy()
            for col in numeric_cols:
                df_temp[col].fillna(df_temp[col].mean(), inplace=True)
            # Cluster on temporarily imputed data
            labels_all = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(StandardScaler().fit_transform(df_temp))
            df_clusters['__cluster'] = labels_all
            # Impute missing values within each cluster
            for col in df_clusters.columns:
                if df_clusters[col].isnull().sum() > 0:
                    for cluster_id in np.unique(labels_all):
                        cluster_mask = (df_clusters['__cluster'] == cluster_id)
                        # Gather non-missing values in this cluster for the column
                        values = df_clusters.loc[cluster_mask, col].dropna()
                        if values.empty:
                            continue
                        if pd.api.types.is_numeric_dtype(df_clusters[col]):
                            fill_val = values.median()
                        else:
                            fill_val = values.mode().iloc[0] if not values.mode().empty else None
                        df_clusters.loc[cluster_mask & df_clusters[col].isnull(), col] = fill_val
 
            
            # Remove the helper cluster column and update session state
            df_imputed = df_clusters.drop(columns=['__cluster'])
            st.session_state['df_cluster_imputed'] = df_imputed
            st.success('✅ Clustering-based imputation complete. Compare it in the profiling section!')
            st.dataframe(df_imputed.head())

def show_imputation_comparison(df, df_cluster_imputed=None):
    import streamlit as st
    from sklearn.impute import KNNImputer

    missing_cols = df.columns[df.isnull().any()]
    if len(missing_cols) == 0:
        st.info("No missing values in the dataset to compare imputation methods.")
        return

    df_mean = df.copy()
    df_knn = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    # 1. Mean Imputation
    for col in missing_cols:
        if df_mean[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_mean[col]):
                df_mean[col].fillna(df_mean[col].mean(), inplace=True)
            else:
                mode_val = df_mean[col].mode()
                if not mode_val.empty:
                    df_mean[col].fillna(mode_val[0], inplace=True)

    # 2. KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    df_knn[numeric_cols] = imputer.fit_transform(df_knn[numeric_cols])

    st.markdown("#### Comparing Imputation Methods (Mean vs KNN vs Clustering)")
    for col in missing_cols:
        st.write(f"**Column:** `{col}`")
        columns = ["Mean", "KNN", "Clustering"] if (df_cluster_imputed is not None and col in df_cluster_imputed.columns) else ["Mean", "KNN"]
        cols = st.columns(len(columns))
        with cols[0]:
            st.caption("Mean Imputed (first 5 rows)")
            st.dataframe(df_mean[[col]].head())
        with cols[1]:
            st.caption("KNN Imputed (first 5 rows)")
            st.dataframe(df_knn[[col]].head())
        if len(columns) == 3:
            with cols[2]:
                st.caption("Cluster Imputed (first 5 rows)")
                st.dataframe(df_cluster_imputed[[col]].head())
