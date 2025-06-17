import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer

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


    chosen_method = st.selectbox(
    "Which imputation method would you like to apply to your working dataset?",
    ["No Imputation", "Mean", "KNN"] + (["Clustering"] if df_cluster_imputed is not None else [])
)

    if st.button("Apply Selected Imputation to Working Dataset"):
        if chosen_method == "Mean":
            st.session_state['df_cleaned'] = df_mean
        elif chosen_method == "KNN":
            st.session_state['df_cleaned'] = df_knn
        elif chosen_method == "Clustering" and df_cluster_imputed is not None:
            st.session_state['df_cleaned'] = df_cluster_imputed
        st.success(f"{chosen_method} imputation applied to working dataset.")





def show_profiling(df, df_cleaned):
    st.markdown(f"**Dataset Dimensions:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.markdown("**Column Data Types:**")
    types_df = pd.DataFrame({'Column': df.dtypes.index, 'Type': df.dtypes.values})
    st.dataframe(types_df, height=150)

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        st.markdown("#### Want to preview how different imputation methods affect your data?")
        if st.button("Compare Imputation Methods (Mean, KNN, Clustering)"):
            # Pull clustering-based imputation from session if available
            df_cluster_imputed = st.session_state.get('df_cluster_imputed')
            show_imputation_comparison(df, df_cluster_imputed)

    
    
    
    if total_missing == 0:
        st.success("No missing values in the dataset.")
    else:
        st.markdown("**Missing Values by Column:**")
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': missing_pct})
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        st.dataframe(missing_df)

        # User can choose global or per-column imputation strategy
        impute_mode = st.radio(
            "How do you want to handle missing values?",
            ["Apply one method to all columns", "Customize per column"]
        )
        missing_cols = df.columns[df.isnull().any()]
        handling_options = {}

        if impute_mode == "Apply one method to all columns":
            methods = [
                'Drop rows with missing values',
                'Fill with mean',
                'Fill with median',
                'Fill with mode',
                'Fill with custom value',
                'KNN Imputation (numeric only)',
                'Drop entire column'
            ]
            method_all = st.selectbox("Select a method for all columns:", methods)
            custom_val = None
            confirm_drop = False
            if method_all == 'Fill with custom value':
                custom_val = st.text_input("Enter custom value to fill all columns:")
            if method_all == 'Drop entire column':
                confirm_drop = st.checkbox("⚠️ Confirm: Drop all columns with missing values")
            for col in missing_cols:
                handling_options[col] = (method_all, custom_val, confirm_drop)
        else:
            st.markdown("Customize imputation for each column:")
            for col in missing_cols:
                with st.expander(f"Options for `{col}`"):
                    dtype = df[col].dtype
                    options = [
                        'Drop rows with missing values',
                        'Fill with mean',
                        'Fill with median',
                        'Fill with mode',
                        'Fill with custom value',
                        'KNN Imputation (this column only)',
                        'Drop entire column'
                    ]
                    # Only show mean/median/KNN if numeric
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        options.remove('Fill with mean')
                        options.remove('Fill with median')
                        options.remove('KNN Imputation (this column only)')
                    selected = st.selectbox(f"Method for `{col}`", options, key=f"select_{col}")
                    custom_val = None
                    confirm_drop = False
                    if selected == 'Fill with custom value':
                        custom_val = st.text_input(f"Custom value for `{col}`", key=f"custom_{col}")
                    if selected == 'Drop entire column':
                        confirm_drop = st.checkbox(f"⚠️ Confirm: Drop `{col}`", key=f"dropcol_{col}")
                    handling_options[col] = (selected, custom_val, confirm_drop)

        if st.button("🚀 Apply Missing Value Handling"):
            df_handled = df_cleaned.copy()
            for col, (method, custom_val, confirm_drop) in handling_options.items():
                if method == 'Drop rows with missing values':
                    df_handled = df_handled[df_handled[col].notna()]
                elif method == 'Drop entire column':
                    if confirm_drop:
                        df_handled.drop(columns=[col], inplace=True)
                    else:
                        st.warning(f"Skipping `{col}` — Drop not confirmed.")
                elif method == 'Fill with mean':
                    df_handled[col].fillna(df_handled[col].mean(), inplace=True)
                elif method == 'Fill with median':
                    df_handled[col].fillna(df_handled[col].median(), inplace=True)
                elif method == 'Fill with mode':
                    mode = df_handled[col].mode()
                    if not mode.empty:
                        df_handled[col].fillna(mode[0], inplace=True)
                elif method == 'Fill with custom value':
                    if custom_val is not None:
                        try:
                            val_converted = df_handled[col].astype(df_handled[col].dropna().dtype).iloc[0].__class__(custom_val)
                        except:
                            val_converted = custom_val
                        df_handled[col].fillna(val_converted, inplace=True)
                elif method in ['KNN Imputation (numeric only)', 'KNN Imputation (this column only)']:
                    numeric_cols = df_handled.select_dtypes(include='number').columns
                    if col in numeric_cols:
                        imputer = KNNImputer(n_neighbors=5)
                        df_handled[numeric_cols] = imputer.fit_transform(df_handled[numeric_cols])
                    else:
                        st.warning(f"⚠️ KNN skipped for `{col}` (not numeric)")
            st.session_state['df_cleaned'] = df_handled
            st.success("✅ Missing value handling applied.")
            st.dataframe(df_handled.head())

    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe(include='all').T)
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        st.markdown("### Correlation Matrix (Numeric Features)")
        corr = df[numeric_cols].corr()
        st.dataframe(corr)

    # Descriptive statistics
    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe(include='all').T)  # transpose for readability
    
    # Correlation matrix for numeric columns (if more than one numeric column)
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        st.markdown("### Correlation Matrix (Numeric Features)")
        corr = df[numeric_cols].corr()
        st.dataframe(corr)
