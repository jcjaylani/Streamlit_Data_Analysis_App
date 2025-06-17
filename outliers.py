import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_outlier_handling(df):
    numeric_cols = df.select_dtypes(include='number').columns
    if numeric_cols.empty:
        st.info("No numeric columns to check for outliers.")
        return
    
    # Calculate IQR bounds and outlier counts for each numeric column
    summary_data = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct_outliers = (n_outliers / len(df)) * 100
        summary_data.append({
            "Column": col,
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "# Outliers": int(n_outliers),
            "% Outliers": f"{pct_outliers:.2f}%"
        })
    summary_df = pd.DataFrame(summary_data)
    st.markdown("**Outlier Summary:**")
    st.dataframe(summary_df)
    
    # Choose handling mode (global or per-column)
    mode = st.radio("How would you like to handle outliers?", 
                    ["Apply one method to all numeric columns", "Customize per column"])
    outlier_actions = {}
    if mode == "Apply one method to all numeric columns":
        method = st.selectbox("Global outlier handling method:", 
                              ["Do nothing", "Drop outliers", "Cap outliers to IQR bounds"])
        for col in numeric_cols:
            outlier_actions[col] = method
    else:
        st.markdown("Customize outlier handling per column:")
        for col in numeric_cols:
            with st.expander(f"Column: {col}"):
                # Show a boxplot for visualizing outliers
                fig, ax = plt.subplots()
                sns.boxplot(y=df[col], ax=ax)
                ax.set_title(f"{col} Distribution")
                st.pyplot(fig)
                action = st.selectbox(f"Handling for `{col}`:", 
                                       ["Do nothing", "Drop outliers", "Cap outliers to IQR bounds"], 
                                       key=f"outlier_{col}")
                outlier_actions[col] = action
    
    if st.button("Apply Outlier Handling"):
        df_result = df.copy()
        report = []
        for col, action in outlier_actions.items():
            if action == "Drop outliers":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                before_count = len(df_result)
                # Drop rows outside the IQR bounds for this column
                df_result = df_result[(df_result[col] >= lower) & (df_result[col] <= upper)]
                removed = before_count - len(df_result)
                report.append({"Column": col, "Method": "Dropped", "Rows Removed": removed})
            elif action == "Cap outliers to IQR bounds":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                before_vals = df_result[col].copy()
                # Cap values outside IQR bounds
                df_result[col] = df_result[col].clip(lower=lower, upper=upper)
                changed = (before_vals != df_result[col]).sum()
                report.append({"Column": col, "Method": "Capped", "Values Capped": int(changed)})
            else:
                report.append({"Column": col, "Method": "None", "Values Affected": 0})
        
        # Update the cleaned dataframe in session state
        st.session_state['df_cleaned'] = df_result
        st.success("Outlier handling applied.")
        # Display a summary of actions taken
        report_df = pd.DataFrame(report)
        st.markdown("**Outlier Handling Summary:**")
        st.dataframe(report_df)
