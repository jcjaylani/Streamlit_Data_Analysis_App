import streamlit as st
import pandas as pd

# Note: The openai package is required for this module
try:
    import openai
except ImportError:
    openai = None

def show_ai_review(df, ml_results=None):
    # Prompt for OpenAI API key
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if not api_key:
        st.info("🔑 Enter your OpenAI API key to generate AI insights.")
        return
    if openai is None:
        st.error("The OpenAI package is not installed. Please install `openai` to use this feature.")
        return
    openai.api_key = api_key
    
    # Compose a prompt with dataset insights and model results
    num_rows, num_cols = df.shape
    prompt = f"The dataset contains {num_rows} rows and {num_cols} columns.\\n"
    prompt += "Columns and types: " + ", ".join([f"{col} ({str(dtype)})" for col, dtype in zip(df.columns, df.dtypes)]) + ".\\n"
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        missing_info = df.isnull().sum().sort_values(ascending=False)
        top_missing_col = missing_info.index[0]
        top_missing_count = int(missing_info.iloc[0])
        prompt += f"There are {int(total_missing)} missing values in total, with '{top_missing_col}' having the most ({top_missing_count} missing).\\n"
    else:
        prompt += "There are no missing values in the dataset.\\n"
    # If numeric features exist, mention highest correlation pair
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        # Set self-correlations to 0 to find max off-diagonal
        for c in corr_matrix.columns:
            corr_matrix.loc[c, c] = 0.0
        max_corr = corr_matrix.max().max()
        if max_corr >= 0.5:
            # Identify the pair with max correlation
            max_pair = corr_matrix.stack().idxmax()
            prompt += f"The strongest correlation is {max_corr:.2f} between '{max_pair[0]}' and '{max_pair[1]}'.\\n"
    # Include ML results if available
    if ml_results:
        task = ml_results.get('task')
        if task == 'classification':
            best_model = ml_results.get('best_model')
            acc = ml_results.get('accuracy')
            prompt += f"The best classification model was {best_model} with an accuracy of {acc:.2%}.\\n"
        elif task == 'regression':
            best_model = ml_results.get('best_model')
            r2 = ml_results.get('r2_score')
            prompt += f"The best regression model was {best_model} with an R^2 score of {r2:.2f}.\\n"
        elif task == 'clustering':
            best_k = ml_results.get('best_k')
            sil = ml_results.get('silhouette')
            prompt += f"Clustering analysis suggests {best_k} clusters (Silhouette Score = {sil:.2f}).\\n"
        # List important features if identified
        if ml_results.get('top_features'):
            feature_names = [feat for feat, val in ml_results['top_features']]
            prompt += "Important features identified: " + ", ".join(feature_names) + ".\\n"
    prompt += "Provide a comprehensive analysis of the dataset and the above results."
    
    # Generate AI review when button is clicked
    if st.button("Generate AI Review"):
        with st.spinner("Generating AI review..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_text = response['choices'][0]['message']['content']
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
                return
        st.markdown("### AI-Generated Insights")
        st.write(ai_text)
        # Download buttons for the summary text/PDF
        st.download_button("Download Summary (.txt)", ai_text, file_name="AI_Summary.txt")
        
        try:
            from fpdf import FPDF

            # Use a basic built-in font for safety
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("helvetica", size=12)
            # Replace problematic characters
            safe_text = ai_text.encode('latin-1', 'replace').decode('latin-1')
            for line in safe_text.split('\n'):
                pdf.multi_cell(0, 10, line)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download Summary (.pdf)", data=pdf_bytes, file_name="AI_Summary.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}. But you can download the text file above.")
