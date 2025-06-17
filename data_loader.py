import pandas as pd
import io, json
import streamlit as st

# Attempt to import PDF parsing library
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

def load_file(uploaded_file):
    """
    Load an uploaded file into a pandas DataFrame.
    Supports CSV, Excel, JSON, and PDF (tables) formats.
    Returns: (df, error) tuple, where df is the DataFrame or None, and error is None or an error message.
    """
    extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif extension == 'json':
            data = json.load(uploaded_file)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # If dict values are all lists of equal length, convert to DataFrame directly; otherwise treat as single record
                if all(isinstance(v, list) for v in data.values()):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            else:
                return None, "Unsupported JSON structure."
        elif extension == 'pdf':
            if pdfplumber is None:
                return None, "PDF parsing requires the pdfplumber library. Please install it."
            # Extract tables from PDF using pdfplumber
            pdf = pdfplumber.open(io.BytesIO(uploaded_file.read()))
            tables = []
            for page in pdf.pages:
                for table in page.extract_tables():
                    if len(table) > 1:  # if table contains header + rows
                        try:
                            df_table = pd.DataFrame(table[1:], columns=table[0])
                        except Exception:
                            df_table = pd.DataFrame(table)
                        tables.append(df_table)
            pdf.close()
            if tables:
                df = pd.concat(tables, ignore_index=True)
            else:
                return None, "No tables found in PDF file."
        else:
            return None, "Unsupported file type."
        return df, None
    except Exception as e:
        return None, f"Error loading file: {e}"
