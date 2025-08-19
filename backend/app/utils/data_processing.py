import pandas as pd
import camelot

def load_dataset(file_path: str):
    """Load dataset from CSV, Excel, or PDF"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    elif file_path.endswith('.pdf'):
        tables = camelot.read_pdf(file_path, pages='all')
        return pd.concat([table.df for table in tables], ignore_index=True)
    else:
        raise ValueError("Unsupported file format")

def anonymize_data(df: pd.DataFrame, columns_to_anonymize: list[str]):
    """Anonymize the given columns in the DataFrame."""
    for col in columns_to_anonymize:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"anon_{hash(x) % 10000}")
    return df
