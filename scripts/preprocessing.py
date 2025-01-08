import json
from openai import OpenAI
# from dotenv import load_dotenv
import os
import logging
import pandas as pd
import chardet
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")


def detect_file_encoding(file_path):
    """
    Detect the encoding of a file using chardet.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Detected file encoding.
    """
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result["encoding"]
    

# Unified log file path
log_file = os.path.abspath("data/logs/log.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# def generate_dynamic_columns(df, max_rows=5):
#     """
#     Use OpenAI to analyze the input dataframe and suggest columns for concatenation.

#     Args:
#         df (pd.DataFrame): The input dataframe.
#         max_rows (int): Number of sample rows to include in the analysis.

#     Returns:
#         list: Suggested columns for concatenation.
#     """
#     logging.info("Starting column suggestion using OpenAI.")
#     logging.info(f"DataFrame Columns: {df.columns.tolist()}")
    
#     # Prepare a sample of the data
#     sample_data = df.head(max_rows).to_dict(orient="records")
    
#     # Convert Timestamp objects to strings
#     for row in sample_data:
#         for key, value in row.items():
#             if isinstance(value, pd.Timestamp):
#                 row[key] = value.isoformat()  # Convert to ISO 8601 string format
    
#     logging.info(f"Sample Data: {sample_data}")

#     # Initialize OpenAI client
#     client = OpenAI(api_key=openai_api_key)

#     # Prepare the prompt
#     column_names = df.columns.tolist()
#     # print(f"column_names: {column_names}")
#     sample_data = df.head(max_rows).to_dict(orient="records")
#     prompt = (
#         f"You are an expert in analyzing product datasets for e-commerce platforms like Amazon and eBay.\n\n"
#         f"### Column Names ###\n{column_names}\n\n"
#         f"### Sample Data ###\n{json.dumps(sample_data, indent=2)}\n\n"
#         f"### Task ###\n"
#         f"Identify and suggest the most relevant columns to concatenate into a single query column that describes "
#         f"the product completely and accurately. Ensure to include:\n"
#         f"- Brand or make names.\n"
#         f"- Product models and titles.\n"
#         f"- Numerical features such as RAM, storage, or size that add specificity to the query.\n\n"
#         f"- If multiple columns represent the same information, then select only one column name among them. \n\n"
#         f"Your response should strictly follow this format:\n[\"column1\", \"column2\", \"column3\"]."
#     )

#     try:
#         # Call the OpenAI API
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             timeout=120  # Increase the timeout to 120 seconds
#         )
#         logging.info(f"OpenAI response: {response}")

#         # Parse the response
#         suggested_columns = json.loads(response.choices[0].message.content.strip())
#         if not isinstance(suggested_columns, list):
#             raise ValueError("Response is not a valid JSON array.")
#         return suggested_columns
#     except Exception as e:
#         logging.error(f"Error in OpenAI response: {e}")
#         raise ValueError(f"Error decoding OpenAI response: {e}")

def generate_dynamic_columns(df, max_rows=5):
    """
    Use OpenAI to analyze the input dataframe and suggest columns for concatenation.

    Args:
        df (pd.DataFrame): The input dataframe.
        max_rows (int): Number of sample rows to include in the analysis.

    Returns:
        list: Suggested columns for concatenation.
    """
    logging.info("Starting column suggestion using OpenAI.")
    logging.info(f"DataFrame Columns: {df.columns.tolist()}")

    # Prepare a sample of the data
    sample_data = df.head(max_rows).to_dict(orient="records")

    # Convert Timestamp objects to strings
    for row in sample_data:
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                row[key] = value.isoformat()  # Convert to ISO 8601 string format

    logging.info(f"Sample Data: {sample_data}")

    # Prepare the OpenAI prompt
    column_names = df.columns.tolist()
    prompt = (
        f"You are an expert in analyzing product datasets for e-commerce platforms like Amazon and eBay.\n\n"
        f"### Column Names ###\n{column_names}\n\n"
        f"### Sample Data ###\n{json.dumps(sample_data, indent=2)}\n\n"
        f"### Task ###\n"
        f"Identify and suggest the most relevant columns to concatenate into a single query column that describes "
        f"the product completely and accurately. Ensure to include:\n"
        f"- Brand or make names.\n"
        f"- Product models and titles.\n"
        f"- Numerical features such as RAM, storage, or size that add specificity to the query.\n\n"
        f"Your response should strictly follow this format:\n[\"column1\", \"column2\", \"column3\"]."
    )

    # Call the OpenAI API
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            timeout=120  # Increase timeout if needed
        )
        logging.info(f"OpenAI response: {response}")

        # Parse the response
        suggested_columns = json.loads(response.choices[0].message.content.strip())
        if not isinstance(suggested_columns, list):
            raise ValueError("Response is not a valid JSON array.")
        return suggested_columns
    except Exception as e:
        logging.error(f"Error in OpenAI response: {e}")
        raise ValueError(f"Error generating dynamic columns: {e}")


def clean_excel_data(df):
    """
    Clean potentially corrupted cells in an Excel DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: str(x).encode("utf-8", "ignore").decode("utf-8") if isinstance(x, str) else x)
    return df


def clean_and_concat(row, suggested_columns):
    """
    Clean and concatenate values in a row to form a query column.

    Args:
        row (pd.Series): A single row of the DataFrame.
        suggested_columns (list): List of columns to use for concatenation.

    Returns:
        str: Cleaned and concatenated query string.
    """
    invalid_values = {"nan", "0", "unknown", "none", "null", "n/a", ""}
    cleaned_values = [
        str(row[col]).strip().lower()
        for col in suggested_columns
        if pd.notna(row[col]) and str(row[col]).strip().lower() not in invalid_values
    ]
    unique_values = list(dict.fromkeys(cleaned_values))
    query = " ".join(unique_values).strip()
    return query.capitalize() if query else "Unknown Product"

def preprocess_file_with_query(input_path):
    """
    Preprocess the input file to generate a query column based on OpenAI's suggestions.

    Args:
        input_path (str): Path to the input file.

    Returns:
        str: Path to the preprocessed file with a new query column.
    """
    logging.info(f"Starting preprocessing for file: {input_path}")

    # Determine file type and encoding
    file_extension = os.path.splitext(input_path)[1].lower()
    encoding = None

    try:
        if file_extension == ".csv":
            encoding = detect_file_encoding(input_path)
            logging.info(f"Detected file encoding: {encoding}")
            df = pd.read_csv(input_path, encoding=encoding)
        elif file_extension == ".xlsx":
            df = pd.read_excel(input_path, engine="openpyxl")
            df = clean_excel_data(df)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        logging.info("File loaded successfully.")
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise

    # Generate dynamic column suggestions
    try:
        suggested_columns = generate_dynamic_columns(df)
        logging.info(f"Suggested columns: {suggested_columns}")
    except Exception as e:
        logging.error(f"Error generating dynamic columns: {e}")
        raise

    # Parallelize the query column creation
    max_workers = max(1, os.cpu_count())  # Use half the CPU count
    logging.info(f"Using {max_workers} workers for parallel preprocessing.")

    try:
        def process_row(row):
            logging.info(f"Processing row index: {row[0]}")
            return clean_and_concat(row[1], suggested_columns)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            query_results = list(executor.map(process_row, df.iterrows()))

        df["query_col"] = query_results
        logging.info("Query column created successfully.")
    except Exception as e:
        logging.error(f"Error creating query column: {e}")
        raise

    # Save the preprocessed file
    output_path = os.path.join("data/input", "preprocessed_" + os.path.basename(input_path))
    try:
        if file_extension == ".csv":
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif file_extension == ".xlsx":
            df.to_excel(output_path, index=False, engine="openpyxl")
        logging.info(f"Preprocessed file saved at: {output_path}")
    except Exception as e:
        logging.error(f"Error saving preprocessed file: {e}")
        raise

    return output_path