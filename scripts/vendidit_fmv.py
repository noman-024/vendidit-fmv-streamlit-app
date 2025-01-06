import pandas as pd
import requests
import logging
import os
from concurrent.futures import ThreadPoolExecutor

# API endpoint
API_URL = "http://ai-lb-776202452.us-east-2.elb.amazonaws.com/predict/fmv/"

# File paths
INPUT_FILE = "data/output/reconext_product list_bid request_output.xlsx"  # Change to your input file path
OUTPUT_FILE = "data/output/reconext_product list_bid request_output_V2.xlsx"
LOG_FILE = "/home/noman/vendidit-globalsku/data/logs/logs.txt"

# Mapping of conditions to column names
CONDITION_MAP = {
    6: "vendidit_fmv_new",
    5: "vendidit_fmv_open_box",
    4: "vendidit_fmv_excellent_refurbished",
    3: "vendidit_fmv_very_good_refurbished",
    2: "vendidit_fmv_good_refurbished",
    1: "vendidit_fmv_used"
}

# CONDITION_MAP = {
#     6: "vendidit_fmv_new",
#     5: "vendidit_fmv_like_new",
#     4: "vendidit_fmv_very_good",
#     3: "vendidit_fmv_good",
#     2: "vendidit_fmv_acceptable",
#     1: "vendidit_fmv_unacceptable"
# }

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def call_api(condition_weight, market_price):
    """
    Calls the Vendidit FMV API with the given condition weight and market price.

    Args:
        condition_weight: The condition weight of the device (1-6).
        market_price: The market price of the device.

    Returns:
        Predicted price or None if API call fails.
    """
    params = {"condition_weight": condition_weight, "MarketPrice": market_price}
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("predicted_used_phone_price")
    except requests.RequestException as e:
        logging.error(f"API request failed for {params}: {e}")
        return None


def read_file(file_path):
    """
    Reads input data from a CSV or Excel file based on the file extension.

    Args:
        file_path: Path to the input file.

    Returns:
        A pandas DataFrame containing the file data.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.xlsx':
        return pd.read_excel(file_path)
    elif file_extension == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def save_file(df, file_path):
    """
    Saves the DataFrame to a CSV or Excel file based on the file extension.

    Args:
        df: The pandas DataFrame to save.
        file_path: Path to save the file.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.xlsx':
        df.to_excel(file_path, index=False)
    elif file_extension == '.csv':
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_row_conditions(index, row, market_price):
    """
    Process all condition weights for a single row in parallel.

    Args:
        index (int): Index of the row.
        row (pd.Series): A single row of the DataFrame.
        market_price (float): Market price for the product.

    Returns:
        dict: Predicted prices for all conditions.
    """
    logging.info(f"Processing row index: {index} with market price: {market_price}")
    results = {}
    for condition, column_name in CONDITION_MAP.items():
        logging.info(f"Processing condition {condition} for row index: {index}")
        try:
            predicted_price = call_api(condition, market_price)
        except Exception as e:
            logging.error(f"Error processing condition {condition} for row index {index}: {e}")
            predicted_price = None
        results[column_name] = predicted_price
    return results


def process_vendidit_file(input_file, output_file):
    """
    Process the file using the Vendidit FMV API logic with parallelization.

    Args:
        input_file: Path to the input file.
        output_file: Path to save the updated file.
    """
    logging.info(f"Starting Vendidit FMV processing for file: {input_file}")

    # Load the input file dynamically
    try:
        file_extension = os.path.splitext(input_file)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(input_file)
        elif file_extension == ".xlsx":
            df = pd.read_excel(input_file, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported input file format: {file_extension}")

        # Add new columns for condition-based predictions
        for col in CONDITION_MAP.values():
            df[col] = None

        # Parallelize row processing
        max_workers = max(1, os.cpu_count())  # Use half the CPU count
        logging.info(f"Using {max_workers} workers for Vendidit FMV processing.")

        def process_task(row_data):
            index, row = row_data
            market_price = row.get("new_price")
            if pd.isna(market_price):
                logging.warning(f"Skipping row index: {index} due to missing market price.")
                return {col: None for col in CONDITION_MAP.values()}
            return process_row_conditions(index, row, market_price)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_task, df.iterrows()))
        except Exception as e:
            logging.error(f"Error during parallel processing: {e}")
            return

        # Update DataFrame with results
        for idx, row_results in enumerate(results):
            for column_name, value in row_results.items():
                df.at[idx, column_name] = value

        # Save the processed file
        if file_extension == ".csv":
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
        elif file_extension == ".xlsx":
            df.to_excel(output_file, index=False, engine="openpyxl")
        logging.info(f"Processed file saved at: {output_file}")

    except Exception as e:
        logging.error(f"Error in Vendidit FMV processing: {e}")
