import os
import json
import pandas as pd
# from dotenv import load_dotenv
from openai import OpenAI
from difflib import get_close_matches
import logging
import time
from concurrent.futures import ThreadPoolExecutor


# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

"""
Author: Md Mutasim Billah Noman
Date: 27/11/2024
"""

# Configure logging
log_file = "data/logs/log.log"
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize OpenAI client
def generate_with_openai(prompt, model="gpt-4o-mini"):
    """
    Call the OpenAI API to generate a response based on a given prompt.

    Args:
        prompt (str): The prompt to generate a response for.
        model (str, optional): The model to use for generation. Defaults to "gpt-4o-mini".

    Returns:
        dict: The generated JSON response.
    """
    client = OpenAI(api_key=openai_api_key)
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_response = completion.choices[0].message.content.strip()
        
        # Save the raw response to a text file
        with open("data/logs/openai_responses.txt", "a") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {raw_response}\n")
            f.write("=" * 80 + "\n")  # Separator for readability

        elapsed_time = time.time() - start_time
        logging.info(f"API call took {elapsed_time:.2f} seconds.")

        # Parse and return the JSON response
        return json.loads(raw_response)
    except Exception as e:
        logging.error(f"Error while generating response: {e}")
        return None


def generate_new_price_prompt(description):
    """
    Generate a prompt for predicting new price.

    Args:
        description (str): Product description.

    Returns:
        str: The generated prompt.
    """
    return (
        f"""You are an expert in pricing new products in the USA.
        ### Task ###
        Search for the market price (in USD) of the product described below. Use Amazon and Ebay as pricing source. Ensure the product is in 'New' condition (no refurbished, used, or open box items).

        ### Input ###
        Product Description: '{description}'

        ### Output Requirements ###
        Return the result in the following JSON format:
        {{
            "price": "<price_without_currency>",
            "condition": "New",
            "source-url": "<product_page_url>",
            "currency": "<currency>",
            "specifications": "<specifications>"
        }}
        Ensure the response follows this exact format without any additional tags or symbols, such as ```json or other formatting markers.
        """
    )


def generate_used_price_prompt(description, new_price, currency):
    """
    Generate a prompt for predicting used price (FMV).

    Args:
        description (str): Product description.
        new_price (str): Predicted new price of the product.
        currency (str): Currency of the new price.

    Returns:
        str: The generated prompt.
    """
    return (
        f"""You are an expert in evaluating used product prices in the USA, specializing in analyzing patterns from Amazon and eBay.
        ### Task ###
        Estimate the Fair Market Valuation (FMV) for a product in 'Used' condition by analyzing similar products on Amazon and eBay.

        ### Input ###
        - Product Description: '{description}'
        - New Price: '{new_price} {currency}'

        ### Pattern Analysis ###
        Analyze pricing patterns for similar products on Amazon and eBay:
        - Depreciation trends based on condition and age.
        - Brand-specific pricing retention patterns.

        ### Output Requirements ###
        Return the result in the following JSON format:
        {{
            "used_price": "<price_without_currency>",
            "currency": "<currency>",
            "source-url": "<product_page_url>",
            "explanation": "<explanation of how the price was calculated>"
        }}
        Ensure the response follows this exact format without any additional tags or symbols, such as ```json or other formatting markers.
        """
    )


def find_best_column(columns, candidates):
    """
    Dynamically identify the best matching column from a list of candidates.

    Args:
        columns (list): List of dataframe column names.
        candidates (list): List of candidate column names to match.

    Returns:
        str or None: The best matching column name, or None if no match is found.
    """
    normalized_columns = [str(col).strip().lower() for col in columns]
    normalized_candidates = [cand.strip().lower() for cand in candidates]

    matches = get_close_matches(normalized_candidates[0], normalized_columns, n=1, cutoff=0.5)
    logging.info(f"Closest match for column: {matches}")
    if matches:
        return columns[normalized_columns.index(matches[0])]
    return None

def process_row(index, row, description_column):
    """
    Process a single row to predict new and used prices using OpenAI API.

    Args:
        index (int): Index of the row.
        row (pd.Series): A single row of the DataFrame.
        description_column (str): Column name containing product descriptions.

    Returns:
        dict: Processed data including new_price, currency, and used_price.
    """
    logging.info(f"Processing row index: {index}")
    description = row[description_column]
    result = {"new_price": None, "currency": None, "used_price": None}

    try:
        # Generate new price
        new_price_prompt = generate_new_price_prompt(description)
        new_price_response = generate_with_openai(new_price_prompt)
        result["new_price"] = new_price_response.get("price", "Not Found")
        result["currency"] = new_price_response.get("currency", "Not Found")

        # Generate used price if new price is available
        if result["new_price"] != "Not Found":
            used_price_prompt = generate_used_price_prompt(description, result["new_price"], result["currency"])
            used_price_response = generate_with_openai(used_price_prompt)
            result["used_price"] = used_price_response.get("used_price", "Not Found")
    except Exception as e:
        logging.error(f"Error processing row index {index}: {e}")

    return result



def process_file(file_path):
    """
    Process a single file to predict new and used prices using OpenAI API with parallelization.

    Args:
        file_path (str): Path to the input file.
    """
    logging.info(f"Processing file with OpenAI FMV: {file_path}")
    # start_time = time.time()

    # Determine file type
    file_extension = os.path.splitext(file_path)[1].lower()
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    # Load input file
    try:
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return

    # Clean column names
    df.columns = df.columns.str.strip()

    # Find the best column to use for descriptions
    description_column = find_best_column(df.columns, ["query_col"])
    if not description_column:
        logging.error(f"No valid description column found in {file_path}. Skipping...")
        return

    # Parallelize row processing
    max_workers = max(1, os.cpu_count())  # Use half the CPU count
    logging.info(f"Using {max_workers} workers for OpenAI API calls.")

    def process_task(row_data):
        index, row = row_data
        return process_row(index, row, description_column)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_task, df.iterrows()))
    except Exception as e:
        logging.error(f"Error during parallel processing: {e}")
        return

    # Update DataFrame with results
    df["new_price"] = [r["new_price"] for r in results]
    df["currency"] = [r["currency"] for r in results]
    df["openai_fmv"] = [r["used_price"] for r in results]

    # Save processed file
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_output_openai{file_extension}")
    try:
        if file_extension == ".csv":
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
        else:
            df.to_excel(output_file, index=False, engine="openpyxl")
        logging.info(f"Processed file saved at: {output_file}")
    except Exception as e:
        logging.error(f"Error saving processed file: {e}")
        
def fetch_marketplace_data(query_col, condition, currency):
    """
    Fetch similar product data from Amazon and eBay using OpenAI.

    Args:
        query_col (str): Product description.
        condition (str): Condition of the product (e.g., New, Used).
        currency (str): Currency for the prices.

    Returns:
        list: A list of product data fetched from marketplaces.
    """
    logging.info(f"Fetching marketplace data for: {query_col} | Condition: {condition} | Currency: {currency}")

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    # Construct the prompt
    prompt = f"""
    You are an expert in analyzing product prices on online marketplaces. If you can't browse the internet directly, please respond based on your knowledge.

    ### Task ###
    Search the prices for 5 products similar to the one described below, under the specified condition and currency.

    ### Input ###
    Product: {query_col}
    Condition: {condition} (e.g., New, Used, Good Refurbished)
    Currency: {currency}

    ### Output Requirements ###
    Return the results in this JSON format:
    [
        {{"price": "<price>", "condition": "<condition>","product_url": "<url>", "marketplace": "Amazon"}},
        {{"price": "<price>", "condition": "<condition>", "product_url": "<url>", "marketplace": "eBay"}},
        ...
    ]
    Ensure to exclude any markers like  ```json and just use the floating point number without any other symbols in price. 
    """

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        logging.info(f"OpenAI response received for: {query_col} | Condition: {condition} | Currency: {currency}")
        # print(response.choices[0].message.content)
        # Parse the response
        result = json.loads(response.choices[0].message.content.strip())
        # with open("data/output/marketplace_data.json", "a") as f:
        #     json.dump(result, f)
        return result
    except Exception as e:
        logging.error(f"Error fetching marketplace data: {e}")
        return []
    
def process_marketplace_data(file_path):
    """
    Process all rows in the file to fetch marketplace data with parallelization.

    Args:
        file_path (str): Path to the input file.

    Returns:
        dict: A dictionary with row indices and conditions as string keys, and fetched data as values.
    """
    logging.info(f"Processing marketplace data for file: {file_path}")
    results = {}

    # Load input file
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)

    # Prepare tasks
    tasks = []
    for index, row in df.iterrows():
        query_col = row["query_col"]
        currency = row["currency"]
        for condition in ["new", "open box", "excellent refurbished", "very good refurbished", "good refurbished", "used"]:
            tasks.append((index, query_col, condition, currency))

    # Helper function for processing a single task
    def process_task(task):
        """
        Process a single task to fetch marketplace data.

        Args:
            task (tuple): A tuple containing row index, query column, condition, and currency.

        Returns:
            tuple: A tuple containing a string key (row index and condition) and fetched data.
        """
        row_index, query_col, condition, currency = task
        key = f"{row_index}_{condition}"  # Use a string key
        try:
            data = fetch_marketplace_data(query_col, condition, currency)
            return key, data
        except Exception as e:
            logging.error(f"Error processing task for row {row_index}, condition {condition}: {e}")
            return key, []

    # Use ThreadPoolExecutor for parallel processing
    max_workers = max(1, os.cpu_count())  # Use half the CPU cores
    logging.info(f"Using {max_workers} workers for marketplace data fetching.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_results = executor.map(process_task, tasks)

    # Collect results
    for key, data in future_results:
        results[key] = data

    # Dynamically name the output JSON file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f"data/output/{base_name}_marketplace_data.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Marketplace data fetched and saved successfully at: {output_path}")

    return results, output_path


def main():
    """
    Main function to process all files in the "data/input" directory.

    Iterate over all files in the directory and process each CSV or Excel file
    using the process_file function.
    """
    data_dir = "data/input"
    if not os.path.exists(data_dir):
        logging.error(f"Data directory '{data_dir}' does not exist.")
        return

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith((".csv", ".xlsx")):
            process_file(file_path)

  
# if __name__ == "__main__":
#     script_start_time = time.time()
#     main()
#     script_total_time = time.time() - script_start_time
#     logging.info(f"Total script execution time: {script_total_time:.2f} seconds.")
 

# if __name__ == "__main__":
#     test_query = "Apple iPhone 13"
#     test_condition = "New"
#     test_currency = "USD"

#     results = fetch_marketplace_data(test_query, test_condition, test_currency)
#     print(json.dumps(results, indent=4))


# if __name__ == "__main__":
#     input_file = "data/output/preprocessed_reconext_product list_bid request_output_openai_vendidit.xlsx"  # Replace with your input file path

#     try:
#         results = process_marketplace_data(input_file)
#         print("Marketplace data processed successfully.")
#         print("Sample Results:")
#         for key, value in list(results.items())[:5]:  # Show a sample of results
#             print(f"Key: {key}, Data: {value}")
#     except Exception as e:
#         print(f"Error during marketplace data processing: {e}")
