import pandas as pd
import json
import logging
import os

def aggregate_comparison_data(input_file, marketplace_data_path):
    """
    Aggregate marketplace data with Vendidit FMV predictions.

    Args:
        input_file (str): Path to the input file with Vendidit predictions.
        marketplace_data_path (str): Path to the JSON file with marketplace data.

    Returns:
        pd.DataFrame: DataFrame containing aggregated comparison data.
    """
    logging.info("Aggregating marketplace data with Vendidit FMV predictions...")

    # Load the input file with Vendidit FMV predictions
    df = pd.read_excel(input_file) if input_file.endswith(".xlsx") else pd.read_csv(input_file)

    # Load the marketplace data
    with open(marketplace_data_path, "r") as f:
        marketplace_data = json.load(f)

    # Prepare comparison data
    comparison_rows = []

    for key, market_data in marketplace_data.items():
        row_index, condition = key.split("_", 1)  # Extract row index and condition
        row_index = int(row_index)

        if not market_data:  # Skip if no data available
            continue

        # Retrieve marketplace prices
        prices = [float(item["price"].replace("$", "").replace("USD", "").replace(",", "").strip()) for item in market_data if "price" in item]
        avg_market_price = sum(prices) / len(prices) if prices else None
        min_market_price = min(prices) if prices else None
        max_market_price = max(prices) if prices else None

        # Retrieve Vendidit FMV prediction
        vendidit_col = f"vendidit_fmv_{condition.replace(' ', '_')}"
        vendidit_price = df.at[row_index, vendidit_col] if vendidit_col in df.columns else None

        # Append comparison data
        comparison_rows.append({
            "row_index": row_index,
            "query_col": df.at[row_index, "query_col"],
            "condition": condition,
            "currency": df.at[row_index, "currency"],
            "avg_market_price": avg_market_price,
            "min_market_price": min_market_price,
            "max_market_price": max_market_price,
            "vendidit_fmv": vendidit_price,
        })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_rows)

    # Generate dynamic filename
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_path = f"data/output/{input_filename}_aggregated_comparison_data.csv"
    comparison_df.to_csv(output_path, index=False)
    logging.info(f"Aggregated comparison data saved at: {output_path}")

    return comparison_df, output_path
