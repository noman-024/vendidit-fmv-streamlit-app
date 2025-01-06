## app.py
import os
import streamlit as st
import pandas as pd
import logging
import time
from scripts.preprocessing import preprocess_file_with_query
from scripts.openai_fmv import process_file as process_openai_file, process_marketplace_data
from scripts.vendidit_fmv import process_vendidit_file
from scripts.marketplace_analysis import aggregate_comparison_data
import altair as alt

# Configure logging
log_file = os.path.abspath("data/logs/log.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create necessary directories
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/output", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

# Initialize session state
if "steps_completed" not in st.session_state:
    st.session_state["steps_completed"] = {
        "step_1": False, "step_2": False, "step_3": False,
        "step_4": False, "step_5": False, "step_6": False
    }

if "file_paths" not in st.session_state:
    st.session_state["file_paths"] = {}

# # Reset Workflow
# if st.button("Reset Workflow"):
#     for key in ["steps_completed", "file_paths"]:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.experimental_set_query_params()  # Reset URL query parameters
#     st.stop()

# Streamlit App Title
st.title("Vendidit - GlobalSKU FMV")

def load_file(file_path):
    """Loads a file and returns a DataFrame, handling both CSV and Excel."""
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path, encoding="utf-8")
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel files are allowed.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error loading file: {file_path} | {e}")
        return None

def save_file(dataframe, file_path):
    """Saves a DataFrame to the specified file path."""
    try:
        if file_path.endswith(".csv"):
            dataframe.to_csv(file_path, index=False, encoding="utf-8")
        elif file_path.endswith(".xlsx"):
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                dataframe.to_excel(writer, index=False, sheet_name="Sheet1")
    except Exception as e:
        st.error(f"Error saving file: {e}")
        logging.error(f"Error saving file: {file_path} | {e}")

# Step 1: File Upload
st.header("Step 1: Upload Input File")
if not st.session_state["steps_completed"]["step_1"]:
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        start_time = time.time()
        input_path = os.path.join("data/input", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["file_paths"]["input_path"] = input_path
        st.success(f"File uploaded and saved as: {input_path}")
        logging.info(f"File uploaded: {uploaded_file.name} | Time taken: {time.time() - start_time:.2f}s")
        st.session_state["steps_completed"]["step_1"] = True

if st.session_state["steps_completed"]["step_1"]:
    # Step 2: Preprocessing
    st.header("Step 2: Preprocess File")
    if not st.session_state["steps_completed"]["step_2"]:
        if st.button("Start Preprocessing"):
            start_time = time.time()
            preprocessed_file_path = preprocess_file_with_query(st.session_state["file_paths"]["input_path"])
            st.session_state["file_paths"]["preprocessed_file_path"] = preprocessed_file_path
            st.success(f"Preprocessed file saved as: {preprocessed_file_path}")
            logging.info(f"Preprocessing completed | Time taken: {time.time() - start_time:.2f}s")
            st.session_state["steps_completed"]["step_2"] = True

    if "preprocessed_file_path" in st.session_state["file_paths"]:
        preprocessed_df = load_file(st.session_state["file_paths"]["preprocessed_file_path"])
        if preprocessed_df is not None:
            st.write("Preview of Preprocessed File:")
            st.dataframe(preprocessed_df)
            st.download_button(
                label="Download Preprocessed File",
                data=preprocessed_df.to_csv(index=False, encoding="utf-8"),
                file_name="preprocessed_file.csv",
                mime="text/csv",
            )

if st.session_state["steps_completed"]["step_2"]:
    # Step 3: OpenAI FMV Prediction
    st.header("Step 3: Predict Prices using OpenAI")
    if not st.session_state["steps_completed"]["step_3"]:
        if st.button("Start OpenAI Prediction"):
            start_time = time.time()
            process_openai_file(st.session_state["file_paths"]["preprocessed_file_path"])
            preprocessed_basename = os.path.splitext(os.path.basename(st.session_state["file_paths"]["preprocessed_file_path"]))[0]
            openai_output_file_extension = ".xlsx" if st.session_state["file_paths"]["preprocessed_file_path"].endswith(".xlsx") else ".csv"
            openai_output_file_path = os.path.join("data/output", f"{preprocessed_basename}_output_openai{openai_output_file_extension}")
            st.session_state["file_paths"]["openai_output_file_path"] = openai_output_file_path
            st.success(f"OpenAI FMV prediction completed and saved at: {openai_output_file_path}")
            logging.info(f"OpenAI FMV prediction completed | Time taken: {time.time() - start_time:.2f}s")
            st.session_state["steps_completed"]["step_3"] = True

    if "openai_output_file_path" in st.session_state["file_paths"]:
        openai_df = load_file(st.session_state["file_paths"]["openai_output_file_path"])
        if openai_df is not None:
            st.write("Preview of OpenAI Processed File:")
            st.dataframe(openai_df)
            st.download_button(
                label="Download OpenAI Processed File",
                data=openai_df.to_csv(index=False, encoding="utf-8"),
                file_name="openai_processed_file.csv",
                mime="text/csv",
            )

if st.session_state["steps_completed"]["step_3"]:
    # Step 4: Vendidit FMV Prediction
    st.header("Step 4: Predict Prices using Vendidit FMV API")
    if not st.session_state["steps_completed"]["step_4"]:
        if st.button("Start Vendidit Prediction"):
            start_time = time.time()
            preprocessed_basename = os.path.splitext(os.path.basename(st.session_state["file_paths"]["preprocessed_file_path"]))[0]
            vendidit_output_file_extension = ".xlsx" if st.session_state["file_paths"]["openai_output_file_path"].endswith(".xlsx") else ".csv"
            vendidit_output_file_path = os.path.join("data/output", f"{preprocessed_basename}_vendidit{vendidit_output_file_extension}")
            process_vendidit_file(st.session_state["file_paths"]["openai_output_file_path"], vendidit_output_file_path)
            st.session_state["file_paths"]["vendidit_output_file_path"] = vendidit_output_file_path
            st.success(f"Vendidit FMV prediction completed and saved at: {vendidit_output_file_path}")
            logging.info(f"Vendidit FMV prediction completed | Time taken: {time.time() - start_time:.2f}s")
            st.session_state["steps_completed"]["step_4"] = True

    if "vendidit_output_file_path" in st.session_state["file_paths"]:
        vendidit_df = load_file(st.session_state["file_paths"]["vendidit_output_file_path"])
        if vendidit_df is not None:
            st.write("Preview of Vendidit Processed File:")
            st.dataframe(vendidit_df)
            st.download_button(
                label="Download Vendidit Processed File",
                data=vendidit_df.to_csv(index=False, encoding="utf-8"),
                file_name="vendidit_processed_file.csv",
                mime="text/csv",
            )

if st.session_state["steps_completed"]["step_4"]:
    # Step 5: Fetch Marketplace Data
    st.header("Step 5: Fetch Marketplace Data")
    if not st.session_state["steps_completed"]["step_5"]:
        if st.button("Fetch Marketplace Data"):
            start_time = time.time()
            marketplace_results, json_path = process_marketplace_data(st.session_state["file_paths"]["vendidit_output_file_path"])
            st.session_state["file_paths"]["json_path"] = json_path
            st.success(f"Marketplace data fetched successfully and saved at: {json_path}")
            logging.info(f"Marketplace data fetching completed | Time taken: {time.time() - start_time:.2f}s")
            st.session_state["steps_completed"]["step_5"] = True

    if "json_path" in st.session_state["file_paths"]:
        st.write("Marketplace data is ready for aggregation.")

if st.session_state["steps_completed"]["step_5"]:
    # Step 6: Aggregate Data and Visualize
    st.header("Step 6: Aggregate Data and Visualize")
    if st.button("Aggregate Data"):
        start_time = time.time()
        aggregated_data, aggregated_output_path = aggregate_comparison_data(
            st.session_state["file_paths"]["vendidit_output_file_path"],
            st.session_state["file_paths"]["json_path"]
        )
        st.session_state["file_paths"]["aggregated_output_path"] = aggregated_output_path
        st.success(f"Data aggregated successfully and saved at: {aggregated_output_path}")
        logging.info(f"Data aggregation completed | Time taken: {time.time() - start_time:.2f}s")
        st.write("Preview of Aggregated Data:")
        st.dataframe(aggregated_data)
        st.download_button(
            label="Download Aggregated Data",
            data=aggregated_data.to_csv(index=False, encoding="utf-8"),
            file_name="aggregated_data.csv",
            mime="text/csv",
        )

        # Visualization
        st.write("### Visualization and Insights")
        st.subheader("Comparison of Avg Market Price and Vendidit FMV")
        bar_chart_data = aggregated_data[['condition', 'currency', 'avg_market_price', 'vendidit_fmv']].copy()
        bar_chart_data = bar_chart_data.melt(id_vars=['condition', 'currency'], var_name='Metric', value_name='Price')
        st.bar_chart(data=bar_chart_data, x='condition', y='Price', color='Metric')

        st.subheader("Scatter Plot of Min/Max Market Prices vs Vendidit FMV")
        scatter_chart_data = aggregated_data[['condition', 'currency', 'min_market_price', 'max_market_price', 'vendidit_fmv']].copy()
        scatter_chart_data = scatter_chart_data.melt(id_vars=['condition', 'currency'], var_name='Price Type', value_name='Price')
        st.altair_chart(
            alt.Chart(scatter_chart_data).mark_circle(size=60).encode(
                x='condition:O',
                y='Price:Q',
                color='Price Type:N',
                tooltip=['condition', 'currency', 'Price Type', 'Price']
            ).interactive()
        )