# Vendidit FMV Streamlit App

- Author - Md Mutasim Billah Noman
- Updated on - 7 January 2024

## Overview

The **Vendidit FMV Streamlit App** provides a streamlined interface for evaluating product prices based on Fair Market Valuation (FMV). The application integrates with OpenAI's API and the Vendidit API to preprocess data, generate predictions, and fetch marketplace insights. It is designed to handle both CSV and Excel files and supports multi-step workflows, including data preprocessing, FMV predictions, and data visualization.

---

## Features

- **Multi-Step Workflow:**
  - Upload CSV or Excel files.
  - Preprocess data with dynamic query column generation.
  - Predict new and used prices using OpenAI's API.
  - Predict FMV across different conditions using Vendidit's API.
  - Fetch marketplace data from Amazon and eBay.
  - Aggregate and visualize the final data.

- **Visualization:**
  - Compare predicted prices and marketplace data.
  - Analyze minimum, maximum, and average market prices.

- **Error Handling:**
  - Cleans data to avoid encoding and corruption issues.
  - Provides detailed logs for debugging.

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- An OpenAI API key (add it to a `.env` file as `OPENAI_API_KEY`)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/noman-024/vendidit-fmv-streamlit-app.git
   cd vendidit-fmv-streamlit-app
   ```

2. **Set Up the Virtual Environment:**
   
   Run the setup script to create and activate a virtual environment:

   ```bash
   ./setup_env.sh
   ```

3. **Set Up the OpenAI API Key:**
   
   Add your OpenAI API key to a `.env` file in the root directory:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Install Dependencies:**
   
   All required dependencies are specified in the requirements.txt file and will be installed by the setup script.

### Usage

1. **Run the Application:**
   
   Start the Streamlit application locally:

   ```bash
   streamlit run app.py
   ```

2. **Workflow Steps:**
   
   - **Step 1:** Upload your input file (CSV or Excel).
   - **Step 2:** Preprocess the file to create a dynamic query column.
   - **Step 3:** Predict prices for new and used conditions using OpenAI.
   - **Step 4:** Predict FMV values for multiple conditions using Vendidit's API.
   - **Step 5:** Fetch marketplace data for comparisons.
   - **Step 6:** Aggregate, analyze, and visualize the final data.
  
3. **Deploy to Streamlit Cloud:**
   
   This app can be deployed to Streamlit Cloud for public access. Ensure all dependencies and environment variables are configured in the cloud environment.

### Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── setup_env.sh           # Environment setup script
├── .env                   # Environment variables
├── scripts/               # Directory for backend scripts
│   ├── marketplace_analysis.py  # Aggregation and comparison logic
│   ├── openai_fmv.py             # OpenAI FMV prediction logic
│   ├── preprocessing.py          # File preprocessing logic
│   ├── vendidit_fmv.py           # Vendidit FMV prediction logic
├── data/
│   ├── input/             # Directory for uploaded input files
│   ├── output/            # Directory for generated output files
│   └── logs/              # Directory for application logs
```

### Key Functionalities

1. **Data Preprocessing**
   
   - **Dynamic Column Suggestions:** Uses OpenAI to suggest the most relevant columns for creating a query column.
   - **Data Cleaning:** Handles both CSV and Excel files, resolving encoding and formatting issues.

2. **OpenAI FMV Prediction**
   
   - Predicts new and used product prices based on description and additional specifications.
   - Leverages OpenAI's GPT model for high-accuracy predictions.

3. **Vendidit API Integration**
   
   - Calls the Vendidit API to predict FMV across multiple conditions.
   - Handles parallel processing for efficiency.

4. **Marketplace Data Fetching**
   
   - Retrieves marketplace data (e.g., Amazon and eBay) for further analysis.

5. **Data Visualization**
   
   - Interactive bar charts and scatter plots for comparing FMV and marketplace data.
   - Insights into price variations across different conditions.

### Deployment

The app is hosted on Streamlit Cloud. You can access it here - https://vendidit-fmv-app-app-g7phaw8e53lusx6dxrjlrj.streamlit.app/.

### Logs and Debugging

- Logs are stored in the `data/logs/` directory.
- Each step of the workflow is logged, including error handling and API calls.

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
