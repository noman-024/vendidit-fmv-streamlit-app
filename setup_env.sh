#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the virtual environment name
ENV_NAME="fmv-venv"

# Create the virtual environment
python3 -m venv $ENV_NAME

# Activate the virtual environment
source $ENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt

echo "Setup complete. Virtual environment '$ENV_NAME' created and required packages installed."
echo "To activate the virtual environment, run: source $ENV_NAME/bin/activate"
echo "To deactivate the virtual environment, run: deactivate"
