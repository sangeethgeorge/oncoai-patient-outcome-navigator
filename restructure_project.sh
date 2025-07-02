#!/bin/bash

# This script restructures the oncoai-patient-outcome-navigator project
# to separate backend (oncoai_prototype in src/) and frontend (streamlit_app/).
#
# !! IMPORTANT: BACK UP YOUR PROJECT BEFORE RUNNING THIS SCRIPT !!
#
# Run this script from the root of your 'oncoai-patient-outcome-navigator' directory.

echo "Starting project restructuring..."

# --- 1. Create necessary directories and __init__.py files ---

echo "Creating core package directories within src/oncoai_prototype..."
mkdir -p src/oncoai_prototype/data_processing
mkdir -p src/oncoai_prototype/modeling
mkdir -p src/oncoai_prototype/utils # For general backend utilities

# Create __init__.py in new sub-packages if they don't exist
touch src/oncoai_prototype/data_processing/__init__.py
touch src/oncoai_prototype/modeling/__init__.py
touch src/oncoai_prototype/utils/__init__.py

echo "Ensuring streamlit_app is a package..."
# This assumes streamlit_app/__init__.py already exists, but create if not
touch streamlit_app/__init__.py

# Optional: Create 'pages' and 'utils' within streamlit_app for modularity
echo "Creating optional Streamlit app subdirectories..."
mkdir -p streamlit_app/pages
touch streamlit_app/pages/__init__.py
touch streamlit_app/utils.py # For Streamlit-specific utility functions

# --- 2. Move backend Python scripts into src/oncoai_prototype sub-packages ---

echo "Moving backend Python scripts to src/oncoai_prototype..."

# Data Processing
if [ -f src/feature_engineering_pipeline.py ]; then
    mv src/feature_engineering_pipeline.py src/oncoai_prototype/data_processing/
    echo "Moved src/feature_engineering_pipeline.py to src/oncoai_prototype/data_processing/"
fi

# Modeling/Prediction
if [ -f src/model_training_pipeline.py ]; then
    mv src/model_training_pipeline.py src/oncoai_prototype/modeling/
    echo "Moved src/model_training_pipeline.py to src/oncoai_prototype/modeling/"
fi

if [ -f src/predict.py ]; then
    mv src/predict.py src/oncoai_prototype/modeling/
    echo "Moved src/predict.py to src/oncoai_prototype/modeling/"
fi

if [ -f src/predict_inference_script.py ]; then
    mv src/predict_inference_script.py src/oncoai_prototype/modeling/
    echo "Moved src/predict_inference_script.py to src/oncoai_prototype/modeling/"
fi

echo "Restructuring complete."
echo "Please remember to manually update your pyproject.toml file"
echo "to reflect the new package structure and dependencies as discussed:"
echo 'packages = [ { include = "oncoai_prototype", from = "src" }, { include = "streamlit_app", from = "." } ]'
echo "And then run 'poetry install'."
