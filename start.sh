#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Copy production config
cp .streamlit/config_production.toml .streamlit/config.toml

# Run the Streamlit application
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
