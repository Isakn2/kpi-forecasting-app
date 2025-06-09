![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

# KPI Forecasting Web Application

A comprehensive web application designed for business analysts and non-technical users to forecast key performance indicators (KPIs) based on historical trends.

## Features

- 📊 **Automated Model Selection**: ARIMA, Prophet, Linear Regression, Random Forest
- 🎯 **User-Friendly Interface**: No coding required - just upload your data
- 📈 **Interactive Visualizations**: Historical trends, forecasts, and performance metrics
- 🔍 **Comprehensive Analytics**: Data quality assessment and model performance evaluation
- 💾 **Export Ready**: Download forecasts as Excel or CSV files
- 🚀 **Production Ready**: Deployed on Render with professional configuration

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Prophet, Statsmodels
- **Visualizations**: Plotly, Seaborn, Matplotlib
- **File Handling**: OpenPyXL, XLRD

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided local URL (usually http://localhost:8501)

3. Upload your Excel file containing historical KPI data

4. The application will automatically:
   - Clean and preprocess your data
   - Select the best forecasting model
   - Generate exploratory visualizations
   - Produce forecasts and performance metrics

## Deployment

This application is configured for deployment on Render. Simply connect your repository and deploy.

## File Structure

- `app.py`: Main Streamlit application
- `models/`: Machine learning model implementations
- `utils/`: Utility functions for data processing and visualization
- `requirements.txt`: Python dependencies

## Supported Data Formats

- Excel files (.xlsx, .xls)
- Data should include date/time columns and KPI values
- The application handles various date formats automatically
