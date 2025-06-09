# KPI Forecasting Web Application

A comprehensive web application designed for business analysts and non-technical users to forecast key performance indicators (KPIs) based on historical trends.

## Features

- **Data Ingestion**: Upload Excel files (.xlsx or .xls) with historical KPI data
- **Automated Data Preprocessing**: Intelligent handling of missing values and data cleaning
- **Automated Model Selection**: Automatically selects and trains the most appropriate ML model
- **Exploratory Data Analysis**: Interactive visualizations to understand your data
- **Forecasting & Predictions**: Clear display of forecasted KPI values with visualizations
- **Performance Metrics**: Model accuracy and performance indicators
- **Export Functionality**: Download forecasts in Excel/CSV format

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
