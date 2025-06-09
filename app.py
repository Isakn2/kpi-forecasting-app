import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
# Suppress specific warnings for cleaner UI
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')

# Import custom modules
from utils.data_processor import DataProcessor
from utils.visualization import DataVisualizer
from models.forecasting_models import ForecastingEngine
from utils.model_evaluator import ModelEvaluator

# Configure page
st.set_page_config(
    page_title="KPI Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.markdown('<h1 class="main-header">📈 KPI Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p><strong>Automated KPI forecasting for business analysts and non-technical users</strong></p>
        <p>Upload your historical data and get intelligent forecasts with automated model selection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing your historical KPI data with date and value columns"
        )
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            
            # Configuration options
            st.header("⚙️ Configuration")
            forecast_periods = st.slider(
                "Forecast Periods",
                min_value=1,
                max_value=365,
                value=30,
                help="Number of periods to forecast into the future"
            )
            
            confidence_level = st.selectbox(
                "Confidence Level",
                [0.95, 0.90, 0.80],
                index=0,
                help="Confidence level for prediction intervals"
            )
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Initialize processors
            data_processor = DataProcessor()
            visualizer = DataVisualizer()
            forecasting_engine = ForecastingEngine()
            evaluator = ModelEvaluator()
            
            # Process the uploaded file
            with st.spinner("Processing your data..."):
                df = data_processor.load_data(uploaded_file)
                
            # Display basic data info
            st.header("📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Records", 
                    len(df),
                    help="Number of data points in your dataset. More records generally lead to better forecasts."
                )
            with col2:
                st.metric(
                    "Columns", 
                    len(df.columns),
                    help="Number of features/variables in your data. The system will automatically identify date and KPI columns."
                )
            with col3:
                st.metric(
                    "Date Range", 
                    f"{len(df)} periods",
                    help="Number of time periods covered by your data. Longer time series provide more patterns for forecasting."
                )
            with col4:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                missing_status = "Good" if missing_pct < 5 else "Fair" if missing_pct < 15 else "Poor"
                st.metric(
                    "Missing Data", 
                    f"{missing_pct:.1f}%",
                    help=f"Percentage of missing values in your dataset. Status: {missing_status}. <5% is ideal, <15% is acceptable."
                )
            
            # Show raw data preview
            with st.expander("📋 View Raw Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data preprocessing
            with st.spinner("Cleaning and preprocessing data..."):
                processed_data, preprocessing_report = data_processor.preprocess_data(df)
                
            # Show preprocessing report
            if preprocessing_report:
                st.header("🔧 Data Preprocessing Report")
                for message in preprocessing_report:
                    if "warning" in message.lower() or "removed" in message.lower():
                        st.markdown(f'<div class="warning-message">{message}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
            
            # Exploratory Data Analysis
            st.header("🔍 Exploratory Data Analysis")
            
            # Create tabs for different visualizations
            eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📈 Time Series", "📊 Distribution", "🔗 Correlations"])
            
            with eda_tab1:
                st.subheader("Historical KPI Trends")
                time_series_fig = visualizer.create_time_series_plot(processed_data)
                st.plotly_chart(time_series_fig, use_container_width=True)
            
            with eda_tab2:
                st.subheader("Data Distribution Analysis")
                dist_figs = visualizer.create_distribution_plots(processed_data)
                for fig in dist_figs:
                    st.plotly_chart(fig, use_container_width=True)
            
            with eda_tab3:
                st.subheader("Feature Correlations")
                corr_fig = visualizer.create_correlation_heatmap(processed_data)
                if corr_fig:
                    st.plotly_chart(corr_fig, use_container_width=True)
                else:
                    st.info("Correlation analysis requires multiple numeric columns.")
            
            # Model Training and Forecasting
            st.header("🤖 Automated Model Training")
            
            with st.spinner("Training models and generating forecasts..."):
                # Determine the best model and make predictions
                model_results = forecasting_engine.auto_select_and_train(
                    processed_data, 
                    forecast_periods=forecast_periods,
                    confidence_level=confidence_level
                )
                
                selected_model = model_results['model_name']
                predictions = model_results['predictions']
                model_performance = model_results['performance']
                forecast_data = model_results['forecast_data']
            
            # Display model information
            st.success(f"🎯 **Model Used**: {selected_model}")
            
            # Model Performance Metrics
            st.header("📏 Model Performance")
            st.markdown("**Understanding Your Model's Accuracy:**")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                if 'accuracy' in model_performance:
                    st.metric(
                        "Accuracy", 
                        f"{model_performance['accuracy']:.2%}",
                        help="Percentage of correct predictions. Higher is better (closer to 100%)."
                    )
                elif 'mae' in model_performance:
                    st.metric(
                        "Mean Absolute Error", 
                        f"{model_performance['mae']:.4f}",
                        help="Average difference between predicted and actual values. Lower is better (closer to 0). This tells you how far off your predictions typically are."
                    )
            
            with perf_col2:
                if 'r2_score' in model_performance:
                    r2_value = model_performance['r2_score']
                    r2_interpretation = "Excellent" if r2_value > 0.9 else "Good" if r2_value > 0.7 else "Fair" if r2_value > 0.5 else "Poor"
                    st.metric(
                        "R² Score", 
                        f"{r2_value:.4f}",
                        help=f"Coefficient of determination (0-1). Measures how well the model explains the data variance. Current score: {r2_interpretation}. 1.0 = Perfect fit, 0.0 = No better than average."
                    )
                elif 'mse' in model_performance:
                    st.metric(
                        "Mean Squared Error", 
                        f"{model_performance['mse']:.4f}",
                        help="Average of squared differences between predicted and actual values. Lower is better. Penalizes larger errors more heavily than MAE."
                    )
            
            with perf_col3:
                if 'mape' in model_performance:
                    mape_value = model_performance['mape']
                    mape_interpretation = "Excellent" if mape_value < 5 else "Good" if mape_value < 10 else "Fair" if mape_value < 20 else "Poor"
                    st.metric(
                        "MAPE", 
                        f"{mape_value:.2f}%",
                        help=f"Mean Absolute Percentage Error. Shows prediction accuracy as a percentage. Current accuracy: {mape_interpretation}. <5% = Excellent, 5-10% = Good, 10-20% = Fair, >20% = Needs improvement."
                    )
            
            # Add interpretive summary
            if 'mape' in model_performance and 'r2_score' in model_performance:
                mape_val = model_performance['mape']
                r2_val = model_performance['r2_score']
                
                if mape_val < 10 and r2_val > 0.8:
                    interpretation = "🟢 **Excellent Model Performance** - Your forecasts are highly reliable!"
                elif mape_val < 20 and r2_val > 0.6:
                    interpretation = "🟡 **Good Model Performance** - Your forecasts are reasonably accurate."
                else:
                    interpretation = "🔴 **Model Needs Improvement** - Consider using more data or different model parameters."
                
                st.markdown(f"**Overall Assessment:** {interpretation}")
                
            # Add expandable section with detailed explanations
            with st.expander("📚 Learn More About These Metrics"):
                st.markdown("""
                **Mean Absolute Error (MAE):**
                - Measures the average magnitude of errors in your predictions
                - Same units as your data (e.g., if predicting sales in dollars, MAE is in dollars)
                - Example: MAE of 15.83 means predictions are typically off by about 15.83 units
                
                **R² Score (Coefficient of Determination):**
                - Measures how well your model explains the variability in your data
                - Scale: 0 to 1 (higher is better)
                - 0.95 means the model explains 95% of the data variance - excellent!
                - 0.50 means the model explains 50% of the variance - fair
                
                **MAPE (Mean Absolute Percentage Error):**
                - Shows prediction accuracy as a percentage of actual values
                - Easy to interpret: 1.29% means predictions are typically 1.29% off
                - Industry benchmarks: <5% Excellent, 5-10% Good, 10-20% Acceptable, >20% Poor
                
                **What These Numbers Mean for Your Business:**
                - Lower MAE = More precise predictions
                - Higher R² = Model captures patterns well
                - Lower MAPE = More accurate relative predictions
                """)
            
            # Forecasting Results
            st.header("🔮 Forecasting Results")
            
            # Create forecast visualization
            forecast_fig = visualizer.create_forecast_plot(
                processed_data, 
                forecast_data, 
                selected_model
            )
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Forecast summary table
            st.subheader("📋 Forecast Summary")
            forecast_df = pd.DataFrame(forecast_data)
            st.dataframe(forecast_df, use_container_width=True)
            
            # Download options
            st.header("💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # Create Excel file for download
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    forecast_df.to_excel(writer, sheet_name='Forecasts', index=False)
                    processed_data.to_excel(writer, sheet_name='Processed_Data', index=False)
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_buffer.getvalue(),
                    file_name=f"kpi_forecast_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Create CSV file for download
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"kpi_forecasts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"An error occurred while processing your data: {str(e)}")
            st.info("Please ensure your Excel file contains proper date and numeric columns.")
    
    else:
        # Instructions for users
        st.header("🚀 Getting Started")
        st.markdown("""
        ### How to use this KPI Forecasting Dashboard:
        
        1. **📁 Upload Data**: Click "Browse files" in the sidebar to upload your Excel file
        2. **⚙️ Configure**: Set your forecast parameters (periods, confidence level)
        3. **📊 Analyze**: Review the automated data analysis and visualizations
        4. **🤖 Forecast**: Get automated model selection and predictions
        5. **💾 Download**: Export your results in Excel or CSV format
        
        ### Data Requirements:
        - Excel file (.xlsx or .xls format)
        - At least one date/time column
        - At least one numeric KPI column
        - Minimum 10 data points recommended
        
        ### Supported Models:
        - **Time Series**: ARIMA, Prophet, Exponential Smoothing
        - **Regression**: Random Forest, Linear Regression, XGBoost
        - **Classification**: Random Forest, Logistic Regression, SVM
        
        The application automatically selects the best model based on your data characteristics.
        """)

if __name__ == "__main__":
    main()
