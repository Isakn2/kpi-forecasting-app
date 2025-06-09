# API Documentation - KPI Forecasting Application

## Code Structure

### Main Application (`app.py`)

The main Streamlit application that orchestrates the entire forecasting workflow.

**Key Functions:**
- `main()`: Main application entry point
- Handles file upload, data processing, model training, and result visualization

### Data Processing (`utils/data_processor.py`)

**Class: `DataProcessor`**

Handles data loading, cleaning, and preprocessing.

**Key Methods:**
- `load_data(uploaded_file)`: Load Excel file into pandas DataFrame
- `detect_date_column(df)`: Automatically detect date columns
- `detect_kpi_columns(df)`: Automatically detect numeric KPI columns
- `handle_missing_values(df, numeric_columns)`: Handle missing data
- `preprocess_data(df)`: Main preprocessing orchestrator
- `prepare_for_modeling(df, target_column)`: Prepare data for ML models

**Usage Example:**
```python
processor = DataProcessor()
df = processor.load_data(uploaded_file)
processed_data, report = processor.preprocess_data(df)
```

### Visualization (`utils/visualization.py`)

**Class: `DataVisualizer`**

Creates interactive visualizations using Plotly.

**Key Methods:**
- `create_time_series_plot(df)`: Historical trend visualization
- `create_distribution_plots(df)`: Data distribution analysis
- `create_correlation_heatmap(df)`: Feature correlation matrix
- `create_forecast_plot(historical_data, forecast_data, model_name)`: Forecast visualization
- `create_model_performance_plot(actual, predicted, model_name)`: Model performance

**Usage Example:**
```python
visualizer = DataVisualizer()
fig = visualizer.create_time_series_plot(df)
st.plotly_chart(fig)
```

### Forecasting Models (`models/forecasting_models.py`)

**Class: `ForecastingEngine`**

Automated model selection and training for forecasting.

**Key Methods:**
- `detect_problem_type(df, target_column)`: Determine if problem is classification, regression, or time series
- `train_prophet_model(df, target_column, forecast_periods)`: Train Facebook Prophet model
- `train_arima_model(df, target_column, forecast_periods)`: Train ARIMA model
- `train_ml_regression_model(df, target_column, forecast_periods)`: Train ML regression models
- `auto_select_and_train(df, forecast_periods, confidence_level)`: Main method for automatic model selection

**Supported Models:**
- **Time Series**: Prophet, ARIMA
- **Machine Learning**: Random Forest, Linear Regression

**Usage Example:**
```python
engine = ForecastingEngine()
results = engine.auto_select_and_train(
    df, 
    forecast_periods=30, 
    confidence_level=0.95
)
```

### Model Evaluation (`utils/model_evaluator.py`)

**Class: `ModelEvaluator`**

Evaluates model performance and provides metrics.

**Key Methods:**
- `evaluate_regression_model(actual, predicted)`: Calculate regression metrics
- `evaluate_classification_model(actual, predicted)`: Calculate classification metrics
- `evaluate_forecast_quality(historical_data, forecast_data)`: Assess forecast quality
- `generate_model_report(model_name, performance_metrics)`: Create comprehensive report

**Metrics Provided:**
- **Regression**: MAE, MSE, RMSE, R², MAPE
- **Classification**: Accuracy, Precision, Recall, F1-Score

## Data Flow

1. **File Upload** → `DataProcessor.load_data()`
2. **Data Preprocessing** → `DataProcessor.preprocess_data()`
3. **Exploratory Analysis** → `DataVisualizer.create_*_plot()`
4. **Model Training** → `ForecastingEngine.auto_select_and_train()`
5. **Performance Evaluation** → `ModelEvaluator.evaluate_*_model()`
6. **Results Visualization** → `DataVisualizer.create_forecast_plot()`
7. **Export** → Excel/CSV download

## Configuration

### Streamlit Configuration

**Local Development** (`.streamlit/config.toml`):
```toml
[server]
address = "0.0.0.0"
port = 8501

[browser]
gatherUsageStats = false
```

**Production** (`.streamlit/config_production.toml`):
```toml
[server]
port = $PORT
address = "0.0.0.0"

[browser]
gatherUsageStats = false
```

### Dependencies

See `requirements.txt` for complete list. Key dependencies:
- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `plotly`: Interactive visualizations
- `scikit-learn`: Machine learning models
- `prophet`: Time series forecasting
- `statsmodels`: Statistical models

## Error Handling

### Data Validation
- Automatic detection of date and numeric columns
- Missing value handling with multiple strategies
- Outlier detection and removal
- Data quality reporting

### Model Fallbacks
- If time series models fail, falls back to ML models
- Multiple model attempts with best selection
- Graceful error handling with user-friendly messages

### File Processing
- Support for multiple Excel formats (.xlsx, .xls)
- File size validation
- Format verification

## Customization Points

### Adding New Models

To add a new forecasting model:

1. Create a new method in `ForecastingEngine`:
```python
def train_new_model(self, df, target_column, forecast_periods):
    # Implementation here
    return model, performance, forecast_data
```

2. Add to `auto_select_and_train()` method:
```python
# Try new model
model, performance, forecast = self.train_new_model(df, target_column, forecast_periods)
if model is not None:
    models_tried.append(('New Model', performance, forecast))
```

### Adding New Visualizations

To add new visualization types:

1. Create method in `DataVisualizer`:
```python
def create_new_plot(self, df):
    # Create plotly figure
    return fig
```

2. Add to main application:
```python
new_fig = visualizer.create_new_plot(processed_data)
st.plotly_chart(new_fig)
```

### Adding New Metrics

To add new performance metrics:

1. Extend `ModelEvaluator`:
```python
def evaluate_new_metric(self, actual, predicted):
    # Calculate new metric
    return metric_value
```

2. Include in performance dictionary in forecasting models

## Performance Considerations

### Memory Usage
- Large files are processed in chunks where possible
- Temporary data is cleaned up automatically
- Consider data sampling for very large datasets

### Processing Speed
- Model training is optimized for common use cases
- Progress indicators for long-running operations
- Async processing where applicable

### Scalability
- Designed for single-user deployment
- Can be scaled horizontally with load balancers
- Database integration possible for multi-user scenarios

## Security Notes

- No persistent data storage (files processed in memory)
- No user authentication (suitable for internal use)
- HTTPS enforced in production deployment
- Input validation for uploaded files
