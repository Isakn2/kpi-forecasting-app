import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
# Suppress ML library warnings during model training
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', module='statsmodels')

# Time series specific imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from datetime import datetime, timedelta

class ForecastingEngine:
    """
    Automated model selection and training for KPI forecasting
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def detect_problem_type(self, df, target_column):
        """
        Detect if the problem is classification, regression, or time series forecasting
        """
        target_data = df[target_column].dropna()
        
        # Check if it's categorical (classification)
        if target_data.dtype == 'object' or len(target_data.unique()) < 10:
            return 'classification'
        
        # Check if it has a clear time component (time series)
        date_col = df.attrs.get('date_column')
        if date_col and len(df) > 20:
            return 'time_series'
        
        # Default to regression
        return 'regression'
    
    def prepare_time_series_data(self, df, target_column):
        """
        Prepare data for time series forecasting
        """
        date_col = df.attrs.get('date_column')
        
        # Create time series dataframe
        ts_df = df[[date_col, target_column]].copy()
        ts_df = ts_df.dropna()
        ts_df = ts_df.sort_values(date_col)
        
        return ts_df
    
    def train_prophet_model(self, df, target_column, forecast_periods=30):
        """
        Train Facebook Prophet model for time series forecasting
        """
        if not PROPHET_AVAILABLE:
            return None, None, "Prophet not available"
        
        try:
            ts_df = self.prepare_time_series_data(df, target_column)
            date_col = df.attrs.get('date_column')
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_df = pd.DataFrame({
                'ds': ts_df[date_col],
                'y': ts_df[target_column]
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                uncertainty_samples=100
            )
            
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = model.predict(future)
            
            # Calculate performance on historical data
            historical_pred = forecast[:-forecast_periods] if forecast_periods > 0 else forecast
            actual_values = prophet_df['y'].values
            predicted_values = historical_pred['yhat'].values
            
            mae = mean_absolute_error(actual_values, predicted_values)
            mse = mean_squared_error(actual_values, predicted_values)
            r2 = r2_score(actual_values, predicted_values)
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            
            performance = {
                'mae': mae,
                'mse': mse,
                'r2_score': r2,
                'mape': mape
            }
            
            # Extract forecast data
            forecast_data = []
            if forecast_periods > 0:
                forecast_future = forecast[-forecast_periods:]
                for _, row in forecast_future.iterrows():
                    forecast_data.append({
                        'date': row['ds'],
                        'forecast': row['yhat'],
                        'lower_bound': row['yhat_lower'],
                        'upper_bound': row['yhat_upper']
                    })
            
            return model, performance, forecast_data
            
        except Exception as e:
            return None, None, f"Prophet model failed: {str(e)}"
    
    def train_arima_model(self, df, target_column, forecast_periods=30):
        """
        Train ARIMA model for time series forecasting
        """
        if not STATSMODELS_AVAILABLE:
            return None, None, "Statsmodels not available"
        
        try:
            ts_df = self.prepare_time_series_data(df, target_column)
            ts_data = ts_df[target_column].values
            
            # Auto ARIMA (simplified approach)
            # Try different parameter combinations
            best_aic = float('inf')
            best_order = None
            best_model = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                                best_model = fitted_model
                        except:
                            continue
            
            if best_model is None:
                return None, None, "Could not fit ARIMA model"
            
            # Make forecasts
            forecast = best_model.forecast(steps=forecast_periods)
            forecast_ci = best_model.get_forecast(steps=forecast_periods).conf_int()
            
            # Calculate performance
            fitted_values = best_model.fittedvalues
            actual_values = ts_data[1:]  # Skip first value due to differencing
            
            mae = mean_absolute_error(actual_values, fitted_values)
            mse = mean_squared_error(actual_values, fitted_values)
            r2 = r2_score(actual_values, fitted_values)
            mape = np.mean(np.abs((actual_values - fitted_values) / actual_values)) * 100
            
            performance = {
                'mae': mae,
                'mse': mse,
                'r2_score': r2,
                'mape': mape,
                'aic': best_aic
            }
            
            # Prepare forecast data
            date_col = df.attrs.get('date_column')
            last_date = pd.to_datetime(ts_df[date_col].max())
            
            forecast_data = []
            for i in range(forecast_periods):
                next_date = last_date + pd.Timedelta(days=i+1)  # Assume daily frequency
                forecast_data.append({
                    'date': next_date,
                    'forecast': forecast[i],
                    'lower_bound': forecast_ci.iloc[i, 0] if len(forecast_ci) > i else forecast[i] * 0.9,
                    'upper_bound': forecast_ci.iloc[i, 1] if len(forecast_ci) > i else forecast[i] * 1.1
                })
                last_date = next_date
            
            return best_model, performance, forecast_data
            
        except Exception as e:
            return None, None, f"ARIMA model failed: {str(e)}"
    
    def train_ml_regression_model(self, df, target_column, forecast_periods=30):
        """
        Train machine learning models for regression forecasting
        """
        try:
            # Prepare features
            df_model, _ = self._prepare_ml_features(df, target_column)
            
            # Select features and target
            feature_columns = [col for col in df_model.columns if col != target_column and df_model[col].dtype in ['int64', 'float64']]
            
            if len(feature_columns) == 0:
                return None, None, "No suitable features for ML model"
            
            X = df_model[feature_columns].fillna(0)
            y = df_model[target_column].dropna()
            
            # Align X and y
            X = X.loc[y.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Try different models
            models_to_try = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
            
            best_score = -float('inf')
            best_model_name = None
            best_model = None
            
            for name, model in models_to_try.items():
                try:
                    if name == 'Linear Regression':
                        model.fit(X_train_scaled, y_train)
                        score = model.score(X_test_scaled, y_test)
                    else:
                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                        best_model = model
                except:
                    continue
            
            if best_model is None:
                return None, None, "No suitable ML model found"
            
            # Calculate performance metrics
            if best_model_name == 'Linear Regression':
                y_pred = best_model.predict(X_test_scaled)
            else:
                y_pred = best_model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            performance = {
                'mae': mae,
                'mse': mse,
                'r2_score': r2,
                'mape': mape,
                'model_name': best_model_name
            }
            
            # Generate forecasts (simplified approach)
            forecast_data = self._generate_ml_forecasts(df, best_model, best_model_name, feature_columns, target_column, forecast_periods)
            
            return best_model, performance, forecast_data
            
        except Exception as e:
            return None, None, f"ML regression model failed: {str(e)}"
    
    def _prepare_ml_features(self, df, target_column):
        """
        Prepare features for machine learning models
        """
        df_model = df.copy()
        
        # Add time-based features if date column exists
        date_col = df.attrs.get('date_column')
        if date_col and date_col in df_model.columns:
            df_model['year'] = df_model[date_col].dt.year
            df_model['month'] = df_model[date_col].dt.month
            df_model['day'] = df_model[date_col].dt.day
            df_model['dayofweek'] = df_model[date_col].dt.dayofweek
            df_model['quarter'] = df_model[date_col].dt.quarter
        
        # Add lag features for time series
        if len(df_model) > 10:
            for lag in [1, 2, 3, 7]:
                if len(df_model) > lag:
                    df_model[f'{target_column}_lag_{lag}'] = df_model[target_column].shift(lag)
        
        return df_model, target_column
    
    def _generate_ml_forecasts(self, df, model, model_name, feature_columns, target_column, forecast_periods):
        """
        Generate forecasts using trained ML model
        """
        forecast_data = []
        
        try:
            # Get the last row for feature continuation
            last_row = df.iloc[-1:].copy()
            date_col = df.attrs.get('date_column')
            
            if date_col:
                last_date = pd.to_datetime(last_row[date_col].iloc[0])
            else:
                last_date = pd.to_datetime(datetime.now())
            
            # Simple forecasting approach - use last known values with time progression
            for i in range(forecast_periods):
                next_date = last_date + pd.Timedelta(days=i+1)
                
                # Create feature vector (simplified)
                feature_vector = []
                for col in feature_columns:
                    if col in ['year', 'month', 'day', 'dayofweek', 'quarter']:
                        if col == 'year':
                            feature_vector.append(next_date.year)
                        elif col == 'month':
                            feature_vector.append(next_date.month)
                        elif col == 'day':
                            feature_vector.append(next_date.day)
                        elif col == 'dayofweek':
                            feature_vector.append(next_date.weekday())
                        elif col == 'quarter':
                            feature_vector.append((next_date.month - 1) // 3 + 1)
                    else:
                        # Use last known value or mean
                        if col in df.columns:
                            feature_vector.append(df[col].fillna(df[col].mean()).iloc[-1])
                        else:
                            feature_vector.append(0)
                
                # Make prediction
                feature_array = np.array(feature_vector).reshape(1, -1)
                
                if model_name == 'Linear Regression':
                    feature_array = self.scaler.transform(feature_array)
                
                prediction = model.predict(feature_array)[0]
                
                forecast_data.append({
                    'date': next_date,
                    'forecast': prediction,
                    'lower_bound': prediction * 0.9,  # Simple confidence interval
                    'upper_bound': prediction * 1.1
                })
        
        except Exception as e:
            print(f"Error generating ML forecasts: {str(e)}")
            # Fallback: simple linear trend
            date_col = df.attrs.get('date_column')
            if date_col:
                last_date = pd.to_datetime(df[date_col].max())
            else:
                last_date = pd.to_datetime(datetime.now())
                
            recent_values = df[target_column].tail(10).values
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            last_value = recent_values[-1]
            
            for i in range(forecast_periods):
                next_date = last_date + pd.Timedelta(days=i+1)
                prediction = last_value + trend * (i + 1)
                
                forecast_data.append({
                    'date': next_date,
                    'forecast': prediction,
                    'lower_bound': prediction * 0.9,
                    'upper_bound': prediction * 1.1
                })
        
        return forecast_data
    
    def auto_select_and_train(self, df, forecast_periods=30, confidence_level=0.95):
        """
        Automatically select and train the best model based on data characteristics
        """
        kpi_columns = df.attrs.get('kpi_columns', [])
        if not kpi_columns:
            raise Exception("No KPI columns found in data")
        
        # Use the first KPI column as target
        target_column = kpi_columns[0]
        
        # Detect problem type
        problem_type = self.detect_problem_type(df, target_column)
        
        models_tried = []
        best_model = None
        best_performance = None
        best_forecast = None
        best_model_name = ""
        
        # Try time series models first if applicable
        if problem_type == 'time_series' and len(df) >= 20:
            # Try Prophet
            if PROPHET_AVAILABLE:
                model, performance, forecast = self.train_prophet_model(df, target_column, forecast_periods)
                if model is not None:
                    models_tried.append(('Prophet', performance, forecast))
            
            # Try ARIMA
            if STATSMODELS_AVAILABLE:
                model, performance, forecast = self.train_arima_model(df, target_column, forecast_periods)
                if model is not None:
                    models_tried.append(('ARIMA', performance, forecast))
        
        # Try ML models
        model, performance, forecast = self.train_ml_regression_model(df, target_column, forecast_periods)
        if model is not None:
            model_name = performance.get('model_name', 'ML Regression')
            models_tried.append((model_name, performance, forecast))
        
        # Select best model based on performance
        if models_tried:
            # Use R² score as primary metric, fallback to MAE
            best_score = -float('inf')
            
            for model_name, performance, forecast in models_tried:
                score = performance.get('r2_score', 0)
                if score is None or np.isnan(score):
                    # Use negative MAE as score if R² is not available
                    score = -performance.get('mae', float('inf'))
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_performance = performance
                    best_forecast = forecast
        
        if not best_model_name:
            raise Exception("No suitable model could be trained with the provided data")
        
        return {
            'model_name': best_model_name,
            'performance': best_performance,
            'predictions': best_forecast,
            'forecast_data': best_forecast
        }
