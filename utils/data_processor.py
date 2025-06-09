import pandas as pd
import numpy as np
from datetime import datetime
import warnings
# Suppress pandas warnings during data processing
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='pandas')

class DataProcessor:
    """
    Handles data loading, cleaning, and preprocessing for KPI forecasting
    """
    
    def __init__(self):
        self.preprocessing_report = []
    
    def load_data(self, uploaded_file):
        """
        Load data from uploaded Excel file
        """
        try:
            # Try to read Excel file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                df = pd.read_excel(uploaded_file, engine='xlrd')
            
            self.preprocessing_report.append(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def detect_date_column(self, df):
        """
        Automatically detect date columns in the dataset
        """
        date_columns = []
        
        for col in df.columns:
            # Check if column name suggests it's a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'period', 'month', 'year', 'day']):
                date_columns.append(col)
                continue
            
            # Skip numeric columns for date detection
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Check if column values look like dates
            try:
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Try to parse as date
                    pd.to_datetime(sample_values, errors='raise')
                    date_columns.append(col)
            except:
                continue
        
        return date_columns
    
    def detect_kpi_columns(self, df, exclude_columns=None):
        """
        Automatically detect KPI (numeric) columns
        """
        if exclude_columns is None:
            exclude_columns = []
        
        kpi_columns = []
        
        for col in df.columns:
            if col in exclude_columns:
                continue
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                kpi_columns.append(col)
        
        return kpi_columns
    
    def handle_missing_values(self, df, numeric_columns):
        """
        Handle missing values in numeric columns
        """
        missing_info = []
        
        for col in numeric_columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                if missing_pct > 50:
                    # If more than 50% missing, consider dropping the column
                    df = df.drop(columns=[col])
                    missing_info.append(f"⚠️ Dropped column '{col}' due to {missing_pct:.1f}% missing values")
                elif missing_pct > 20:
                    # Use median for high missing percentage
                    df[col].fillna(df[col].median(), inplace=True)
                    missing_info.append(f"🔧 Filled {missing_count} missing values in '{col}' with median")
                else:
                    # Use mean for low missing percentage
                    df[col].fillna(df[col].mean(), inplace=True)
                    missing_info.append(f"🔧 Filled {missing_count} missing values in '{col}' with mean")
        
        return df, missing_info
    
    def convert_date_column(self, df, date_column):
        """
        Convert date column to proper datetime format
        """
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(by=date_column)
            return df, f"✅ Converted '{date_column}' to datetime and sorted data"
        except Exception as e:
            return df, f"⚠️ Could not convert '{date_column}' to datetime: {str(e)}"
    
    def detect_data_frequency(self, df, date_column):
        """
        Detect the frequency of the time series data
        """
        try:
            df_sorted = df.sort_values(by=date_column)
            date_diffs = df_sorted[date_column].diff().dropna()
            
            # Get the most common difference
            most_common_diff = date_diffs.mode().iloc[0]
            
            if most_common_diff.days == 1:
                return "Daily"
            elif most_common_diff.days == 7:
                return "Weekly"
            elif 28 <= most_common_diff.days <= 31:
                return "Monthly"
            elif 90 <= most_common_diff.days <= 92:
                return "Quarterly"
            elif 365 <= most_common_diff.days <= 366:
                return "Yearly"
            else:
                return f"Custom ({most_common_diff.days} days)"
        except:
            return "Unknown"
    
    def remove_outliers(self, df, numeric_columns, method='iqr'):
        """
        Remove outliers from numeric columns
        """
        outlier_info = []
        original_length = len(df)
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0 and outlier_count < len(df) * 0.1:  # Remove only if less than 10%
                    df = df[~outliers]
                    outlier_info.append(f"🧹 Removed {outlier_count} outliers from '{col}'")
        
        if len(df) < original_length:
            outlier_info.append(f"📊 Dataset size reduced from {original_length} to {len(df)} rows")
        
        return df, outlier_info
    
    def preprocess_data(self, df):
        """
        Main preprocessing function that orchestrates all cleaning steps
        """
        self.preprocessing_report = []
        
        try:
            # Make a copy to avoid modifying original
            df_processed = df.copy()
            
            # Detect date columns
            date_columns = self.detect_date_column(df_processed)
            if not date_columns:
                raise Exception("No date column detected. Please ensure your data has a proper date column.")
            
            # Use the first detected date column
            primary_date_col = date_columns[0]
            self.preprocessing_report.append(f"📅 Using '{primary_date_col}' as the primary date column")
            
            # Convert date column
            df_processed, date_msg = self.convert_date_column(df_processed, primary_date_col)
            self.preprocessing_report.append(date_msg)
            
            # Detect frequency
            frequency = self.detect_data_frequency(df_processed, primary_date_col)
            self.preprocessing_report.append(f"📈 Detected data frequency: {frequency}")
            
            # Detect KPI columns
            kpi_columns = self.detect_kpi_columns(df_processed, exclude_columns=date_columns)
            if not kpi_columns:
                raise Exception("No numeric KPI columns detected. Please ensure your data has numeric values to forecast.")
            
            self.preprocessing_report.append(f"🎯 Detected KPI columns: {', '.join(kpi_columns)}")
            
            # Handle missing values
            df_processed, missing_info = self.handle_missing_values(df_processed, kpi_columns)
            self.preprocessing_report.extend(missing_info)
            
            # Remove outliers (optional, conservative approach)
            df_processed, outlier_info = self.remove_outliers(df_processed, kpi_columns)
            self.preprocessing_report.extend(outlier_info)
            
            # Ensure we have enough data points
            if len(df_processed) < 10:
                raise Exception("Insufficient data points after preprocessing. Please provide more historical data.")
            
            # Add metadata to the dataframe
            df_processed.attrs['date_column'] = primary_date_col
            df_processed.attrs['kpi_columns'] = kpi_columns
            df_processed.attrs['frequency'] = frequency
            
            self.preprocessing_report.append(f"✅ Preprocessing completed successfully!")
            
            return df_processed, self.preprocessing_report
            
        except Exception as e:
            raise Exception(f"Preprocessing failed: {str(e)}")
    
    def prepare_for_modeling(self, df, target_column=None):
        """
        Prepare data specifically for machine learning models
        """
        df_model = df.copy()
        
        # If no target specified, use the first KPI column
        if target_column is None:
            target_column = df.attrs.get('kpi_columns', [df.select_dtypes(include=[np.number]).columns[0]])[0]
        
        # Create time-based features
        date_col = df.attrs.get('date_column')
        if date_col:
            df_model['year'] = df_model[date_col].dt.year
            df_model['month'] = df_model[date_col].dt.month
            df_model['day'] = df_model[date_col].dt.day
            df_model['dayofweek'] = df_model[date_col].dt.dayofweek
            df_model['quarter'] = df_model[date_col].dt.quarter
        
        return df_model, target_column
