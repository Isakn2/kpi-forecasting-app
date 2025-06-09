import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_kpi_data():
    """
    Create sample KPI data for testing the application
    """
    # Generate date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample KPI data with trends and seasonality
    np.random.seed(42)
    n_days = len(dates)
    
    # Base trend
    trend = np.linspace(1000, 1500, n_days)
    
    # Seasonal component (yearly)
    seasonal = 100 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    
    # Weekly seasonality
    weekly_seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Random noise
    noise = np.random.normal(0, 50, n_days)
    
    # Combine components
    revenue = trend + seasonal + weekly_seasonal + noise
    
    # Create additional KPIs
    customers = (revenue / 25) + np.random.normal(0, 5, n_days)
    conversion_rate = 0.05 + 0.02 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + np.random.normal(0, 0.005, n_days)
    avg_order_value = revenue / customers
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Revenue': np.maximum(revenue, 0),  # Ensure no negative revenue
        'Customers': np.maximum(customers, 0).astype(int),
        'Conversion_Rate': np.clip(conversion_rate, 0, 1),
        'Avg_Order_Value': np.maximum(avg_order_value, 0)
    })
    
    # Add some missing values randomly
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'Revenue'] = np.nan
    
    return df

if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_kpi_data()
    
    # Save to Excel
    sample_data.to_excel('sample_kpi_data.xlsx', index=False)
    print("Sample KPI data created and saved as 'sample_kpi_data.xlsx'")
    print(f"Dataset contains {len(sample_data)} rows with the following columns:")
    print(sample_data.columns.tolist())
    print("\nFirst few rows:")
    print(sample_data.head())
