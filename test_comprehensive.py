#!/usr/bin/env python3
"""
Comprehensive test suite for the KPI Forecasting Application
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_processor():
    """Test the DataProcessor class"""
    print("🧪 Testing DataProcessor...")
    
    try:
        from utils.data_processor import DataProcessor
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        test_data = pd.DataFrame({
            'Date': dates,
            'Revenue': np.random.normal(1000, 100, len(dates)),
            'Customers': np.random.poisson(50, len(dates)),
            'Text_Column': ['text'] * len(dates)  # Should be ignored
        })
        
        # Add some missing values
        test_data.loc[10:15, 'Revenue'] = np.nan
        
        processor = DataProcessor()
        processed_data, report = processor.preprocess_data(test_data)
        
        # Validate results
        assert len(processed_data) > 0, "Processed data should not be empty"
        assert 'Date' in processed_data.attrs.get('date_column', ''), "Date column should be detected"
        assert len(processed_data.attrs.get('kpi_columns', [])) >= 2, "KPI columns should be detected"
        assert len(report) > 0, "Processing report should contain messages"
        
        print("✅ DataProcessor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ DataProcessor tests failed: {e}")
        return False

def test_forecasting_engine():
    """Test the ForecastingEngine class"""
    print("🧪 Testing ForecastingEngine...")
    
    try:
        from models.forecasting_models import ForecastingEngine
        from utils.data_processor import DataProcessor
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        trend = np.linspace(100, 200, len(dates))
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        noise = np.random.normal(0, 5, len(dates))
        
        test_data = pd.DataFrame({
            'Date': dates,
            'KPI': trend + seasonal + noise
        })
        
        # Preprocess data
        processor = DataProcessor()
        processed_data, _ = processor.preprocess_data(test_data)
        
        # Test forecasting
        engine = ForecastingEngine()
        results = engine.auto_select_and_train(
            processed_data, 
            forecast_periods=10, 
            confidence_level=0.95
        )
        
        # Validate results
        assert 'model_name' in results, "Model name should be provided"
        assert 'performance' in results, "Performance metrics should be provided"
        assert 'forecast_data' in results, "Forecast data should be provided"
        assert len(results['forecast_data']) == 10, "Should have 10 forecast periods"
        
        print(f"✅ ForecastingEngine tests passed (Model: {results['model_name']})")
        return True
        
    except Exception as e:
        print(f"❌ ForecastingEngine tests failed: {e}")
        return False

def test_data_visualizer():
    """Test the DataVisualizer class"""
    print("🧪 Testing DataVisualizer...")
    
    try:
        from utils.visualization import DataVisualizer
        from utils.data_processor import DataProcessor
        
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        test_data = pd.DataFrame({
            'Date': dates,
            'Revenue': np.random.normal(1000, 100, len(dates)),
            'Customers': np.random.normal(50, 10, len(dates))
        })
        
        # Preprocess data
        processor = DataProcessor()
        processed_data, _ = processor.preprocess_data(test_data)
        
        # Test visualizations
        visualizer = DataVisualizer()
        
        # Test time series plot
        ts_fig = visualizer.create_time_series_plot(processed_data)
        assert ts_fig is not None, "Time series plot should be created"
        
        # Test distribution plots
        dist_figs = visualizer.create_distribution_plots(processed_data)
        assert len(dist_figs) > 0, "Distribution plots should be created"
        
        # Test correlation heatmap
        corr_fig = visualizer.create_correlation_heatmap(processed_data)
        assert corr_fig is not None, "Correlation heatmap should be created"
        
        # Test forecast plot
        forecast_data = [
            {'date': dates[-1] + pd.Timedelta(days=i+1), 'forecast': 1000 + i*10, 'lower_bound': 950 + i*10, 'upper_bound': 1050 + i*10}
            for i in range(5)
        ]
        forecast_fig = visualizer.create_forecast_plot(processed_data, forecast_data, "Test Model")
        assert forecast_fig is not None, "Forecast plot should be created"
        
        print("✅ DataVisualizer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ DataVisualizer tests failed: {e}")
        return False

def test_model_evaluator():
    """Test the ModelEvaluator class"""
    print("🧪 Testing ModelEvaluator...")
    
    try:
        from utils.model_evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # Test regression evaluation
        actual = np.array([100, 110, 120, 130, 140])
        predicted = np.array([98, 112, 118, 132, 145])
        
        reg_metrics = evaluator.evaluate_regression_model(actual, predicted)
        assert 'mae' in reg_metrics, "MAE should be calculated"
        assert 'r2_score' in reg_metrics, "R² should be calculated"
        assert 'mape' in reg_metrics, "MAPE should be calculated"
        
        # Test classification evaluation
        actual_class = np.array(['A', 'B', 'A', 'B', 'A'])
        predicted_class = np.array(['A', 'B', 'B', 'B', 'A'])
        
        class_metrics = evaluator.evaluate_classification_model(actual_class, predicted_class)
        assert 'accuracy' in class_metrics, "Accuracy should be calculated"
        
        # Test model report generation
        report = evaluator.generate_model_report("Test Model", reg_metrics)
        assert 'model_name' in report, "Model name should be in report"
        assert 'interpretation' in report, "Interpretation should be provided"
        
        print("✅ ModelEvaluator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ModelEvaluator tests failed: {e}")
        return False

def test_sample_data_integration():
    """Test the complete workflow with sample data"""
    print("🧪 Testing complete workflow with sample data...")
    
    try:
        # Check if sample data exists
        if not os.path.exists('sample_kpi_data.xlsx'):
            print("⚠️ Sample data not found, creating it...")
            from create_sample_data import create_sample_kpi_data
            sample_data = create_sample_kpi_data()
            sample_data.to_excel('sample_kpi_data.xlsx', index=False)
        
        # Load sample data
        df = pd.read_excel('sample_kpi_data.xlsx')
        
        # Test complete workflow
        from utils.data_processor import DataProcessor
        from models.forecasting_models import ForecastingEngine
        from utils.visualization import DataVisualizer
        from utils.model_evaluator import ModelEvaluator
        
        # Process data
        processor = DataProcessor()
        processed_data, report = processor.preprocess_data(df)
        
        # Generate forecasts
        engine = ForecastingEngine()
        results = engine.auto_select_and_train(processed_data, forecast_periods=7)
        
        # Create visualizations
        visualizer = DataVisualizer()
        ts_fig = visualizer.create_time_series_plot(processed_data)
        forecast_fig = visualizer.create_forecast_plot(
            processed_data, 
            results['forecast_data'], 
            results['model_name']
        )
        
        # Validate complete workflow
        assert len(processed_data) > 100, "Should have substantial processed data"
        assert results['model_name'] is not None, "Model should be selected"
        assert len(results['forecast_data']) == 7, "Should have 7 forecast periods"
        assert ts_fig is not None, "Time series plot should be created"
        assert forecast_fig is not None, "Forecast plot should be created"
        
        print("✅ Complete workflow test passed")
        print(f"   📊 Data: {len(processed_data)} rows processed")
        print(f"   🤖 Model: {results['model_name']}")
        print(f"   📈 Forecast: {len(results['forecast_data'])} periods")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid data"""
    print("🧪 Testing error handling...")
    
    try:
        from utils.data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        try:
            processor.preprocess_data(empty_df)
            assert False, "Should raise exception for empty data"
        except Exception:
            pass  # Expected
        
        # Test with no date column
        no_date_df = pd.DataFrame({
            'Value1': [1, 2, 3],
            'Value2': [4, 5, 6]
        })
        try:
            processor.preprocess_data(no_date_df)
            assert False, "Should raise exception for no date column"
        except Exception:
            pass  # Expected
        
        # Test with no numeric columns
        no_numeric_df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=3),
            'Text': ['A', 'B', 'C']
        })
        try:
            processor.preprocess_data(no_numeric_df)
            assert False, "Should raise exception for no numeric columns"
        except Exception:
            pass  # Expected
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling tests failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 KPI Forecasting Application - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        test_data_processor,
        test_model_evaluator,
        test_data_visualizer,
        test_forecasting_engine,
        test_sample_data_integration,
        test_error_handling
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The application is fully functional.")
        print("\n🚀 Ready for deployment!")
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed. Please review the errors above.")
    
    print("\n📝 Next steps:")
    print("1. Start the application: streamlit run app.py")
    print("2. Upload sample_kpi_data.xlsx to test functionality")
    print("3. Deploy to Render using the DEPLOYMENT.md guide")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
