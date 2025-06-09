import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

class ModelEvaluator:
    """
    Evaluate model performance and provide metrics
    """
    
    def __init__(self):
        pass
    
    def evaluate_regression_model(self, actual, predicted):
        """
        Calculate regression metrics
        """
        metrics = {}
        
        try:
            # Remove any NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                return {'error': 'No valid predictions to evaluate'}
            
            # Calculate metrics
            metrics['mae'] = mean_absolute_error(actual_clean, predicted_clean)
            metrics['mse'] = mean_squared_error(actual_clean, predicted_clean)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2_score'] = r2_score(actual_clean, predicted_clean)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape_values = []
            for i in range(len(actual_clean)):
                if actual_clean[i] != 0:
                    mape_values.append(abs((actual_clean[i] - predicted_clean[i]) / actual_clean[i]) * 100)
            
            if mape_values:
                metrics['mape'] = np.mean(mape_values)
            else:
                metrics['mape'] = 0
            
            # Calculate directional accuracy (for time series)
            if len(actual_clean) > 1:
                actual_direction = np.diff(actual_clean) > 0
                predicted_direction = np.diff(predicted_clean) > 0
                metrics['directional_accuracy'] = np.mean(actual_direction == predicted_direction)
            
        except Exception as e:
            metrics['error'] = f"Error calculating metrics: {str(e)}"
        
        return metrics
    
    def evaluate_classification_model(self, actual, predicted, class_names=None):
        """
        Calculate classification metrics
        """
        metrics = {}
        
        try:
            # Remove any NaN values
            mask = ~(pd.isna(actual) | pd.isna(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                return {'error': 'No valid predictions to evaluate'}
            
            # Calculate accuracy
            metrics['accuracy'] = accuracy_score(actual_clean, predicted_clean)
            
            # Get classification report
            try:
                report = classification_report(actual_clean, predicted_clean, target_names=class_names, output_dict=True)
                metrics['classification_report'] = report
                
                # Extract key metrics
                if 'weighted avg' in report:
                    metrics['precision'] = report['weighted avg']['precision']
                    metrics['recall'] = report['weighted avg']['recall']
                    metrics['f1_score'] = report['weighted avg']['f1-score']
            except:
                pass
            
        except Exception as e:
            metrics['error'] = f"Error calculating metrics: {str(e)}"
        
        return metrics
    
    def evaluate_forecast_quality(self, historical_data, forecast_data, actual_future_data=None):
        """
        Evaluate the quality of forecasts
        """
        evaluation = {}
        
        try:
            # Basic forecast statistics
            if isinstance(forecast_data, list) and len(forecast_data) > 0:
                forecast_values = [item['forecast'] for item in forecast_data if 'forecast' in item]
                
                if forecast_values:
                    evaluation['forecast_mean'] = np.mean(forecast_values)
                    evaluation['forecast_std'] = np.std(forecast_values)
                    evaluation['forecast_min'] = np.min(forecast_values)
                    evaluation['forecast_max'] = np.max(forecast_values)
                    
                    # Calculate forecast trend
                    if len(forecast_values) > 1:
                        trend = np.polyfit(range(len(forecast_values)), forecast_values, 1)[0]
                        evaluation['forecast_trend'] = 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'
                        evaluation['trend_slope'] = trend
            
            # Compare with historical data if available
            if hasattr(historical_data, 'attrs') and 'kpi_columns' in historical_data.attrs:
                kpi_columns = historical_data.attrs['kpi_columns']
                if kpi_columns:
                    historical_values = historical_data[kpi_columns[0]].dropna().values
                    
                    evaluation['historical_mean'] = np.mean(historical_values)
                    evaluation['historical_std'] = np.std(historical_values)
                    
                    # Compare forecast range with historical range
                    if 'forecast_mean' in evaluation:
                        mean_diff_pct = ((evaluation['forecast_mean'] - evaluation['historical_mean']) / evaluation['historical_mean']) * 100
                        evaluation['forecast_vs_historical_change'] = mean_diff_pct
            
            # If actual future data is provided, calculate accuracy
            if actual_future_data is not None and len(forecast_values) > 0:
                actual_values = actual_future_data[:len(forecast_values)]  # Match lengths
                forecast_subset = forecast_values[:len(actual_values)]
                
                if len(actual_values) == len(forecast_subset):
                    accuracy_metrics = self.evaluate_regression_model(
                        np.array(actual_values), 
                        np.array(forecast_subset)
                    )
                    evaluation['forecast_accuracy'] = accuracy_metrics
        
        except Exception as e:
            evaluation['error'] = f"Error evaluating forecast: {str(e)}"
        
        return evaluation
    
    def generate_model_report(self, model_name, performance_metrics, forecast_evaluation=None):
        """
        Generate a comprehensive model report
        """
        report = {
            'model_name': model_name,
            'timestamp': pd.Timestamp.now(),
            'performance_metrics': performance_metrics
        }
        
        if forecast_evaluation:
            report['forecast_evaluation'] = forecast_evaluation
        
        # Add interpretation
        interpretation = []
        
        if 'r2_score' in performance_metrics:
            r2 = performance_metrics['r2_score']
            if r2 > 0.8:
                interpretation.append("Excellent model fit (R² > 0.8)")
            elif r2 > 0.6:
                interpretation.append("Good model fit (R² > 0.6)")
            elif r2 > 0.4:
                interpretation.append("Moderate model fit (R² > 0.4)")
            else:
                interpretation.append("Poor model fit (R² < 0.4)")
        
        if 'mape' in performance_metrics:
            mape = performance_metrics['mape']
            if mape < 10:
                interpretation.append("Highly accurate predictions (MAPE < 10%)")
            elif mape < 20:
                interpretation.append("Good prediction accuracy (MAPE < 20%)")
            elif mape < 50:
                interpretation.append("Moderate prediction accuracy (MAPE < 50%)")
            else:
                interpretation.append("Low prediction accuracy (MAPE > 50%)")
        
        if 'accuracy' in performance_metrics:
            acc = performance_metrics['accuracy']
            if acc > 0.9:
                interpretation.append("Excellent classification accuracy (> 90%)")
            elif acc > 0.8:
                interpretation.append("Good classification accuracy (> 80%)")
            elif acc > 0.7:
                interpretation.append("Moderate classification accuracy (> 70%)")
            else:
                interpretation.append("Low classification accuracy (< 70%)")
        
        report['interpretation'] = interpretation
        
        return report
    
    def compare_models(self, model_results):
        """
        Compare multiple model results and recommend the best one
        """
        if not model_results:
            return None
        
        comparison = {
            'models': [],
            'best_model': None,
            'comparison_criteria': []
        }
        
        for model_name, results in model_results.items():
            model_info = {
                'name': model_name,
                'performance': results.get('performance', {}),
                'score': 0
            }
            
            # Calculate composite score
            performance = results.get('performance', {})
            
            # For regression models
            if 'r2_score' in performance:
                model_info['score'] += performance['r2_score'] * 0.4
            
            if 'mape' in performance:
                # Lower MAPE is better, so we use negative
                mape_score = max(0, 1 - (performance['mape'] / 100))
                model_info['score'] += mape_score * 0.3
            
            if 'mae' in performance:
                # Normalize MAE (this is a simplified approach)
                mae_score = 1 / (1 + performance['mae'])
                model_info['score'] += mae_score * 0.3
            
            # For classification models
            if 'accuracy' in performance:
                model_info['score'] += performance['accuracy'] * 0.7
            
            comparison['models'].append(model_info)
        
        # Sort by score and select best
        comparison['models'].sort(key=lambda x: x['score'], reverse=True)
        if comparison['models']:
            comparison['best_model'] = comparison['models'][0]['name']
        
        return comparison
