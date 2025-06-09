import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DataVisualizer:
    """
    Creates various visualizations for KPI data analysis and forecasting
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_time_series_plot(self, df):
        """
        Create interactive time series plot for historical KPI data
        """
        date_col = df.attrs.get('date_column')
        kpi_columns = df.attrs.get('kpi_columns', [])
        
        if not date_col or not kpi_columns:
            # Fallback: detect columns automatically
            date_col = df.select_dtypes(include=['datetime64']).columns[0] if len(df.select_dtypes(include=['datetime64']).columns) > 0 else df.columns[0]
            kpi_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        fig = go.Figure()
        
        # Add traces for each KPI column
        for i, col in enumerate(kpi_columns[:5]):  # Limit to 5 KPIs for readability
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(color=self.color_palette[i % len(self.color_palette)]),
                hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Historical KPI Trends",
            xaxis_title="Date",
            yaxis_title="KPI Value",
            hovermode='x unified',
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_distribution_plots(self, df):
        """
        Create distribution plots for numeric columns
        """
        kpi_columns = df.attrs.get('kpi_columns', df.select_dtypes(include=[np.number]).columns.tolist())
        figures = []
        
        # Create subplots for multiple KPIs
        if len(kpi_columns) > 1:
            fig = make_subplots(
                rows=min(2, len(kpi_columns)),
                cols=2,
                subplot_titles=[f"{col} Distribution" for col in kpi_columns[:4]],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]] * min(2, len(kpi_columns))
            )
            
            for i, col in enumerate(kpi_columns[:4]):
                row = (i // 2) + 1
                col_idx = (i % 2) + 1
                
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        name=f"{col}",
                        nbinsx=30,
                        opacity=0.7,
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=row, col=col_idx
                )
            
            fig.update_layout(
                title="KPI Value Distributions",
                height=600,
                template="plotly_white",
                showlegend=False
            )
            
            figures.append(fig)
        
        # Box plots for outlier detection
        if len(kpi_columns) <= 5:
            box_fig = go.Figure()
            
            for i, col in enumerate(kpi_columns):
                box_fig.add_trace(go.Box(
                    y=df[col],
                    name=col,
                    marker_color=self.color_palette[i % len(self.color_palette)],
                    boxpoints='outliers'
                ))
            
            box_fig.update_layout(
                title="Box Plots - Outlier Detection",
                yaxis_title="Values",
                template="plotly_white",
                height=400
            )
            
            figures.append(box_fig)
        
        return figures
    
    def create_correlation_heatmap(self, df):
        """
        Create correlation heatmap for numeric columns
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_forecast_plot(self, historical_data, forecast_data, model_name):
        """
        Create combined plot showing historical data and forecasts
        """
        date_col = historical_data.attrs.get('date_column')
        kpi_columns = historical_data.attrs.get('kpi_columns', [])
        
        if not date_col:
            date_col = historical_data.select_dtypes(include=['datetime64']).columns[0]
        
        if not kpi_columns:
            kpi_columns = [historical_data.select_dtypes(include=[np.number]).columns[0]]
        
        # Use the first KPI column for the main forecast
        main_kpi = kpi_columns[0]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data[date_col],
            y=historical_data[main_kpi],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
        ))
        
        # Forecast data
        forecast_df = pd.DataFrame(forecast_data)
        if 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
            ))
            
            # Add confidence intervals if available
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    name='Confidence Interval',
                    hovertemplate='<b>Confidence Interval</b><br>Date: %{x}<br>Lower: %{y:,.2f}<extra></extra>'
                ))
        
        # Note: Vertical line removed due to timestamp compatibility issues with Plotly
        
        fig.update_layout(
            title=f"KPI Forecast using {model_name}",
            xaxis_title="Date",
            yaxis_title=f"{main_kpi}",
            hovermode='x unified',
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def create_model_performance_plot(self, actual_values, predicted_values, model_name):
        """
        Create actual vs predicted scatter plot
        """
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual_values,
            y=predicted_values,
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                color=self.color_palette[0],
                opacity=0.6
            ),
            hovertemplate='<b>Actual:</b> %{x:,.2f}<br><b>Predicted:</b> %{y:,.2f}<extra></extra>'
        ))
        
        # Perfect prediction line
        min_val = min(min(actual_values), min(predicted_values))
        max_val = max(max(actual_values), max(predicted_values))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash'),
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f"Model Performance: {model_name}",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_names, importance_values, model_name):
        """
        Create feature importance plot
        """
        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=self.color_palette[0],
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Feature Importance - {model_name}",
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            height=400
        )
        
        return fig
