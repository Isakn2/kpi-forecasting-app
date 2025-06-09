# Deployment Guide for Render

## Prerequisites

1. A Render account (https://render.com)
2. Your code pushed to a GitHub repository

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository contains all the files from this KPI forecasting application:

```
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── start.sh                  # Startup script for Render
├── README.md                 # Documentation
├── create_sample_data.py     # Sample data generator
├── sample_kpi_data.xlsx      # Sample dataset
├── .streamlit/
│   ├── config.toml           # Local development config
│   └── config_production.toml # Production config
├── models/
│   ├── __init__.py
│   └── forecasting_models.py # ML models
└── utils/
    ├── __init__.py
    ├── data_processor.py     # Data preprocessing
    ├── model_evaluator.py    # Model evaluation
    └── visualization.py      # Plotting functions
```

### 2. Create a Web Service on Render

1. **Login to Render**: Go to https://render.com and sign in
2. **Create New Web Service**: Click "New +" and select "Web Service"
3. **Connect Repository**: Connect your GitHub repository containing this code
4. **Configure Service**:
   - **Name**: `kpi-forecasting-app` (or your preferred name)
   - **Environment**: `Python 3`
   - **Region**: Choose your preferred region
   - **Branch**: `main` (or your default branch)
   - **Build Command**: Leave empty (handled by start.sh)
   - **Start Command**: `./start.sh`

### 3. Environment Variables

No special environment variables are required for this application.

### 4. Advanced Settings

- **Auto-Deploy**: Enable to automatically deploy on git pushes
- **Instance Type**: Free tier is sufficient for testing, Starter for production use

## Configuration Details

### Memory and CPU Requirements

- **Minimum**: 512MB RAM, 0.1 CPU (Free tier)
- **Recommended**: 1GB RAM, 0.5 CPU (Starter tier)
- **For large datasets**: 2GB+ RAM (Standard tier)

### Startup Process

The `start.sh` script handles the deployment process:

1. Installs Python dependencies from `requirements.txt`
2. Copies production configuration
3. Starts the Streamlit application on the port provided by Render

### Expected Startup Time

- **Initial Deploy**: 3-5 minutes (dependency installation)
- **Subsequent Deploys**: 1-2 minutes
- **Cold Start**: 10-30 seconds

## Troubleshooting

### Common Issues

1. **Build Fails Due to Dependencies**
   - Check `requirements.txt` for version conflicts
   - Consider using `>=` instead of `==` for version specifications

2. **Application Won't Start**
   - Verify `start.sh` has executable permissions
   - Check that `app.py` is in the root directory

3. **Memory Issues**
   - Upgrade to a higher tier if you see memory-related crashes
   - Consider optimizing data processing for large files

4. **Slow Performance**
   - Upload smaller files for testing
   - Consider implementing data sampling for very large datasets

### Monitoring

- **Logs**: Available in the Render dashboard under your service
- **Metrics**: CPU and memory usage visible in dashboard
- **Health Checks**: Render automatically monitors your application

## Cost Estimates

- **Free Tier**: $0/month (limited hours, sleeps after inactivity)
- **Starter**: $7/month (always on, 512MB RAM)
- **Standard**: $25/month (1GB RAM, better performance)

## Security Considerations

1. **Data Privacy**: Files uploaded are processed in memory and not permanently stored
2. **HTTPS**: Automatically provided by Render
3. **Environment Isolation**: Each deployment runs in an isolated container

## Custom Domain (Optional)

1. Go to your service settings in Render
2. Add your custom domain
3. Configure DNS as instructed by Render

## Monitoring and Maintenance

### Regular Tasks

1. **Monitor Usage**: Check Render dashboard for resource usage
2. **Update Dependencies**: Regularly update `requirements.txt`
3. **Review Logs**: Check for any errors or performance issues

### Scaling

If you need to handle more users or larger datasets:

1. Upgrade to a higher instance type
2. Consider implementing caching for frequently processed data
3. Add horizontal scaling if needed (multiple instances)

## Support

- **Render Documentation**: https://render.com/docs
- **Streamlit Documentation**: https://docs.streamlit.io
- **Application Issues**: Check the GitHub repository for this project
