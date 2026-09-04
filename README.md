git # ET-GEO-Hackathon

## Overview

ET-GEO-Hackathon is a geospatial data science project focused on **Evapotranspiration (ETa) prediction and irrigation optimization** for agricultural applications. The project leverages remote sensing data (Sentinel-2 satellite imagery), weather data, and machine learning to predict crop water demand and optimize irrigation schedules.

## Project Goals

- **Predict Evapotranspiration (ETa)**: Use Random Forest models to forecast actual evapotranspiration based on NDVI, weather patterns, and historical data
- **Optimize Irrigation**: Determine optimal irrigation schedules considering soil moisture, rainfall, field capacity, and water constraints
- **Geospatial Analysis**: Process satellite imagery and vector data for agricultural monitoring across the Tokara region

## Project Structure

```
ET-GEO-Hackathon/
├── backend/
│   ├── notebooks/
│   │   └── data_pipeline.ipynb          # Data processing and exploration pipeline
│   └── scripts/
│       ├── predictive_model.py          # ETa prediction using Random Forest
│       └── optimizer.py                 # Irrigation schedule optimization
├── data/
│   ├── historic_eta_data.csv            # Historical evapotranspiration data
│   ├── historic_weather_data.csv        # Historical weather records
│   ├── monthly-precipitation-*.csv      # Precipitation data for the region
│   ├── 2022_2023/                       # Data for 2022-2023 season
│   ├── 2023_2024/                       # Data for 2023-2024 season
│   ├── 2024_2025/                       # Data for 2024-2025 season
│   │   ├── ETa/                         # Actual evapotranspiration rasters (10m, 3m)
│   │   ├── ETo/                         # Reference evapotranspiration rasters
│   │   ├── Kc/                          # Crop coefficient data with tiles
│   │   ├── NDVI/                        # Normalized Difference Vegetation Index
│   │   └── S2_Tiles/                    # Sentinel-2 satellite imagery tiles
│   └── Shapefiles/
│       └── Tokara_Polygons.*            # Vector field boundary data
├── frontend/                            # Frontend application (in development)
├── requirements.txt                     # Python dependencies
├── install.bat                          # Installation script for Windows
└── README.md                            # This file
```

## Data Description

### Raster Data
- **ETa (10m & 3m)**: Actual evapotranspiration in mm, derived from satellite and weather data
- **ETo (10m & 3m)**: Reference evapotranspiration (grass reference)
- **NDVI (10m & 3m)**: Normalized Difference Vegetation Index indicating vegetation health
- **Kc (10m & 3m)**: Crop coefficient tiles relating actual to reference evapotranspiration

### Vector Data
- **Tokara_Polygons**: Shapefile containing field/block boundaries for the study area

### Time Series
- **Historic ETa/Weather Data**: CSV files with historical observations for model training
- **Sentinel-2 Tiles**: Multi-temporal satellite imagery for monitoring vegetation and land-use changes

## Installation

### Requirements
- Python 3.8+
- Windows (uses `install.bat`) or manually install dependencies

### Quick Start

1. **Clone/navigate to the repository:**
   ```bash
   cd ET-GEO-Hackathon
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or on Windows:
   ```bash
   install.bat
   ```

3. **Verify installation:**
   ```bash
   jupyter notebook
   ```

## Dependencies

### Core Data Science
- `numpy`, `pandas`, `scipy` – numerical and data processing
- `matplotlib`, `seaborn` – visualization
- `scikit-learn` – machine learning (Random Forest models)

### Geospatial
- `geopandas`, `shapely` – vector data processing
- `rasterio`, `xarray`, `rioxarray` – raster data handling
- `pyproj` – coordinate reference system transformations

### Additional
- `jupyter`, `ipykernel` – notebook support
- `openmeteo_requests`, `requests_cache` – weather data APIs

## Usage

### ETa Prediction

Use the `ETaPredictor` class to train a model on historical data and make predictions:

```python
from backend.scripts.predictive_model import ETaPredictor

predictor = ETaPredictor()
# Train on historical data
predictor.fit(X_train, y_train)
# Make predictions
eta_predictions = predictor.predict(X_test)
```

### Irrigation Optimization

Use the `IrrigationOptimizer` to generate optimal irrigation schedules:

```python
from backend.scripts.optimizer import IrrigationOptimizer

optimizer = IrrigationOptimizer(
    block=block_data,
    eta_predictions=predicted_eta,
    horizon_days=7,
    field_capacity=300.0,
    wilting_point=150.0,
    max_irrigation=25.0
)
irrigation_schedule = optimizer.optimize()
```

### Data Processing

Start with the Jupyter notebook for exploratory data analysis and preprocessing:

```bash
jupyter notebook backend/notebooks/data_pipeline.ipynb
```

## Key Features

✅ **Machine Learning**: Random Forest model for ETa prediction  
✅ **Optimization**: Irrigation schedule optimization with soil constraints  
✅ **Geospatial Analysis**: Process satellite imagery and vector field data  
✅ **Time Series**: Multi-year historical data (2022-2025)  
✅ **Remote Sensing**: Sentinel-2 satellite data integration  

## Next Steps

- [ ] Complete frontend application development
- [ ] Deploy predictive models as API endpoints
- [ ] Integrate real-time weather data
- [ ] Add model validation and performance metrics
- [ ] Create visualization dashboards

## License

[To be determined]

## Contact

For questions or contributions, please reach out through the hackathon organizers.