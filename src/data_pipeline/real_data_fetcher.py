"""
real_data_fetcher.py
--------------------
Downloads real-world satellite and meteorological data.

1. Google Earth Engine (GEE): Landsat 8/9 for LST, NDVI, Albedo
2. Copernicus CDS: ERA5-Land for Air Temp, Humidity, Wind Speed
"""

import os
import time
import urllib.request

try:
    import ee
    import cdsapi
except ImportError:
    print("Missing libraries. Run: pip install earthengine-api cdsapi")

# Ensure directories exist
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_gee_data(city_name="Delhi", bbox=[76.84, 28.40, 77.34, 28.88]):
    """
    Connects to Google Earth Engine to download Landsat 8/9 data.
    Requires running `earthengine authenticate` in terminal first.
    """
    print(f"\n[GEE] Authenticating and Initializing Google Earth Engine for {city_name}...")
    try:
        ee.Initialize(project='sheetsapiproject-485018')
    except Exception as e:
        print("[ERROR] GEE Initialization failed. Please run 'earthengine authenticate' in terminal.")
        print(e)
        return

    # Define Area of Interest (Bounding Box)
    aoi = ee.Geometry.Rectangle(bbox)

    # 1. Fetch Landsat 9 Surface Reflectance (for NDVI and Albedo)
    print("[GEE] Querying Landsat 9 for latest clear-sky images...")
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(aoi)
          .filterDate('2023-01-01', '2023-12-31')
          .filter(ee.Filter.lt('CLOUD_COVER', 5))
          .median()
          .clip(aoi))

    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = l9.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    
    # Get LST (Thermal Band 10), convert from Kelvin to Celsius
    lst = l9.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

    # Combine bands
    dataset = ee.Image.cat([lst, ndvi])

    # Generate Download URL
    print("[GEE] Generating GeoTIFF download link...")
    url = dataset.getDownloadUrl({
        'scale': 100, # 100m resolution to match our grid
        'crs': 'EPSG:4326',
        'region': aoi,
        'format': 'GEO_TIFF'
    })
    
    out_path = os.path.join(RAW_DIR, f"{city_name.lower()}_landsat.tif")
    print(f"[GEE] Downloading {out_path}...")
    urllib.request.urlretrieve(url, out_path)
    print("[GEE] Download Complete!")


def download_era5_data(city_name="Delhi", bbox=[28.88, 76.84, 28.40, 77.34]):
    """
    Connects to Copernicus CDS API to download ERA5-Land weather data.
    Requires a ~/.cdsapirc file with your UID and API Key.
    """
    print(f"\n[CDS] Connecting to Copernicus Climate Data Store for {city_name}...")
    out_path = os.path.join(RAW_DIR, f"{city_name.lower()}_era5.nc")
    
    try:
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    '2m_temperature', 
                    '2m_dewpoint_temperature', 
                    '10m_u_component_of_wind', 
                    '10m_v_component_of_wind'
                ],
                'year': '2023',
                'month': '05', # Hottest month
                'day': '15',
                'time': '14:00', # Peak heat hour
                'area': bbox, # North, West, South, East
                'format': 'netcdf',
            },
            out_path)
        print(f"[CDS] Download Complete: {out_path}")
    except Exception as e:
        print("[ERROR] CDS API failed. Did you set up the .cdsapirc file?")
        print(e)


if __name__ == "__main__":
    print("===================================================")
    print("  URBAN HEAT MITIGATION: REAL DATA FETCHER")
    print("===================================================")
    
    download_gee_data("Delhi")
    download_era5_data("Delhi")