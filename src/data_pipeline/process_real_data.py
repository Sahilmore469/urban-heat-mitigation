import os
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

def process_real_data():
    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
    
    TIF_PATH = os.path.join(RAW_DIR, "delhi_landsat.tif")
    NC_PATH = os.path.join(RAW_DIR, "delhi_era5.nc")

    if not os.path.exists(TIF_PATH) or not os.path.exists(NC_PATH):
        print("[ERROR] Missing raw data files. Did the fetcher script finish successfully?")
        return

    print("1. Reading Landsat 9 Imagery (Extracting a 200x200 City Block)...")
    with rasterio.open(TIF_PATH) as src:
        # Band 1 is LST, Band 2 is NDVI (based on our fetcher script)
        lst_raw = src.read(1)
        ndvi_raw = src.read(2)
        
        height, width = lst_raw.shape
        target_size = 200
        
        # Calculate center crop
        start_y = max(0, (height - target_size) // 2)
        start_x = max(0, (width - target_size) // 2)
        end_y = start_y + target_size
        end_x = start_x + target_size
        
        lst_crop = lst_raw[start_y:end_y, start_x:end_x]
        ndvi_crop = ndvi_raw[start_y:end_y, start_x:end_x]
        
        # Get coordinates for the crop
        cols_grid, rows_grid = np.meshgrid(
            np.arange(start_x, end_x), 
            np.arange(start_y, end_y)
        )
        xs, ys = rasterio.transform.xy(src.transform, rows_grid, cols_grid)
        
        lon = np.array(xs).flatten()
        lat = np.array(ys).flatten()
        lst = lst_crop.flatten()
        ndvi = ndvi_crop.flatten()

    # Clean the satellite data: We MUST keep exactly 40,000 pixels so the 2D map doesn't break.
    # Instead of dropping invalid pixels, we fill them with reasonable defaults.
    lst = np.where((lst < -50) | (lst > 100) | np.isnan(lst), 35.0, lst)
    ndvi = np.where((ndvi < -2) | (ndvi > 2) | np.isnan(ndvi), 0.1, ndvi)

    print("2. Reading Copernicus ERA5 Weather Data...")
    
    import zipfile
    if zipfile.is_zipfile(NC_PATH):
        print("   [!] Copernicus sent a ZIP file, extracting data_0.nc...")
        with zipfile.ZipFile(NC_PATH, 'r') as zip_ref:
            zip_ref.extract("data_0.nc", RAW_DIR)
        NC_PATH = os.path.join(RAW_DIR, "data_0.nc")
        
    ds = xr.open_dataset(NC_PATH)
    
    # Extract temperature and convert Kelvin to Celsius
    t2m = ds['t2m'].mean().item() - 273.15 
    d2m = ds['d2m'].mean().item() - 273.15
    
    # Calculate Wind Speed from U and V vectors
    u10 = ds['u10'].mean().item()
    v10 = ds['v10'].mean().item()
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    # Approximate Relative Humidity from Air Temp and Dewpoint Temp
    rh = 100 * (np.exp((17.625 * d2m)/(243.04 + d2m)) / np.exp((17.625 * t2m)/(243.04 + t2m)))

    print("3. Building Machine Learning DataFrame...")
    df = pd.DataFrame({
        'lat': lat,
        'lon': lon,
        'lst': lst,
        'ndvi': ndvi,
        'air_temp': t2m,
        'humidity': rh,
        'wind_speed': wind_speed
    })

    print("4. Approximating Urban Morphology from Vegetation Density...")
    # Because we didn't download 3D building data, we estimate it based on vegetation density
    df['is_water'] = (df['ndvi'] < 0.0).astype(int)
    df['impervious_fraction'] = np.clip(1.0 - (df['ndvi'] / 0.5), 0, 1)
    df['albedo'] = 0.10 + (df['ndvi'] * 0.15)
    df['building_height'] = df['impervious_fraction'] * 25.0 
    df['svf'] = np.clip(1.0 - (df['building_height'] / 40.0), 0.2, 1.0)
    
    # Assign standard LULC classes based on NDVI/Impervious rules
    conditions = [
        (df['is_water'] == 1),
        (df['ndvi'] > 0.4),
        (df['impervious_fraction'] > 0.7)
    ]
    choices = [0, 1, 5] # Water, Dense Veg, High-Density Urban
    df['lulc'] = np.select(conditions, choices, default=4) # Default to low-density

    print("5. Formatting for Streamlit...")
    # Do not drop or sample anything, otherwise the 2D map geometry breaks!
    # Just fill any potential missing data from the weather grid.
    df = df.bfill().ffill()

    # Save to the processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "real_city_data.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n✅ SUCCESS! Processed {len(df)} real-world locations.")
    print(f"✅ Saved ready-to-use ML data to: {out_path}")


def load_real_city_grid(config: dict) -> dict:
    """Loads the real_city_data.csv and reshapes it into the 2D grid the dashboard expects."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "real_city_data.csv")
    df = pd.read_csv(csv_path)
    
    # FIX: Rename lat and lon to match the exact keys expected by grid_to_dataframe
    df.rename(columns={"lat": "lat_grid", "lon": "lon_grid"}, inplace=True)
    
    # FIX: Uses 'grid' instead of 'city_grid' to match config.yaml
    rows = config["grid"]["rows"]
    cols = config["grid"]["cols"]
    
    data = {}
    for col in df.columns:
        data[col] = df[col].values.reshape((rows, cols))
        
    data["rows"] = rows
    data["cols"] = cols
    data["config"] = config
    return data


if __name__ == "__main__":
    process_real_data()