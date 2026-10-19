import pandas as pd
import geopandas as gpd
import numpy as np
import pyogrio
from sklearn.neighbors import KernelDensity
import time
import os
regions_path = ""
regions_gdf = gpd.read_file(regions_path)
region_towns = regions_gdf[['市', '县', '乡']]
print("Starting to process cities...")
city_list_path = ""
city_list = pd.read_csv(city_list_path)
city_list = city_list['地区'].to_list()
pop_data_path = ""
pop_index = pyogrio.read_dataframe(pop_data_path, columns=['市'], read_geometry=False, use_arrow=True)
city_fids = pop_index.groupby('市').indices
del pop_index
print("Population index loaded.")
res_folder = ""
os.makedirs(res_folder, exist_ok=True)
crime_path = ""
crime_data = pd.read_csv(crime_path)
coords = gpd.points_from_xy(crime_data['longitude'], crime_data['latitude'])
crime_gdf = gpd.GeoDataFrame(crime_data, geometry=coords, crs='EPSG:4326')
print("Crime data loaded.")
for city in city_list:
    print(f'Processing city: {city}')
    city_name = city+'市'
    fids = city_fids.get(city_name)
    if fids is None:
        print(f'Skip {city_name}, no population data.')
        continue
    pop_city = pyogrio.read_dataframe(pop_data_path, fids=fids, columns=['VALUE', '省', '市', '县', '乡']).to_crs(epsg=4326)
    city_towns = region_towns[region_towns['市'] == city_name]
    in_region = pd.MultiIndex.from_frame(pop_city[['县', '乡']]).isin(pd.MultiIndex.from_frame(city_towns[['县', '乡']]))
    pop_city = pop_city[in_region]
    crime_city = crime_gdf[crime_gdf['city'] == city_name]
    city_crs = regions_gdf[regions_gdf['市'] == city_name].estimate_utm_crs()
    pop_city = pop_city.to_crs(city_crs)
    crime_city = crime_city.to_crs(city_crs)
    pop_city = pop_city.reset_index(drop=True)
    crime_city = crime_city.reset_index(drop=True)
    print(len(pop_city), len(crime_city))
    if len(pop_city) == 0 or len(crime_city) == 0:
        print(f'Skip {city_name}, no data.')
        print(len(pop_city), len(crime_city))
        time.sleep(1)
        continue
    kde = KernelDensity(bandwidth=500, kernel='gaussian', rtol=1e-8)
    crime_coords = list(zip(crime_city.geometry.x, crime_city.geometry.y))
    kde.fit(crime_coords)
    pop_coords = list(zip(pop_city.geometry.x, pop_city.geometry.y))
    log_dens = kde.score_samples(pop_coords)
    pop_city['crime_density'] = np.exp(log_dens) * 1e6
    tail_95 = pop_city['crime_density'].quantile(0.95)
    pop_city['crime_density_tail95'] = pop_city['crime_density'].clip(upper=tail_95)
    pop_city['crime_density_norm'] = ((pop_city['crime_density_tail95'] - pop_city['crime_density_tail95'].min()) / (pop_city['crime_density_tail95'].max() - pop_city['crime_density_tail95'].min())).fillna(0)
    pop_city.to_crs(epsg=4326).to_parquet(f'{res_folder}/pop_with_crime_{city}.parquet', index=False)
