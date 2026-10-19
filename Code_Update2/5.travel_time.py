import pandas as pd
import geopandas as gpd
import os
from road_speed import SPEED_KMH, DEFAULT_SPEED_KMH, travel_time_min
road_path = ""
road_gdf = gpd.read_file(road_path)
road_gdf = road_gdf.to_crs(epsg=4326)
regions_path = ""
regions_gdf = gpd.read_file(regions_path)
road_gdf['length_m'] = float('nan')
for city_name, idx in road_gdf.groupby('市').groups.items():
    city_crs = regions_gdf[regions_gdf['市'] == city_name].estimate_utm_crs()
    road_gdf.loc[idx, 'length_m'] = road_gdf.loc[idx].geometry.to_crs(city_crs).length
road_gdf['speed'] = road_gdf['road_type'].map(SPEED_KMH).fillna(DEFAULT_SPEED_KMH)
road_gdf['time_m'] = travel_time_min(road_gdf['length_m'], road_gdf['speed'])
output_path = ""
road_gdf.to_file(output_path)
