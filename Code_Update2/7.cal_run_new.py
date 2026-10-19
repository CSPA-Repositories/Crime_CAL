import pandas as pd
import geopandas as gpd
import shapely
from pyproj import CRS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import os
import numpy as np
import gc
from tqdm import tqdm
city_list_df = pd.read_csv("")
city_list = city_list_df['地区'].to_list()
police_station_path = ""
police_df = pd.read_csv(police_station_path)
police_df = police_df[~police_df.drop(columns=['Unnamed: 0', 'updatetime']).duplicated()]
road_folder = ""
regions_path = ""
regions_gdf = gpd.read_file(regions_path)
pop_path_folder = ""
res_path = ""
if os.path.exists(res_path) == False:
    os.makedirs(res_path)
print("Data loaded.")
WALK_SPEED_KMH = 5
EPS_MIN = 1e-9
def snap_to_edges(points, edge_tree, edge_geom, edge_time):
    pi, ei = edge_tree.query_nearest(points, all_matches=False)
    e = np.empty(len(points), dtype=np.int64)
    e[pi] = ei
    dist = shapely.distance(points, edge_geom[e])
    frac_u = shapely.line_locate_point(edge_geom[e], points, normalized=True)
    return e, dist, frac_u, frac_u * edge_time[e], (1 - frac_u) * edge_time[e]
for city in tqdm(city_list):
    city_name = city + '市'
    print(f'Processing city: {city_name}')
    city_pop = gpd.read_parquet(f'{pop_path_folder}/pop_with_crime_{city}.parquet')
    city_crs = regions_gdf[regions_gdf['市'] == city_name].estimate_utm_crs()
    city_region = regions_gdf[regions_gdf['市'] == city_name].to_crs(city_crs)
    city_pop = city_pop.to_crs(city_crs)
    country_list = city_region['县'].unique().tolist()
    for country in tqdm(country_list):
        print(f'  Processing country: {country}')
        town_list = city_region[city_region['县'] == country]['乡'].unique().tolist()
        has_done = 0
        for town in town_list:
            res_file_path = f'{res_path}/{city}_{country}_{town}_with_police.parquet'
            if not os.path.exists(res_file_path):
                break
            has_done += 1
        if has_done == len(town_list):
            print(f'    已存在所有乡镇结果文件，跳过 {city}_{country}')
            continue
        country_road_path = f'{road_folder}/{city}_{country}_road_edges.parquet'
        if not os.path.exists(country_road_path):
            print(f'    没有路网文件（该区县没有道路数据），跳过 {city}_{country}')
            continue
        edges = gpd.read_parquet(country_road_path)
        if CRS(edges.crs) != city_crs:
            raise ValueError(f'{country_road_path} 的投影是 {edges.crs}，与本市投影 {city_crs.to_string()} 不一致，请用 6 号脚本重建')
        if edges['time_m'].isna().any():
            raise ValueError(f'{country_road_path} 有 {edges["time_m"].isna().sum()} 条边缺少 time_m，请用 6 号脚本重建')
        edge_u = edges['u'].to_numpy(np.int64)
        edge_v = edges['v'].to_numpy(np.int64)
        edge_time = edges['time_m'].to_numpy(float)
        edge_geom = edges.geometry.to_numpy()
        edge_tree = shapely.STRtree(edge_geom)
        n_nodes = int(max(edge_u.max(), edge_v.max())) + 1
        loop = edge_u == edge_v
        road_rows = np.r_[edge_u[~loop], edge_v[~loop]]
        road_cols = np.r_[edge_v[~loop], edge_u[~loop]]
        road_w = np.r_[edge_time[~loop], edge_time[~loop]]
        del edges
        last_police_key, reach, reach_src = None, None, None
        for town in town_list:
            print(f'  Processing town: {town}')
            res_file_path = f'{res_path}/{city}_{country}_{town}_with_police.parquet'
            if os.path.exists(res_file_path):
                print(f'    已存在结果文件，跳过 {city}_{town}')
                continue
            town_police_df = police_df[(police_df['市'] == city_name) & (police_df['县'] == country) & (police_df['乡'] == town)]
            if town_police_df.empty:
                print('    No police station found in town, searching in county...')
                town_region = city_region[(city_region['县'] == country) & (city_region['乡'] == town)]
                town_centroid = town_region.geometry.centroid.iloc[0] if not town_region.empty else None
                if town_centroid is not None:
                    police_in_country = police_df[(police_df['市'] == city_name) & (police_df['县'] == country)]
                    if not police_in_country.empty:
                        police_in_country_gdf = gpd.GeoDataFrame(police_in_country, geometry=gpd.points_from_xy(police_in_country['lon'], police_in_country['lat']), crs='EPSG:4326')
                        police_in_country_gdf = police_in_country_gdf.to_crs(city_crs)
                        nearest_police_idx = police_in_country_gdf.geometry.distance(town_centroid).idxmin()
                        town_police_df = police_df.loc[[nearest_police_idx]]
                    else:
                        print(f'    No police station found in county {country}. Skipping...')
                        continue
                else:
                    print(f'    No geometry for town {town}. Skipping...')
                    continue
            town_pop_df = city_pop[(city_pop['县'] == country) & (city_pop['乡'] == town)].copy()
            if town_pop_df.empty:
                print(f'    No population data found in {town}. Skipping...')
                continue
            police_key = tuple(sorted(town_police_df.index))
            if police_key != last_police_key:
                police_pts = gpd.points_from_xy(town_police_df['lon'], town_police_df['lat'], crs='EPSG:4326').to_crs(city_crs)
                pol_e, pol_dist, pol_frac_u, pol_tu, pol_tv = snap_to_edges(
                    np.asarray(police_pts), edge_tree, edge_geom, edge_time)
                pol_walk = pol_dist / (WALK_SPEED_KMH * 1000 / 60)
                pol_id = town_police_df['id'].to_numpy()
                k = np.arange(len(pol_e))
                src = pd.DataFrame({'from': np.r_[k, k] + n_nodes, 'to': np.r_[edge_u[pol_e], edge_v[pol_e]],
                                    'w': np.r_[pol_walk + pol_tu, pol_walk + pol_tv] + EPS_MIN})
                src = src.groupby(['from', 'to'], as_index=False)['w'].min()
                n_all = n_nodes + len(pol_e)
                graph = csr_matrix(
                    (np.r_[road_w, src['w'].values], (np.r_[road_rows, src['from'].values], np.r_[road_cols, src['to'].values])),
                    shape=(n_all, n_all))
                reach, _, reach_src = dijkstra(graph, directed=True, indices=k + n_nodes, min_only=True, return_predecessors=True)
                last_police_key = police_key
                print(f'    派出所 {len(town_police_df)} 个，到路段的最大吸附距离 {pol_dist.max():.0f} m')
            e, dist, frac_u, tu, tv = snap_to_edges(
                np.asarray(town_pop_df.geometry.values), edge_tree, edge_geom, edge_time)
            via_u, via_v = reach[edge_u[e]] + tu, reach[edge_v[e]] + tv
            police_time = np.minimum(via_u, via_v)
            nearest = np.where(via_u <= via_v, reach_src[edge_u[e]], reach_src[edge_v[e]]) - n_nodes
            for kk, (pe, pf) in enumerate(zip(pol_e, pol_frac_u)):
                same = np.where(e == pe)[0]
                if len(same):
                    direct = pol_walk[kk] + np.abs(frac_u[same] - pf) * edge_time[pe]
                    better = direct < police_time[same]
                    police_time[same[better]] = direct[better]
                    nearest[same[better]] = kk
            if not np.isfinite(police_time).all():
                raise ValueError(f'{city}_{country}_{town} 有格网无法到达派出所，请检查路网连通性')
            town_pop_df["police_id"] = pol_id[nearest]
            town_pop_df["snap_dist_m"] = dist
            town_pop_df["walk_time"] = dist / (WALK_SPEED_KMH * 1000 / 60)
            town_pop_df["police_snap_dist_m"] = pol_dist[nearest]
            town_pop_df["police_walk_time"] = pol_walk[nearest]
            town_pop_df["network_time"] = police_time - pol_walk[nearest]
            town_pop_df["total_time"] = town_pop_df["police_walk_time"] + town_pop_df["network_time"] + town_pop_df["walk_time"]
            town_pop_df.to_crs(epsg=4326).to_parquet(res_file_path, index=False)
            del town_pop_df
        del edge_tree, edge_geom, reach
        gc.collect()
