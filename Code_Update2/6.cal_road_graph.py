import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
import os
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import gc
from tqdm import tqdm
from road_speed import SPEED_KMH, DEFAULT_SPEED_KMH, CONNECTOR_SPEED_KMH, travel_time_min
city_list_df = pd.read_csv("")
city_list = city_list_df['地区'].to_list()
road_path = ""
road_index = pyogrio.read_dataframe(road_path, columns=['市', '县'], read_geometry=False, use_arrow=True)
county_fids = road_index.groupby(['市', '县']).indices
del road_index
regions_path = ""
regions_gdf = gpd.read_file(regions_path)
res_path = ""
os.makedirs(res_path, exist_ok=True)
print("Road index loaded.")
snap_tol_m = 5.0
grid_tol_m = 5.0
attr_cols = ['road_type', 'NAME']
def _clean_lines(geoms, owner):
    parts, idx = shapely.get_parts(geoms, return_index=True)
    keep = (shapely.get_type_id(parts) == 1) & ~shapely.is_empty(parts)
    return parts[keep], owner[idx[keep]]
def fix_road_build_graph(town_road, crs):
    roads = town_road.to_crs(crs)
    attrs = roads[attr_cols].reset_index(drop=True)
    geoms, owner = _clean_lines(shapely.force_2d(roads.geometry.to_numpy()), np.arange(len(roads)))
    coords, cidx = shapely.get_coordinates(geoms, return_index=True)
    bad = np.zeros(len(geoms), bool)
    bad[cidx[~np.isfinite(coords).all(axis=1)]] = True
    keep = ~bad & shapely.is_valid(geoms) & (shapely.get_num_coordinates(geoms) >= 2)
    geoms, owner = geoms[keep], owner[keep]
    geoms, owner = _clean_lines(shapely.set_precision(geoms, grid_tol_m), owner)
    if len(geoms) == 0:
        return None
    left, right = shapely.STRtree(geoms).query(geoms, predicate="dwithin", distance=snap_tol_m)
    order = np.lexsort((right, left))
    left, right = left[order], right[order]
    coords, cidx = shapely.get_coordinates(geoms, return_index=True)
    n_coords = shapely.get_num_coordinates(geoms)
    first = np.r_[0, np.cumsum(n_coords)[:-1]]
    rep = n_coords[right]
    ref_rows = np.repeat(first[right] - np.r_[0, np.cumsum(rep)[:-1]], rep) + np.arange(rep.sum())
    refs = shapely.multipoints(coords[ref_rows], indices=np.repeat(left, rep))
    geoms = shapely.snap(geoms, refs, snap_tol_m)
    geoms, owner = _clean_lines(shapely.set_precision(geoms, grid_tol_m), owner)
    del refs, ref_rows, coords, cidx
    noded = shapely.union_all(geoms)
    segs = shapely.get_parts(noded)
    segs = segs[(shapely.get_type_id(segs) == 1) & (shapely.length(segs) > 0)]
    segs = shapely.get_parts(shapely.set_precision(segs, grid_tol_m))
    segs = segs[(shapely.get_type_id(segs) == 1) & ~shapely.is_empty(segs) & (shapely.length(segs) > 0)]
    attr_speed = attrs['road_type'].map(SPEED_KMH).fillna(DEFAULT_SPEED_KMH).values[owner]
    attr_tree = shapely.STRtree(geoms)
    si, ai = attr_tree.query(segs, predicate="intersects")
    cand = pd.DataFrame({
        "seg": si, "attr": ai, "speed": attr_speed[ai],
        "overlap": np.round(shapely.length(shapely.intersection(segs[si], geoms[ai])), 2),
    })
    cand = cand[cand["overlap"] > 0]
    cand = cand.sort_values(["seg", "overlap", "speed", "attr"], ascending=[True, False, True, True])
    best = cand.drop_duplicates("seg").set_index("seg")["attr"]
    miss = np.setdiff1d(np.arange(len(segs)), best.index.values)
    if len(miss):
        mid = shapely.line_interpolate_point(segs[miss], 0.5, normalized=True)
        mi, ma = attr_tree.query_nearest(mid, max_distance=grid_tol_m, all_matches=False)
        best = pd.concat([best, pd.Series(ma, index=miss[mi])])
    edges = pd.DataFrame({"seg_id": np.arange(len(segs))})
    for c in attr_cols:
        edges[c] = pd.Series(attrs[c].values[owner[best.values]], index=best.index)
    edges["length"] = shapely.length(segs)
    edges["speed"] = edges["road_type"].map(SPEED_KMH).fillna(DEFAULT_SPEED_KMH)
    edges["time_m"] = travel_time_min(edges["length"], edges["speed"])
    p0 = shapely.get_coordinates(shapely.get_point(segs, 0))
    p1 = shapely.get_coordinates(shapely.get_point(segs, -1))
    keys = np.round(np.vstack([p0, p1]) / grid_tol_m).astype(np.int64)
    _, first_idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    node_xy = np.vstack([p0, p1])[first_idx]
    u, v = inv[:len(segs)], inv[len(segs):]
    flip = u > v
    segs[flip] = shapely.reverse(segs[flip])
    edges["u"], edges["v"] = np.where(flip, v, u), np.where(flip, u, v)
    edges = gpd.GeoDataFrame(edges, geometry=segs, crs=crs)
    edges = edges.sort_values(["time_m", "seg_id"]).drop_duplicates(subset=["u", "v"])
    n_nodes = len(node_xy)
    adj = coo_matrix((np.ones(len(edges)), (edges["u"].values, edges["v"].values)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp > 1:
        main = np.bincount(labels).argmax()
        base = np.where(labels == main)[0]
        other = np.where(labels != main)[0]
        dist, j = cKDTree(node_xy[base]).query(node_xy[other], k=1)
        near = pd.DataFrame({"comp": labels[other], "dist": dist, "node": other, "to": base[j]})
        near = near.sort_values(["comp", "dist", "node"]).drop_duplicates("comp")
        conn = gpd.GeoDataFrame({
            "seg_id": -1, "u": near["node"].values, "v": near["to"].values,
            "length": near["dist"].values, "speed": CONNECTOR_SPEED_KMH,
            "time_m": travel_time_min(near["dist"].values, CONNECTOR_SPEED_KMH),
        }, geometry=shapely.linestrings(np.stack([node_xy[near["node"].values], node_xy[near["to"].values]], axis=1)), crs=crs)
        edges = pd.concat([edges, conn], ignore_index=True)
    edges = edges.reset_index(drop=True)
    print(f"    节点 {n_nodes}，边 {len(edges)}，原有分量 {n_comp}（已连回最大分量）")
    return edges
for city in tqdm(city_list):
    city_name = city + '市'
    print(f'Processing city: {city_name}')
    city_crs = regions_gdf[regions_gdf['市'] == city_name].estimate_utm_crs()
    city_region = regions_gdf[regions_gdf['市'] == city_name].to_crs(city_crs)
    country_list = city_region['县'].unique().tolist()
    for country in tqdm(country_list):
        print(f'  Processing country: {country}')
        graph_path = f'{res_path}/{city}_{country}_road_edges.parquet'
        if os.path.exists(graph_path):
            print(f'    已存在路网文件，跳过 {city}_{country}')
            continue
        fids = county_fids.get((city_name, country))
        if fids is None or len(fids) == 0:
            print(f'    {city}_{country} 没有道路数据，跳过')
            continue
        town_road = pyogrio.read_dataframe(road_path, fids=fids, columns=attr_cols)
        edges = fix_road_build_graph(town_road, city_crs)
        if edges is None:
            print(f'    {city}_{country} 没有有效道路几何，跳过')
            continue
        edges.to_parquet(graph_path, index=False)
        del town_road, edges
        gc.collect()
