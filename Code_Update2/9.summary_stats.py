import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu
from tqdm import tqdm
base_path = ""
opt_path = ""
regions_path = ""
province_dir_path = ""
out_path = ""
os.makedirs(out_path, exist_ok=True)
REGION_EN = {'东部': 'East', '中部': 'Central', '西部': 'West', '东北': 'Northeast'}
THRESHOLDS = [5, 10, 15]
FAR_SNAP_M = 5000
SENSITIVITY = [
    ('Baseline', 0, 1.00, 1.00),
    ('Additive-Low', 1, 1.00, 1.00),
    ('Additive-High', 3, 1.00, 1.00),
    ('Differential-Mild', 0, 1.30, 1.10),
    ('Differential-Moderate', 0, 1.60, 1.20),
    ('Differential-Severe', 0, 1.90, 1.30),
    ('Combined-Mild', 1, 1.30, 1.10),
    ('Combined-Moderate', 2, 1.60, 1.20),
    ('Combined-Severe', 3, 1.90, 1.30),
]
KEYS = ['市', '县', '乡']
def tlu_type(name):
    name = str(name)
    if name.endswith('街道'):
        return 'urban'
    if name.endswith('镇') or name.endswith('乡'):
        return 'rural'
    return 'excluded'
def weighted_gini(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if len(x) < 2 or w.sum() <= 0:
        return np.nan
    o = np.argsort(x, kind='stable')
    x, w = x[o], w[o]
    W = w.sum()
    mean = (w * x).sum() / W
    cw = np.cumsum(w)
    return (w * x * ((cw - w) - (W - cw))).sum() / (W * W * mean)
def weighted_median(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x, kind='stable')
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, 0.5 * cw[-1])]
def tlu_sums(folder):
    cols = ['VALUE', '市', '县', '乡', 'crime_density_norm', 'total_time', 'walk_time',
            'police_walk_time', 'network_time', 'snap_dist_m']
    rows = []
    for f in tqdm(sorted(os.listdir(folder)), desc=os.path.basename(folder)):
        if not f.endswith('.parquet'):
            continue
        t = pq.read_table(os.path.join(folder, f), columns=cols).to_pandas()
        p = t['VALUE'].to_numpy(float)
        T = t['total_time'].to_numpy(float)
        C = t['crime_density_norm'].to_numpy(float)
        far = t['snap_dist_m'].to_numpy(float) > FAR_SNAP_M
        row = {'市': t['市'].iat[0], '县': t['县'].iat[0], '乡': t['乡'].iat[0], 'n_cells': len(t),
               'S_p': p.sum(), 'S_pC': (p * C).sum(), 'S_pT': (p * T).sum(), 'S_pCT': (p * C * T).sum(),
               'S_pwalk': (p * t['walk_time'].to_numpy(float)).sum(),
               'S_ppolwalk': (p * t['police_walk_time'].to_numpy(float)).sum(),
               'S_pnet': (p * t['network_time'].to_numpy(float)).sum(),
               'far_cells': int(far.sum()), 'far_pop': p[far].sum()}
        for th in THRESHOLDS:
            row[f'S_p{th}'] = p[T <= th].sum()
        rows.append(row)
    return pd.DataFrame(rows)
regions_gdf = gpd.read_file(regions_path)
province_dir = pd.read_csv(province_dir_path)
city_province = regions_gdf.drop_duplicates('市').set_index('市')['省']
province_region = province_dir.set_index('province')['direction'].map(REGION_EN)
base = tlu_sums(base_path)
opt = tlu_sums(opt_path)
opt_cols = ['S_pT', 'S_pCT', 'S_pwalk', 'S_ppolwalk', 'S_pnet'] + [f'S_p{th}' for th in THRESHOLDS]
tlu = base.merge(opt[KEYS + opt_cols], on=KEYS, how='inner', suffixes=('', '_opt'))
if len(tlu) != len(base) or len(tlu) != len(opt):
    print(f'注意：baseline {len(base)} 个乡镇，benchmark {len(opt)} 个，两者都有的 {len(tlu)} 个')
tlu['type'] = tlu['乡'].map(tlu_type)
tlu['province'] = tlu['市'].map(city_province)
tlu['region'] = tlu['province'].map(province_region)
tlu = tlu[tlu['S_p'] > 0].copy()
far = tlu.groupby(['市', '县'], as_index=False)[['n_cells', 'far_cells', 'S_p', 'far_pop']].sum()
far = far[far['far_cells'] > 0].sort_values('far_pop', ascending=False)
far['far_pop_share'] = far['far_pop'] / far['S_p']
far.to_csv(f'{out_path}/offroad_cells_by_county.csv', index=False)
excluded = tlu[tlu['type'] == 'excluded']
tlu = tlu[tlu['type'] != 'excluded'].copy()
city_sum = tlu.groupby('市')[['S_p', 'S_pC']].sum()
city_lambda = 1 / (city_sum['S_pC'] / city_sum['S_p'])
tlu['lambda_c'] = tlu['市'].map(city_lambda)
tlu['D'] = tlu['S_p'] + tlu['lambda_c'] * tlu['S_pC']
for sfx in ['', '_opt']:
    tlu[f'S_DT{sfx}'] = tlu[f'S_pT{sfx}'] + tlu['lambda_c'] * tlu[f'S_pCT{sfx}']
    tlu[f'T_pop{sfx}'] = tlu[f'S_pT{sfx}'] / tlu['S_p']
    tlu[f'T_risk{sfx}'] = tlu[f'S_DT{sfx}'] / tlu['D']
    for th in THRESHOLDS:
        tlu[f'cov{th}{sfx}'] = tlu[f'S_p{th}{sfx}'] / tlu['S_p']
tlu['C_tilde'] = tlu['S_pC'] / tlu['S_p']
tlu['walk_time'] = tlu['S_pwalk'] / tlu['S_p']
tlu['police_walk_time'] = tlu['S_ppolwalk'] / tlu['S_p']
tlu['network_time'] = tlu['S_pnet'] / tlu['S_p']
tlu['delta'] = tlu['T_pop_opt'] - tlu['T_pop']
tlu['risk_pct'] = tlu.groupby('市')['C_tilde'].rank(pct=True, method='average')
tlu['risk_group'] = np.select([tlu['risk_pct'] > 0.75, tlu['risk_pct'] <= 0.25], ['high', 'low'], 'mid')
city_median = tlu.groupby('市').apply(lambda g: weighted_median(g['T_pop'], g['S_p']))
tlu['speed'] = np.where(tlu['T_pop'] < tlu['市'].map(city_median), 'fast', 'slow')
tlu['typology'] = np.where(tlu['risk_group'] == 'mid', 'mid_risk', tlu['risk_group'] + '_' + tlu['speed'])
tlu['group4'] = np.where(tlu['risk_group'] == 'mid', '', tlu['type'].str.capitalize() + ' ' + tlu['risk_group'] + '-risk')
def city_metrics(g):
    r = {'region': g['region'].iat[0], 'province': g['province'].iat[0], 'n_tlu': len(g),
         'n_urban': int((g['type'] == 'urban').sum()), 'n_rural': int((g['type'] == 'rural').sum()),
         'pop': g['S_p'].sum(), 'lambda_c': g['lambda_c'].iat[0], 'C_bar': 1 / g['lambda_c'].iat[0]}
    for sfx in ['', '_opt']:
        r[f'T_pop{sfx}'] = g[f'S_pT{sfx}'].sum() / g['S_p'].sum()
        r[f'T_risk{sfx}'] = g[f'S_DT{sfx}'].sum() / g['D'].sum()
        for th in THRESHOLDS:
            r[f'cov{th}{sfx}'] = g[f'S_p{th}{sfx}'].sum() / g['S_p'].sum()
        r[f'gini_pop{sfx}'] = weighted_gini(g[f'T_pop{sfx}'], g['S_p'])
        r[f'gini_risk{sfx}'] = weighted_gini(g[f'T_pop{sfx}'], g['D'])
        for kind in ['urban', 'rural']:
            k = g[g['type'] == kind]
            r[f'T_{kind}{sfx}'] = k[f'S_pT{sfx}'].sum() / k['S_p'].sum() if len(k) else np.nan
            r[f'Trisk_{kind}{sfx}'] = k[f'S_DT{sfx}'].sum() / k['D'].sum() if len(k) else np.nan
        for grp in ['Urban high-risk', 'Urban low-risk', 'Rural high-risk', 'Rural low-risk']:
            k = g[g['group4'] == grp]
            r[f'T_{grp}{sfx}'] = k[f'S_pT{sfx}'].sum() / k['S_p'].sum() if len(k) else np.nan
        r[f'gap_rural_urban{sfx}'] = r[f'T_rural{sfx}'] - r[f'T_urban{sfx}']
    r['walk_time'] = g['S_pwalk'].sum() / g['S_p'].sum()
    r['police_walk_time'] = g['S_ppolwalk'].sum() / g['S_p'].sum()
    r['network_time'] = g['S_pnet'].sum() / g['S_p'].sum()
    r['delta'] = r['T_pop_opt'] - r['T_pop']
    r['delta_gini_pop'] = r['gini_pop_opt'] - r['gini_pop']
    r['delta_gini_risk'] = r['gini_risk_opt'] - r['gini_risk']
    return pd.Series(r)
city = tlu.groupby('市').apply(city_metrics).reset_index()
both = city[(city['n_urban'] > 0) & (city['n_rural'] > 0)]
urban, rural = tlu[tlu['type'] == 'urban'], tlu[tlu['type'] == 'rural']
t1 = []
for name, tau, ku, kr in SENSITIVITY:
    t1.append({'Scenario': name, 'tau': tau, 'kappa_u': ku, 'kappa_r': kr,
               'Pop. gap (min)': np.median(tau + kr * rural['T_pop']) - np.median(tau + ku * urban['T_pop']),
               'Risk gap (min)': np.median(tau + kr * rural['T_risk']) - np.median(tau + ku * urban['T_risk']),
               'Rural slower cities': f"{int(((tau + kr * both['T_rural']) > (tau + ku * both['T_urban'])).sum())}/{len(both)}"})
table1 = pd.DataFrame(t1)
def group_row(name, g):
    base_t = g['S_pT'].sum() / g['S_p'].sum()
    opt_t = g['S_pT_opt'].sum() / g['S_p'].sum()
    return {'Group': name, 'TLUs': len(g), 'Pop. (million)': g['S_p'].sum() / 1e6, 'Base': base_t,
            'Bench.': opt_t, 'Red.': base_t - opt_t, 'Red. (%)': 100 * (base_t - opt_t) / base_t}
t2 = [group_row('All TLUs', tlu), group_row('Urban TLUs', urban), group_row('Rural TLUs', rural)]
for reg, label in [('East', 'Eastern'), ('Central', 'Central'), ('West', 'Western'), ('Northeast', 'Northeast')]:
    t2.append(group_row(f'{label} rural TLUs', rural[rural['region'] == reg]))
t2.append(group_row('Rural TLUs with baseline >=10 min', rural[rural['T_pop'] >= 10]))
t2.append(group_row('Western rural TLUs with baseline >=10 min', rural[(rural['region'] == 'West') & (rural['T_pop'] >= 10)]))
table2 = pd.DataFrame(t2)
fig4c = pd.DataFrame([group_row(grp, tlu[tlu['group4'] == grp])
                      for grp in ['Urban high-risk', 'Urban low-risk', 'Rural high-risk', 'Rural low-risk']])
fig4c['Change (opt-base, min)'] = -fig4c['Red.']
mw = mannwhitneyu(rural['T_pop'], urban['T_pop'], alternative='two-sided')
gap_fit = np.polyfit(city['T_pop'], city['gap_rural_urban'], 2) if city['gap_rural_urban'].notna().sum() > 3 else [np.nan] * 3
qj = tlu[tlu['市'] == '曲靖市']
key = {
    'n_cities': int(len(city)), 'n_tlu': int(len(tlu)), 'n_urban_tlu': int(len(urban)), 'n_rural_tlu': int(len(rural)),
    'n_excluded_tlu': int(len(excluded)), 'cities_by_region': city['region'].value_counts().to_dict(),
    'pop_million': tlu['S_p'].sum() / 1e6,
    'city_T_median': city['T_pop'].median(), 'city_T_p25': city['T_pop'].quantile(0.25),
    'share_cities_T_le5': (city['T_pop'] <= 5).mean(),
    'city_T_median_opt': city['T_pop_opt'].median(), 'share_cities_T_le5_opt': (city['T_pop_opt'] <= 5).mean(),
    'tlu_T_median_urban': urban['T_pop'].median(), 'tlu_T_median_rural': rural['T_pop'].median(),
    'mannwhitney_p': mw.pvalue,
    'cities_risk_below_pop': int((city['T_risk'] < city['T_pop']).sum()),
    'cities_gini_risk_above_pop': int((city['gini_risk'] > city['gini_pop']).sum()),
    'cities_improved': int((city['delta'] < 0).sum()),
    'cities_gini_pop_decline': int((city['delta_gini_pop'] < 0).sum()),
    'cities_gini_risk_decline': int((city['delta_gini_risk'] < 0).sum()),
    'cities_faster_and_more_equal': int(((city['delta'] < 0) & (city['delta_gini_pop'] < 0)).sum()),
    'gap_quadratic_fit_coef': list(map(float, gap_fit)),
    'qujing_tlu_median_reduction': float(-qj['delta'].median()) if len(qj) else None,
    'qujing_share_tlu_reduction_gt5': float((qj['delta'] < -5).mean()) if len(qj) else None,
    'offroad_far_pop_million': far['far_pop'].sum() / 1e6, 'offroad_far_cells': int(far['far_cells'].sum()),
}
tlu.to_csv(f'{out_path}/tlu_metrics.csv', index=False)
city.to_csv(f'{out_path}/city_metrics.csv', index=False)
table1.to_csv(f'{out_path}/table1_sensitivity.csv', index=False)
table2.to_csv(f'{out_path}/table2_reassignment_gains.csv', index=False)
fig4c.to_csv(f'{out_path}/fig4c_group_gains.csv', index=False)
with open(f'{out_path}/key_numbers.json', 'w', encoding='utf-8') as fh:
    json.dump(key, fh, ensure_ascii=False, indent=2, default=float)
map_cols = ['type', 'T_pop', 'T_risk', 'T_pop_opt', 'delta', 'C_tilde', 'risk_group', 'speed', 'typology', 'cov5']
tlu_map = regions_gdf[KEYS + ['geometry']].merge(tlu[KEYS + map_cols], on=KEYS, how='left')
tlu_map['type'] = tlu_map['type'].fillna(tlu_map['乡'].map(tlu_type))
tlu_map.loc[tlu_map['type'] == 'excluded', 'typology'] = 'excluded'
tlu_map[tlu_map['市'].isin(city['市'])].to_file(f'{out_path}/maps.gpkg', layer='tlu', driver='GPKG')
city_map = regions_gdf[regions_gdf['市'].isin(city['市'])].dissolve(by='市', as_index=False)[['市', 'geometry']]
city_map.merge(city, on='市').to_file(f'{out_path}/maps.gpkg', layer='city', driver='GPKG')
print(f'输出目录: {out_path}')
print(table1.round(2).to_string(index=False))
print(table2.round(2).to_string(index=False))
print(json.dumps(key, ensure_ascii=False, indent=1, default=float))
