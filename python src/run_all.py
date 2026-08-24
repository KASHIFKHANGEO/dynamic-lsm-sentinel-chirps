import warnings
import time
import json
import os
import glob
from collections import Counter
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from rasterio.warp import reproject, Resampling
import rasterio
from scipy.interpolate import NearestNDInterpolator

warnings.filterwarnings("ignore")

DATA_DIR = "."
TERRAIN_DIR = "./terrain"
TEMPORAL_DIR = "./temporal_rasters"
OUTPUT_DIR = "./final_maps"

MASTER_SEED = 42
MODEL_SEED = 42
WINDOW = 6
W_PAST = 12
N_BOOTSTRAP = 2000
SPATIAL_N_FOLDS = 5

TIMESERIES_FILE = "monthly_timeseries_sample.npz"
COORDINATES_FILE = "coordinates_sample.csv"
EVENTS_FILE = "events_sample.csv"


def load_data():
    ts_data = np.load(f"{DATA_DIR}/{TIMESERIES_FILE}", allow_pickle=True)
    ts_all = ts_data["timeseries"].astype(np.float32)
    months = list(ts_data["months"])
    coords = pd.read_csv(f"{DATA_DIR}/{COORDINATES_FILE}")
    events = pd.read_csv(f"{DATA_DIR}/{EVENTS_FILE}")
    return ts_all, months, coords, events


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def nearest_point_index(coords, lat, lon):
    diffs = np.sqrt((coords["lat"].values - lat) ** 2 + (coords["lon"].values - lon) ** 2)
    return int(np.argmin(diffs))


def build_feature_vector(ts_pt, mo_idx, t_months):
    start = max(0, mo_idx - W_PAST)
    window = ts_pt[start:mo_idx, :]
    if len(window) < W_PAST:
        pad = np.zeros((W_PAST - len(window), 5), dtype=np.float32)
        window = np.vstack([pad, window])

    pre = ts_pt[max(0, mo_idx - WINDOW):mo_idx, :]
    post = ts_pt[mo_idx + 1: min(t_months, mo_idx + WINDOW + 1), :]
    pre = pre if len(pre) > 0 else np.zeros((1, 5), dtype=np.float32)
    post = post if len(post) > 0 else np.zeros((1, 5), dtype=np.float32)
    change = post.mean(0) - pre.mean(0)

    return np.concatenate([
        window.flatten(),
        change,
        window.mean(0), window.std(0),
        window.min(0), window.max(0),
        window[-1] - window[0],
    ])


def build_samples(ts_all, months, coords, events):
    t_months = ts_all.shape[1]
    ls_pts = coords[coords["class"] == 1].reset_index(drop=True)
    buffer_pts = coords[(coords["class"] == 1) & (coords["source"] == "Buffer_sampled")].reset_index(drop=True)

    ls_feat, ls_lats, ls_lons, ls_dates = [], [], [], []

    for _, ev in events.iterrows():
        ev_mo = f"{int(ev['year'])}_{int(ev['month']):02d}"
        if ev_mo not in months:
            continue
        mo_idx = months.index(ev_mo)
        if mo_idx < W_PAST or mo_idx + WINDOW >= t_months:
            continue

        dpts = [
            (haversine(ev["lat"], ev["lon"], float(pt["lat"]), float(pt["lon"])), pt)
            for _, pt in ls_pts.iterrows()
            if haversine(ev["lat"], ev["lon"], float(pt["lat"]), float(pt["lon"])) <= 10.0
        ]
        dpts.sort(key=lambda x: x[0])
        for _, pt in dpts[:15]:
            idx = nearest_point_index(coords, float(pt["lat"]), float(pt["lon"]))
            if idx >= len(ts_all):
                continue
            ls_feat.append(build_feature_vector(ts_all[idx], mo_idx, t_months))
            ls_lats.append(float(pt["lat"]))
            ls_lons.append(float(pt["lon"]))
            ls_dates.append(ev_mo)

    ndvi_ts = ts_all[:206, :, 0]
    for i in range(206):
        s = pd.Series(ndvi_ts[i])
        s[s < 0.1] = np.nan
        s = s.interpolate()
        sm = s.rolling(5, center=True, min_periods=2).mean()
        dr = sm.diff()
        best = None
        for idx in range(W_PAST, t_months - WINDOW):
            dv = dr.iloc[idx]
            if dv > -0.08:
                continue
            mo = int(months[idx][5:])
            if mo in [12, 1, 2, 3]:
                continue
            post_m = sm.iloc[idx + 1: idx + 7].mean()
            pre_m = sm.iloc[max(0, idx - 6): idx].mean()
            rec = post_m / (pre_m + 1e-6)
            if rec > 0.80:
                continue
            recur = sum(1 for yr in [-12, 12, 24]
                        if 0 <= idx + yr < t_months and dr.iloc[idx + yr] < -0.06)
            if recur >= 2:
                continue
            score = abs(dv) * (1 - rec)
            if best is None or score > best[1]:
                best = (idx, score)

        if best:
            mo_idx = best[0]
            olat, olon = float(coords.iloc[i]["lat"]), float(coords.iloc[i]["lon"])
            ls_feat.append(build_feature_vector(ts_all[i], mo_idx, t_months))
            ls_lats.append(olat)
            ls_lons.append(olon)
            ls_dates.append(months[mo_idx])

            nearby = [
                (haversine(olat, olon, float(pt["lat"]), float(pt["lon"])), pt)
                for _, pt in buffer_pts.iterrows()
                if haversine(olat, olon, float(pt["lat"]), float(pt["lon"])) <= 5.0
            ]
            nearby.sort(key=lambda x: x[0])
            for _, pt in nearby[:4]:
                idx = nearest_point_index(coords, float(pt["lat"]), float(pt["lon"]))
                if idx >= len(ts_all):
                    continue
                ls_feat.append(build_feature_vector(ts_all[idx], mo_idx, t_months))
                ls_lats.append(float(pt["lat"]))
                ls_lons.append(float(pt["lon"]))
                ls_dates.append(months[mo_idx])

    n_ls = len(ls_feat)

    np.random.seed(MASTER_SEED)
    ls_arr = coords[coords["class"] == 1][["lat", "lon"]].values
    nls_all = coords[coords["class"] == 0].reset_index(drop=True)
    nls_good = pd.DataFrame([
        pt for _, pt in nls_all.iterrows()
        if min(haversine(float(pt["lat"]), float(pt["lon"]), r[0], r[1]) for r in ls_arr) > 2.0
    ]).reset_index(drop=True)

    yr_dist = Counter([d[:4] for d in ls_dates])
    total_ls = sum(yr_dist.values())
    nls_mo_pool = []
    for yr, cnt in sorted(yr_dist.items()):
        n_yr = max(1, int(n_ls * cnt / total_ls))
        yr_mos = [i for i, m in enumerate(months)
                  if m[:4] == yr and int(m[5:]) not in {5, 6, 7, 8, 9, 10}
                  and i >= W_PAST and i + WINDOW < t_months]
        if not yr_mos:
            yr_mos = [i for i, m in enumerate(months)
                      if m[:4] == yr and i >= W_PAST and i + WINDOW < t_months]
        if yr_mos:
            nls_mo_pool.extend(np.random.choice(yr_mos, size=n_yr, replace=True).tolist())
    while len(nls_mo_pool) < n_ls:
        yr_mos = [i for i, m in enumerate(months)
                  if int(m[5:]) not in {5, 6, 7, 8, 9, 10} and i >= W_PAST and i + WINDOW < t_months]
        nls_mo_pool.append(np.random.choice(yr_mos))
    nls_mo_pool = nls_mo_pool[:n_ls]
    np.random.shuffle(nls_mo_pool)

    nls_feat, nls_lats, nls_lons = [], [], []
    pool = list(range(len(nls_good)))
    np.random.shuffle(pool)
    for i in range(n_ls):
        pt = nls_good.iloc[pool[i % len(pool)]]
        idx = nearest_point_index(coords, float(pt["lat"]), float(pt["lon"]))
        if idx >= len(ts_all):
            continue
        mo_idx = nls_mo_pool[i]
        nls_feat.append(build_feature_vector(ts_all[idx], mo_idx, t_months))
        nls_lats.append(float(pt["lat"]))
        nls_lons.append(float(pt["lon"]))

    X = np.array(ls_feat + nls_feat, dtype=np.float32)
    y = np.array([1] * n_ls + [0] * len(nls_feat), dtype=np.int64)
    lons = np.array(ls_lons + nls_lons)
    dates = ls_dates + ["non_event"] * len(nls_feat)
    return X, y, lons, dates


def get_base_models():
    models = {
        "CatBoost": lambda: CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, random_seed=MODEL_SEED,
            task_type="CPU", verbose=0),
        "RF": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, max_features="sqrt",
            random_state=MODEL_SEED, n_jobs=-1),
        "GBT": lambda: GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=MODEL_SEED),
        "SVM": lambda: SVC(
            kernel="rbf", C=5.0, gamma="scale",
            probability=True, random_state=MODEL_SEED),
        "LR": lambda: LogisticRegression(
            C=1.0, max_iter=500, random_state=MODEL_SEED),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", alpha=0.001, early_stopping=True,
            random_state=MODEL_SEED, max_iter=500),
    }
    return models


def prepare_splits(X, y, lons, dates):
    if np.isnan(X).sum() > 0:
        X = SimpleImputer(strategy="mean").fit_transform(X).astype(np.float32)
    X_sc = StandardScaler().fit_transform(X).astype(np.float32)

    bins = np.quantile(lons, [0.2, 0.4, 0.6, 0.8])
    spatial_block = np.digitize(lons, bins)

    temp_train = np.array([d == "non_event" or int(d[:4]) <= 2022 for d in dates])
    temp_test = np.array([d != "non_event" and int(d[:4]) >= 2023 for d in dates])

    nls_idx = np.where(y == 0)[0]
    n_test = temp_test.sum()
    if n_test > 0:
        np.random.seed(MASTER_SEED)
        nls_test = np.random.choice(nls_idx, size=min(n_test, len(nls_idx)), replace=False)
        temp_test[nls_test] = True
        temp_train[nls_test] = False

    return X_sc, spatial_block, temp_train, temp_test


def evaluate_model(name, model_fn, X_sc, y, spatial_block, temp_train, temp_test, verbose=True):
    t0 = time.time()

    spatial_probs = np.zeros(len(y))
    for b in range(SPATIAL_N_FOLDS):
        test_idx = np.where(spatial_block == b)[0]
        train_idx = np.where(spatial_block != b)[0]
        m = model_fn()
        m.fit(X_sc[train_idx], y[train_idx])
        spatial_probs[test_idx] = m.predict_proba(X_sc[test_idx])[:, 1]
    auc_spatial = roc_auc_score(y, spatial_probs)
    ap_spatial = average_precision_score(y, spatial_probs)

    m = model_fn()
    m.fit(X_sc[temp_train], y[temp_train])
    temp_probs = m.predict_proba(X_sc[temp_test])[:, 1]
    auc_temporal = roc_auc_score(y[temp_test], temp_probs)

    y_temp_true = y[temp_test]
    y_temp_pred = (temp_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_temp_true, y_temp_pred).ravel()
    acc = accuracy_score(y_temp_true, y_temp_pred)
    f1 = f1_score(y_temp_true, y_temp_pred)
    mcc = matthews_corrcoef(y_temp_true, y_temp_pred)

    if verbose:
        print(f"  {name:10s} Spatial AUC={auc_spatial:.4f}  Temporal AUC={auc_temporal:.4f}  "
              f"Gap={auc_spatial - auc_temporal:+.4f}  AP={ap_spatial:.4f}  "
              f"[{(time.time() - t0)/60:.1f} min]")

    return {
        "model": name,
        "spatial_auc": auc_spatial,
        "spatial_ap": ap_spatial,
        "temporal_auc": auc_temporal,
        "gap": auc_spatial - auc_temporal,
        "temporal_acc": acc,
        "temporal_f1": f1,
        "temporal_mcc": mcc,
        "temporal_confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "spatial_probs": spatial_probs.tolist(),
        "temporal_probs": temp_probs.tolist(),
        "temporal_y_true": y_temp_true.tolist(),
    }


def evaluate_ensemble(base_results, y, temp_train, temp_test):
    aucs = [res["spatial_auc"] for res in base_results.values()]
    weights = np.array(aucs) / np.sum(aucs)

    spatial_probs_list = [res["spatial_probs"] for res in base_results.values()]
    spatial_probs = np.zeros_like(spatial_probs_list[0])
    for p, w in zip(spatial_probs_list, weights):
        spatial_probs += w * p

    temporal_probs_list = [res["temporal_probs"] for res in base_results.values()]
    temporal_probs = np.zeros_like(temporal_probs_list[0])
    for p, w in zip(temporal_probs_list, weights):
        temporal_probs += w * p

    auc_spatial = roc_auc_score(y, spatial_probs)
    ap_spatial = average_precision_score(y, spatial_probs)
    auc_temporal = roc_auc_score(y[temp_test], temporal_probs)

    y_temp_true = y[temp_test]
    y_temp_pred = (temporal_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_temp_true, y_temp_pred).ravel()
    acc = accuracy_score(y_temp_true, y_temp_pred)
    f1 = f1_score(y_temp_true, y_temp_pred)
    mcc = matthews_corrcoef(y_temp_true, y_temp_pred)

    return {
        "model": "Ensemble",
        "spatial_auc": auc_spatial,
        "spatial_ap": ap_spatial,
        "temporal_auc": auc_temporal,
        "gap": auc_spatial - auc_temporal,
        "temporal_acc": acc,
        "temporal_f1": f1,
        "temporal_mcc": mcc,
        "temporal_confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "spatial_probs": spatial_probs.tolist(),
        "temporal_probs": temporal_probs.tolist(),
        "temporal_y_true": y_temp_true.tolist(),
        "weights": weights.tolist(),
    }


def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=MASTER_SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_feature_importance(X, y, feature_names):
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X, y)
    importance = model.feature_importances_
    return {name: float(imp) for name, imp in zip(feature_names, importance)}


def evaluate_imbalance_sensitivity(X_sc, y, spatial_block, temp_train, temp_test):
    ratios = [1, 2, 5, 10]
    results = {}
    base_models = get_base_models()

    for ratio in ratios:
        ls_idx = np.where(y == 1)[0]
        nls_idx = np.where(y == 0)[0]
        n_nls = len(ls_idx) * ratio
        if n_nls < len(nls_idx):
            nls_sampled = np.random.choice(nls_idx, size=int(n_nls), replace=False)
        else:
            nls_sampled = nls_idx

        sampled_idx = np.concatenate([ls_idx, nls_sampled])
        np.random.shuffle(sampled_idx)

        X_ratio = X_sc[sampled_idx]
        y_ratio = y[sampled_idx]

        for name, fn in base_models.items():
            aucs = []
            for b in range(SPATIAL_N_FOLDS):
                test_idx = np.where(spatial_block[sampled_idx] == b)[0]
                train_idx = np.where(spatial_block[sampled_idx] != b)[0]
                m = fn()
                m.fit(X_ratio[train_idx], y_ratio[train_idx])
                probs = m.predict_proba(X_ratio[test_idx])[:, 1]
                aucs.append(roc_auc_score(y_ratio[test_idx], probs))
            if name not in results:
                results[name] = []
            results[name].append(np.mean(aucs))

    return results, ratios


def evaluate_foldwise(X_sc, y, spatial_block, model_fn, name):
    fold_aucs = []
    for b in range(SPATIAL_N_FOLDS):
        test_idx = np.where(spatial_block == b)[0]
        train_idx = np.where(spatial_block != b)[0]
        m = model_fn()
        m.fit(X_sc[train_idx], y[train_idx])
        probs = m.predict_proba(X_sc[test_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], probs))
    return fold_aucs


def load_terrain_factors():
    coords = pd.read_csv(f"{DATA_DIR}/{COORDINATES_FILE}")
    factor_names = [
        'elevation', 'slope', 'aspect', 'plan_curvature', 'profile_curvature',
        'roughness', 'relief', 'twi', 'dist_rivers', 'dist_roads',
        'ndvi_longterm', 'rainfall_annual', 'lithology'
    ]
    factors = []
    for name in factor_names:
        with rasterio.open(f"{TERRAIN_DIR}/{name}.tif") as src:
            coords_geo = [(row['lon'], row['lat']) for _, row in coords.iterrows()]
            vals = [x[0] for x in src.sample(coords_geo)]
            factors.append(vals)
    X = np.array(factors).T
    y = coords['class'].values
    return X, y, factor_names


def train_static_model():
    X, y, factor_names = load_terrain_factors()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=42, subsample=0.8
    )
    model.fit(X_scaled, y)
    return model, scaler, factor_names


def predict_static_raster(model, scaler, factor_names, output_path):
    rasters = []
    for name in factor_names:
        with rasterio.open(f"{TERRAIN_DIR}/{name}.tif") as src:
            rasters.append(src.read(1))
            profile = src.profile

    X_static = np.stack(rasters, axis=-1)
    h, w = X_static.shape[:2]
    X_flat = X_static.reshape(-1, len(factor_names))
    X_scaled = scaler.transform(X_flat)
    preds = model.predict_proba(X_scaled)[:, 1]
    preds = preds.reshape(h, w)

    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(preds.astype(np.float32), 1)

    print(f"Static raster saved: {output_path}")
    return preds, profile


def sample_to_points(raster, profile, name, output_dir, step=4):
    h, w = raster.shape
    y_idx, x_idx = np.mgrid[0:h:step, 0:w:step]
    y_idx = y_idx.ravel()
    x_idx = x_idx.ravel()
    values = raster[y_idx, x_idx]

    transform = profile['transform']
    lons, lats = [], []
    for y, x in zip(y_idx, x_idx):
        lon, lat = transform * (x, y)
        lons.append(lon)
        lats.append(lat)

    df = pd.DataFrame({
        'longitude': lons,
        'latitude': lats,
        'susceptibility': values
    })
    df = df.dropna()

    csv_path = f"{output_dir}/gis_susceptibility_points_{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Points saved: {csv_path} ({len(df)} points)")


def fuse_static_temporal(static_path, temporal_dir, output_dir):
    with rasterio.open(static_path) as src:
        static = src.read(1).astype(np.float32)
        static_profile = src.profile

    temporal_files = glob.glob(f"{temporal_dir}/ZHJ_LSM_*.tif")
    if not temporal_files:
        print("  No temporal files found!")
        return

    with rasterio.open(temporal_files[0]) as src:
        temporal_profile = src.profile
        target_shape = (src.height, src.width)

    static_resampled = np.zeros(target_shape, dtype=np.float32)

    reproject(
        source=static,
        destination=static_resampled,
        src_transform=static_profile['transform'],
        src_crs=static_profile['crs'],
        dst_transform=temporal_profile['transform'],
        dst_crs=temporal_profile['crs'],
        resampling=Resampling.bilinear
    )

    print(f"  Static resampled from {static.shape} to {static_resampled.shape}")

    for tif_file in temporal_files:
        name = os.path.basename(tif_file).replace('.tif', '').replace('ZHJ_LSM_', '')

        with rasterio.open(tif_file) as src:
            temporal = src.read(1).astype(np.float32)

            if np.any(temporal == 0):
                mask = temporal == 0
                temporal[mask] = np.nan
                y, x = np.indices(temporal.shape)
                valid = ~np.isnan(temporal)
                interp = NearestNDInterpolator(np.column_stack([x[valid], y[valid]]), temporal[valid])
                temporal = interp(x, y).reshape(temporal.shape)
                temporal = np.nan_to_num(temporal, 0)

        fused = np.sqrt(static_resampled * temporal)
        fused = np.clip(fused, 0, 1)

        out_path = f"{output_dir}/ZHJ_LSM_fused_{name}.tif"
        profile = temporal_profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(fused.astype(np.float32), 1)

        print(f"Fused raster saved: {out_path}")
        sample_to_points(fused, profile, name, output_dir)


def main():
    print("="*80)
    print("Loading data...")
    ts_all, months, coords, events = load_data()

    print("\nConstructing samples...")
    X, y, lons, dates = build_samples(ts_all, months, coords, events)
    print(f"  n_samples = {len(y)}  (landslide={int(y.sum())}, non-landslide={int((1-y).sum())})")

    X_sc, spatial_block, temp_train, temp_test = prepare_splits(X, y, lons, dates)
    print(f"  Training: {temp_train.sum()}  Validation: {len(y)-temp_train.sum()-temp_test.sum()}  Test: {temp_test.sum()}")

    print("\n" + "="*80)
    print("Training and evaluating base models...")
    print("="*80)
    base_models = get_base_models()
    results = {}
    for name, fn in base_models.items():
        results[name] = evaluate_model(name, fn, X_sc, y, spatial_block, temp_train, temp_test)

    print("\n" + "="*80)
    print("Training and evaluating Ensemble (AUC-weighted soft-voting)...")
    print("="*80)
    ensemble_results = evaluate_ensemble(results, y, temp_train, temp_test)
    results["Ensemble"] = ensemble_results

    print(f"\nEnsemble weights:")
    for name, w in zip(base_models.keys(), ensemble_results["weights"]):
        print(f"  {name:10s}: {w:.4f}")

    print("\n" + "="*80)
    print("TABLE 6: Model Performance under Dual Validation (All 7 Models)")
    print("="*80)
    print(f"{'Model':12s} {'Spatial AUC':>12s} {'Temporal AUC':>12s} {'Gap':>10s} {'AP':>10s}")
    print("-"*80)
    for name, res in sorted(results.items(), key=lambda x: x[1]["spatial_auc"], reverse=True):
        print(f"{name:12s} {res['spatial_auc']:12.4f} {res['temporal_auc']:12.4f} {res['gap']:10.4f} {res['spatial_ap']:10.4f}")

    print("\n" + "="*80)
    print("TABLE 8: Prospective Temporal Holdout Confusion Counts (All 7 Models)")
    print("="*80)
    print(f"{'Model':12s} {'TP':>6s} {'FP':>6s} {'TN':>6s} {'FN':>6s} {'Acc':>8s} {'F1':>8s} {'MCC':>8s}")
    print("-"*80)
    for name, res in sorted(results.items(), key=lambda x: x[1]["temporal_mcc"], reverse=True):
        c = res["temporal_confusion"]
        print(f"{name:12s} {c['TP']:6d} {c['FP']:6d} {c['TN']:6d} {c['FN']:6d} "
              f"{res['temporal_acc']:8.4f} {res['temporal_f1']:8.4f} {res['temporal_mcc']:8.4f}")

    print("\n" +
