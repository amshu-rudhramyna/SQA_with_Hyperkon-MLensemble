import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pickle
from scipy.stats import pearsonr, iqr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

TARGETS     = ['B', 'Fe', 'Zn', 'Cu', 'Mn', 'S']
LOG_TARGETS = ['B', 'Cu', 'Zn', 'Mn', 'S']
UNITS       = {'B':'mg/kg','Fe':'g/kg','Zn':'mg/kg','Cu':'mg/kg','Mn':'mg/kg','S':'g/kg'}
DEFICIENCY_THRESHOLD = {'B':0.5,'Fe':4.5,'Zn':1.0,'Cu':0.2,'Mn':1.0,'S':0.1}

def concordance_correlation(y_true, y_pred):
    m_t, m_p  = y_true.mean(), y_pred.mean()
    v_t, v_p  = y_true.var(), y_pred.var()
    cov       = np.cov(y_true, y_pred)[0,1]
    return 2 * cov / (v_t + v_p + (m_t - m_p)**2 + 1e-12)

def rpd(y_true, y_pred):
    return y_true.std() / np.sqrt(mean_squared_error(y_true, y_pred))

def rpiq(y_true, y_pred):
    return iqr(y_true) / np.sqrt(mean_squared_error(y_true, y_pred))

def rpd_verdict(val):
    if val >= 2.0: return "quantitative"
    if val >= 1.4: return "screening"
    return "unreliable"

def heteroscedasticity_check(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return rmse / (mae + 1e-12)

def percentile_mae(y_true, y_pred):
    q1, q3 = np.percentile(y_true, 25), np.percentile(y_true, 75)
    bulk_mask = (y_true >= q1) & (y_true <= q3)
    tail_mask = y_true > q3
    mae_bulk = mean_absolute_error(y_true[bulk_mask], y_pred[bulk_mask]) if bulk_mask.sum() > 0 else np.nan
    mae_tail = mean_absolute_error(y_true[tail_mask], y_pred[tail_mask]) if tail_mask.sum() > 0 else np.nan
    return mae_bulk, mae_tail

def evaluate_target(y_true, y_pred_log, target):
    y_pred = np.expm1(y_pred_log) if target in LOG_TARGETS else y_pred_log
    y_true_orig = np.expm1(y_true) if target in LOG_TARGETS else y_true

    r2   = r2_score(y_true_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred))
    mae  = mean_absolute_error(y_true_orig, y_pred)
    bias = np.mean(y_pred - y_true_orig)
    rpd_ = rpd(y_true_orig, y_pred)
    rpiq_= rpiq(y_true_orig, y_pred)
    ccc  = concordance_correlation(y_true_orig, y_pred)
    ratio= heteroscedasticity_check(y_true_orig, y_pred)
    mb, mt = percentile_mae(y_true_orig, y_pred)
    thr  = DEFICIENCY_THRESHOLD.get(target, None)
    useful = (rmse < thr) if thr else None

    return {
        'R2': round(r2,4), 'RMSE': round(rmse,4), 'MAE': round(mae,4),
        'Bias': round(bias,4), 'RPD': round(rpd_,3), 'RPIQ': round(rpiq_,3),
        'CCC': round(ccc,4), 'RMSE/MAE': round(ratio,3),
        'MAE_bulk': round(mb,4) if not np.isnan(mb) else None,
        'MAE_tail': round(mt,4) if not np.isnan(mt) else None,
        'RPD_verdict': rpd_verdict(rpd_),
        'Agronomically_useful': useful,
    }

def cluster_breakdown(y_true_orig, y_pred, clusters):
    results = {}
    for c in np.unique(clusters):
        mask = clusters == c
        if mask.sum() < 5: continue
        results[f'cluster_{c}'] = {
            'n': int(mask.sum()),
            'R2': round(r2_score(y_true_orig[mask], y_pred[mask]),4),
            'RMSE': round(np.sqrt(mean_squared_error(y_true_orig[mask], y_pred[mask])),4)
        }
    return results

def run_full_evaluation():
    import json
    X        = np.load('cache/X_full.npy')
    clusters = np.load('cache/clusters.npy')
    with open('cache/Y_log.pkl','rb') as f: Y_log = pickle.load(f)
    with open('models/ensemble.pkl','rb') as f: ensemble = pickle.load(f)
    with open('models/correctors.pkl','rb') as f: correctors = pickle.load(f)

    from lightgbm import LGBMRegressor
    from sklearn.metrics.pairwise import cosine_similarity

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    header = f"{'Element':<8} {'R2':>6} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'RPD':>6} {'RPIQ':>6} {'CCC':>6} {'Ratio':>6}  {'RPD_verdict':<14} {'Useful'}"
    print("\n" + "="*110)
    print("  HYPERSOILNET -- FULL EVALUATION REPORT")
    print("="*110)
    print(header)
    print("-"*110)

    for t in TARGETS:
        y_log = Y_log[t]
        oof_corrected = np.zeros_like(y_log)
        rec  = ensemble[t]
        corr = correctors[t]

        for tr_idx, val_idx in skf.split(X, clusters):
            X_sel_val = rec['selector'].transform(X[val_idx])
            g_pred = np.zeros(len(val_idx))
            for name, model in rec['models'].items():
                g_pred += rec['weights'][name] * model.predict(X_sel_val)

            res_tr = corr['residuals'][tr_idx]
            sim    = cosine_similarity(X[val_idx], X[tr_idx])
            for i in range(len(val_idx)):
                nn_idx = np.argsort(sim[i])[-30:]
                lgb_m = LGBMRegressor(n_estimators=100, num_leaves=15,
                                       learning_rate=0.05, verbose=-1, n_jobs=-1)
                lgb_m.fit(X[tr_idx][nn_idx], res_tr[nn_idx])
                g_pred[i] += lgb_m.predict(X[val_idx[[i]]])[0]

            oof_corrected[val_idx] = g_pred

        m = evaluate_target(y_log, oof_corrected, t)
        results[t] = m

        useful_str = ("yes" if m['Agronomically_useful'] else "no") if m['Agronomically_useful'] is not None else "n/a"
        print(f"  {t:<8} {m['R2']:>6.4f} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['Bias']:>8.4f} "
              f"{m['RPD']:>6.3f} {m['RPIQ']:>6.3f} {m['CCC']:>6.4f} {m['RMSE/MAE']:>6.3f}  "
              f"{m['RPD_verdict']:<14} {useful_str}", flush=True)

    print("-"*110)
    print("\n  HETEROSCEDASTICITY FLAGS")
    for t, m in results.items():
        flag = ""
        if m['RMSE/MAE'] > 1.7: flag = " <- consider log-transform of target"
        if m['Bias'] > 0.1:     flag += " <- systematic over-prediction"
        if m['Bias'] < -0.1:    flag += " <- systematic under-prediction"
        if flag: print(f"  [{t}] RMSE/MAE={m['RMSE/MAE']}  Bias={m['Bias']}{flag}")

    print("\n  PER-SOIL-CLUSTER BREAKDOWN")
    for t in ['B', 'Fe']:
        y_orig = np.expm1(Y_log[t]) if t in LOG_TARGETS else Y_log[t]
        oof = np.zeros_like(y_orig)
        rec = ensemble[t]
        for tr_idx, val_idx in skf.split(X, clusters):
            X_sel = rec['selector'].transform(X[val_idx])
            g = np.zeros(len(val_idx))
            for name, model in rec['models'].items():
                g += rec['weights'][name] * model.predict(X_sel)
            oof[val_idx] = g
        y_pred = np.expm1(oof) if t in LOG_TARGETS else oof
        breakdown = cluster_breakdown(y_orig, y_pred, clusters)
        print(f"\n  [{t}]")
        for cl, stats in breakdown.items():
            print(f"    {cl}: n={stats['n']}  R2={stats['R2']}  RMSE={stats['RMSE']} {UNITS[t]}")

    def np_to_py(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: np_to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [np_to_py(x) for x in obj]
        return obj

    os.makedirs('results/', exist_ok=True)
    with open('results/metrics.json','w') as f:
        json.dump(np_to_py(results), f, indent=2)
    print("\nFull metrics saved -> results/metrics.json")
    return results

if __name__ == '__main__':
    run_full_evaluation()
