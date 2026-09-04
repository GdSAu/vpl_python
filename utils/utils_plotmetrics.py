import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── color/marker palettes (same order as before — keep method→color stable) ──
_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#f3ef07', '#01153e', '#ff81c0', '#aaff32']
_MARKERS = ['o', 's', '^', 'D', '*']


def find_run_files(base_csv):
    """
    Given a base CSV path, return all CSVs in the same directory that belong
    to the same method series.

    Two naming conventions are supported:

    1. Suffix run index  (e.g. NBV_ClassicsNBVNet, NBV_ClassicsNBVNet2, …)
       Trailing digits stripped ONLY when the character before them is NOT '_'.

    2. Infix run index  (e.g. NBV_ClassicsAE_4, NBV_ClassicsAE2_4, …)
       Detected when basename ends with _<digits>.  All files sharing the same
       body and angle suffix are grouped regardless of the infix run number.
    """
    directory = os.path.dirname(os.path.abspath(base_csv))
    basename  = os.path.splitext(os.path.basename(base_csv))[0]

    # ── Case 1: infix run index  e.g. NBV_ClassicsAE_4 / NBV_ClassicsAE2_4 ──
    infix = re.match(r'^(.+?)(\d*)(_\d+)$', basename)
    if infix:
        body         = infix.group(1)   # e.g. 'NBV_ClassicsAE'
        angle_suffix = infix.group(3)   # e.g. '_4'
        escaped_body   = re.escape(body)
        escaped_suffix = re.escape(angle_suffix)
        candidates = sorted(glob.glob(
            os.path.join(directory, body + '*' + angle_suffix + '.csv')
        ))
        files = [
            f for f in candidates
            if re.match(r'^' + escaped_body + r'\d*' + escaped_suffix + r'$',
                        os.path.splitext(os.path.basename(f))[0])
        ]
        if files:
            return files

    # ── Case 2: suffix run index  e.g. NBV_ClassicsNBVNet / NBV_ClassicsNBVNet2 ──
    match = re.match(r'^(.+?)(\d+)$', basename)
    if match and not match.group(1).endswith('_'):
        prefix = match.group(1)
    else:
        prefix = basename

    candidates = sorted(glob.glob(os.path.join(directory, prefix + '*.csv')))
    escaped = re.escape(prefix)
    files = [
        f for f in candidates
        if re.match(r'^' + escaped + r'\d*$', os.path.splitext(os.path.basename(f))[0])
    ]
    return files if files else [base_csv]


# ── private helpers ───────────────────────────────────────────────────────────

def _assign_object_blocks(df):
    """Tag each row with a 0-based object index detected from iteracion_objeto resets."""
    df   = df.copy()
    iters = df['iteracion_objeto'].tolist()
    obj_idx, indices = 0, []
    for i, val in enumerate(iters):
        if i > 0 and val == 0 and iters[i - 1] > 0:
            obj_idx += 1
        indices.append(obj_idx)
    df['obj_idx'] = indices
    return df


def _get_object_names(direccion, object_folder, n_objects):
    """
    Return sorted subfolder names from direccion/object_folder.
    Falls back to ['Obj_0', 'Obj_1', ...] if the directory is unreadable or
    the count does not match.
    """
    folder_path = os.path.join(direccion, object_folder)
    try:
        names = sorted([
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ])
        if len(names) == n_objects:
            return names
    except Exception:
        pass
    return [f'Obj_{i}' for i in range(n_objects)]


# ── metrics table ─────────────────────────────────────────────────────────────

def compute_metrics_table(csv_groups, nombres=None, max_iter=10):
    """
    Compute AUC (coverage, views 0–max_iter) and Chamfer at max_iter per
    object per method, averaged across all auto-discovered runs.

    Parameters
    ----------
    csv_groups : list of str
        One representative CSV path per method/variant.
    nombres : list of str, optional
        Display names for each method.  Defaults to the CSV base-name.
    max_iter : int
        Upper iteration bound (inclusive) for both metrics.

    Returns
    -------
    pd.DataFrame  with columns:
        Method, Object, AUC_mean, AUC_std, Chamfer_mean, Chamfer_std, N_runs
    """
    if nombres is None:
        nombres = [os.path.splitext(os.path.basename(f))[0] for f in csv_groups]

    # Derive canonical object order from the first new-format CSV found.
    # New format: id_objeto contains the object name directly (many unique values).
    # Old format: id_objeto is a dict string (one unique value per CSV).
    canonical_names = None
    for base_csv in csv_groups:
        for run_file in find_run_files(base_csv):
            df_probe   = pd.read_csv(run_file)
            unique_ids = list(dict.fromkeys(df_probe['id_objeto']))
            if len(unique_ids) > 1:
                canonical_names = unique_ids
                break
        if canonical_names:
            break

    records = []

    for base_csv, nombre in zip(csv_groups, nombres):
        run_files    = find_run_files(base_csv)
        run_aucs     = {}
        run_chamfers = {}
        print(len(run_files), 'runs found for', nombre)
        for run_file in run_files:
            df         = pd.read_csv(run_file)
            unique_ids = list(dict.fromkeys(df['id_objeto']))
            new_format = len(unique_ids) > 1

            if new_format:
                obj_iter = [(name, df[df['id_objeto'] == name]) for name in unique_ids]
            else:
                df       = _assign_object_blocks(df)
                n_blocks = df['obj_idx'].nunique()
                if canonical_names and len(canonical_names) == n_blocks:
                    names_for_blocks = canonical_names
                else:
                    s          = df.iloc[0]['id_objeto']
                    d_match    = re.search(r"'direccion'\s*:\s*'([^']*)'", s)
                    f_match    = re.search(r"'objectFolder'\s*:\s*'([^']*)'", s)
                    direccion  = d_match.group(1) if d_match else ''
                    obj_folder = f_match.group(1) if f_match else ''
                    names_for_blocks = _get_object_names(direccion, obj_folder, n_blocks)
                obj_iter = [
                    (names_for_blocks[i], subdf)
                    for i, subdf in df.groupby('obj_idx')
                    if i < len(names_for_blocks)
                ]

            for obj_name, obj_df in obj_iter:
                obj_df      = obj_df[obj_df['iteracion_objeto'] <= max_iter].sort_values('iteracion_objeto')
                auc         = np.trapz(obj_df['cobertura'] / 100.0, obj_df['iteracion_objeto'] / max_iter)
                chamfer_row = obj_df[obj_df['iteracion_objeto'] == max_iter]
                if chamfer_row.empty:
                    chamfer_row = obj_df.iloc[[-1]]
                chamfer_val = float(chamfer_row['chamfer'].iloc[0])
                run_aucs.setdefault(obj_name,     []).append(auc)
                run_chamfers.setdefault(obj_name, []).append(chamfer_val)

        obj_order = canonical_names or list(run_aucs.keys())
        for obj_name in obj_order:
            aucs     = run_aucs.get(obj_name, [])
            chamfers = run_chamfers.get(obj_name, [])
            if not aucs:
                continue
            records.append({
                'Method':       nombre,
                'Object':       obj_name,
                'AUC_mean':     np.mean(aucs),
                'AUC_std':      np.std(aucs),
                'Chamfer_mean': np.mean(chamfers),
                'Chamfer_std':  np.std(chamfers),
                'N_runs':       len(aucs),
            })

    return pd.DataFrame(records)


def print_metrics_table(df_metrics):
    """Print AUC and Chamfer tables pivoted by Object × Method."""
    methods = df_metrics['Method'].unique()

    def _fmt_mean_std(df, val_col, std_col, fmt):
        result = df.pivot(index='Object', columns='Method', values=[val_col, std_col])
        lines  = []
        header = f"{'Object':<20}" + ''.join(f"{m:>22}" for m in methods)
        lines.append(header)
        lines.append('-' * len(header))
        for obj in df['Object'].unique():
            row = df[df['Object'] == obj]
            line = f"{obj:<20}"
            for m in methods:
                cell = row[row['Method'] == m]
                if cell.empty:
                    line += f"{'—':>22}"
                else:
                    mean = cell[val_col].iloc[0]
                    std  = cell[std_col].iloc[0]
                    line += f"{fmt.format(mean, std):>22}"
            lines.append(line)
        return '\n'.join(lines)

    print('\n=== AUC Coverage (mean ± std across runs, views 0–10) ===')
    print(_fmt_mean_std(df_metrics, 'AUC_mean', 'AUC_std',    '{:.6f} ± {:.6f}'))

    print('\n=== Chamfer Distance @ iter 10 (mean ± std across runs) ===')
    print(_fmt_mean_std(df_metrics, 'Chamfer_mean', 'Chamfer_std', '{:.5f} ± {:.5f}'))


# ── plotting ──────────────────────────────────────────────────────────────────

def plotAverage(csv_files, img_name, nombres=None, max_iter=None):
    """
    Plot mean coverage ± std for a list of CSV files (or pre-loaded DataFrames).

    Parameters
    ----------
    csv_files : list of str or pd.DataFrame
    img_name  : str   — plot title and output filename (saved as .eps)
    nombres   : list of str, optional
    max_iter  : int, optional — clip x-axis to this iteration
    """
    dfs = [pd.read_csv(f) if isinstance(f, str) else f for f in csv_files]

    if nombres is None:
        nombres = [
            os.path.splitext(os.path.basename(f))[0] if isinstance(f, str) else f'Series_{i}'
            for i, f in enumerate(csv_files)
        ]

    if max_iter is not None:
        dfs = [df[df['iteracion_objeto'] <= max_iter] for df in dfs]

    dfs_agrupados = [
        df.groupby('iteracion_objeto')['cobertura'].agg(['mean', 'std']).reset_index()
        for df in dfs
    ]

    plt.figure(figsize=(10, 6))

    for i, (df_ag, nombre) in enumerate(zip(dfs_agrupados, nombres)):
        x      = df_ag['iteracion_objeto']
        y      = df_ag['mean']
        yerr   = df_ag['std']
        color  = _COLORS[i  % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]

        plt.errorbar(x, y, yerr=yerr,
                     fmt=marker, color=color, capsize=5,
                     label=nombre, linewidth=2, linestyle='-',
                     markersize=8, alpha=0.7)

    x_max = max(df['iteracion_objeto'].max() for df in dfs_agrupados)
    plt.xlim(-0.2, x_max + 0.2)
    plt.xticks(np.arange(0, x_max + 1))
    plt.ylim(0, 100)
    plt.ylabel('Coverage', fontsize=18)
    plt.xlabel('# of Views', fontsize=18)
    plt.title(img_name, fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('{}.eps'.format(img_name), format='eps')
    plt.show()


def plotAverageGrouped(base_csvs, img_name, nombres=None, max_iter=None):
    """
    Like plotAverage but auto-discovers all run files for each base CSV and
    combines them before plotting (one curve per method).

    Parameters
    ----------
    base_csvs : list of str — one representative CSV per method/variant
    img_name  : str
    nombres   : list of str, optional
    max_iter  : int, optional
    """
    if nombres is None:
        nombres = [os.path.splitext(os.path.basename(f))[0] for f in base_csvs]

    combined_dfs = []
    for base_csv in base_csvs:
        run_files = find_run_files(base_csv)
        combined  = pd.concat([pd.read_csv(f) for f in run_files], ignore_index=True)
        combined_dfs.append(combined)

    plotAverage(combined_dfs, img_name, nombres, max_iter)
