#!/usr/bin/env python3
"""
Feature importance analysis for rescaled vs original processors.

Compares feature sets and their relative importance for Chl-a correction.

Author: Analysis Team
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

RESULTS = "data/resultados.csv"
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 10

def load_data():
    """Load and prepare data"""
    df = pd.read_csv(RESULTS, encoding='latin1', decimal=',')
    df.columns = df.columns.str.strip()

    numeric_cols = ['C2RCC', 'C2X', 'C2XC', 'Medicion']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    mask = (df['C2RCC'].notna() & df['C2X'].notna() &
            df['C2XC'].notna() & df['Medicion'].notna())

    df_clean = df[mask].copy()

    # Add trophic state
    reservoir_means = df_clean.groupby('Embalse')['Medicion'].mean()
    trophic_map = {}
    for reservoir, mean_chl in reservoir_means.items():
        if mean_chl <= 25:
            trophic_map[reservoir] = 'Mesotrophic'
        elif mean_chl <= 75:
            trophic_map[reservoir] = 'Eutrophic'
        else:
            trophic_map[reservoir] = 'Hypertrophic'

    df_clean['Trophic_State'] = df_clean['Embalse'].map(trophic_map)

    return df_clean

def create_original_features(df):
    """Create original 16-feature set (WITHOUT rescaling)"""
    features = pd.DataFrame()
    feature_names = []

    # C_orig (3)
    features['C2RCC'] = df['C2RCC']
    features['C2X'] = df['C2X']
    features['C2XC'] = df['C2XC']
    feature_names.extend(['C2RCC', 'C2X', 'C2XC'])

    # T_log (3)
    features['log_C2X'] = np.log1p(df['C2X'])
    features['log_C2XC'] = np.log1p(df['C2XC'])
    features['sqrt_C2RCC'] = np.sqrt(np.abs(df['C2RCC']))
    feature_names.extend(['log_C2X', 'log_C2XC', 'sqrt_C2RCC'])

    # R_ratio (3)
    features['C2X_C2RCC_ratio'] = df['C2X'] / (df['C2RCC'] + 1e-5)
    features['C2XC_C2RCC_ratio'] = df['C2XC'] / (df['C2RCC'] + 1e-5)
    features['C2XC_C2X_ratio'] = df['C2XC'] / (df['C2X'] + 1e-5)
    feature_names.extend(['C2X/C2RCC', 'C2XC/C2RCC', 'C2XC/C2X'])

    # I_inter (4)
    features['C2X_x_C2RCC'] = df['C2X'] * df['C2RCC'] / 1000
    features['C2XC_x_C2RCC'] = df['C2XC'] * df['C2RCC'] / 1000
    features['C2RCC_squared'] = df['C2RCC'] ** 2
    features['C2RCC_x_log_C2X'] = df['C2RCC'] * features['log_C2X']
    feature_names.extend(['C2X×C2RCC', 'C2XC×C2RCC', 'C2RCC²', 'C2RCC×log(C2X)'])

    # S_stat (3)
    features['mean_corrections'] = df[['C2RCC', 'C2X', 'C2XC']].mean(axis=1)
    features['std_corrections'] = df[['C2RCC', 'C2X', 'C2XC']].std(axis=1)
    features['median_corrections'] = df[['C2RCC', 'C2X', 'C2XC']].median(axis=1)
    feature_names.extend(['Mean', 'Std', 'Median'])

    return features, feature_names

def create_rescaled_features_simple(df, train_indices):
    """Create simple rescaled feature set (8 features from strategy 2)"""
    # Calculate rescaling parameters using ONLY training data
    c2x_scale = df.loc[train_indices, 'C2RCC'].std() / df.loc[train_indices, 'C2X'].std()
    c2x_offset = df.loc[train_indices, 'C2RCC'].mean() - (df.loc[train_indices, 'C2X'].mean() * c2x_scale)

    c2xc_scale = df.loc[train_indices, 'C2RCC'].std() / df.loc[train_indices, 'C2XC'].std()
    c2xc_offset = df.loc[train_indices, 'C2RCC'].mean() - (df.loc[train_indices, 'C2XC'].mean() * c2xc_scale)

    # Apply rescaling to all data
    features = pd.DataFrame()
    feature_names = []

    features['C2RCC'] = df['C2RCC']
    features['C2X_r'] = df['C2X'] * c2x_scale + c2x_offset
    features['C2XC_r'] = df['C2XC'] * c2xc_scale + c2xc_offset
    feature_names.extend(['C2RCC', 'C2X_rescaled', 'C2XC_rescaled'])

    features['mean_all'] = features[['C2RCC', 'C2X_r', 'C2XC_r']].mean(axis=1)
    features['std_all'] = features[['C2RCC', 'C2X_r', 'C2XC_r']].std(axis=1)
    feature_names.extend(['Mean_rescaled', 'Std_rescaled'])

    features['C2X_r_x_C2RCC'] = features['C2X_r'] * features['C2RCC'] / 100
    features['C2XC_r_x_C2RCC'] = features['C2XC_r'] * features['C2RCC'] / 100
    features['C2RCC_squared'] = features['C2RCC'] ** 2
    feature_names.extend(['C2X_r×C2RCC', 'C2XC_r×C2RCC', 'C2RCC²'])

    params = {
        'c2x_scale': c2x_scale,
        'c2x_offset': c2x_offset,
        'c2xc_scale': c2xc_scale,
        'c2xc_offset': c2xc_offset
    }

    return features, feature_names, params

def create_rescaled_features_extended(df, train_indices):
    """Create extended rescaled feature set (16 features with rescaling)"""
    # Calculate rescaling parameters
    c2x_scale = df.loc[train_indices, 'C2RCC'].std() / df.loc[train_indices, 'C2X'].std()
    c2x_offset = df.loc[train_indices, 'C2RCC'].mean() - (df.loc[train_indices, 'C2X'].mean() * c2x_scale)

    c2xc_scale = df.loc[train_indices, 'C2RCC'].std() / df.loc[train_indices, 'C2XC'].std()
    c2xc_offset = df.loc[train_indices, 'C2RCC'].mean() - (df.loc[train_indices, 'C2XC'].mean() * c2xc_scale)

    features = pd.DataFrame()
    feature_names = []

    # Rescaled processors (3)
    features['C2RCC'] = df['C2RCC']
    features['C2X_r'] = df['C2X'] * c2x_scale + c2x_offset
    features['C2XC_r'] = df['C2XC'] * c2xc_scale + c2xc_offset
    feature_names.extend(['C2RCC', 'C2X_r', 'C2XC_r'])

    # Transformations (3)
    features['log_C2X_r'] = np.log1p(features['C2X_r'])
    features['log_C2XC_r'] = np.log1p(features['C2XC_r'])
    features['sqrt_C2RCC'] = np.sqrt(np.abs(df['C2RCC']))
    feature_names.extend(['log(C2X_r)', 'log(C2XC_r)', 'sqrt(C2RCC)'])

    # Ratios (3)
    features['C2X_r_C2RCC_ratio'] = features['C2X_r'] / (features['C2RCC'] + 1e-5)
    features['C2XC_r_C2RCC_ratio'] = features['C2XC_r'] / (features['C2RCC'] + 1e-5)
    features['C2XC_r_C2X_r_ratio'] = features['C2XC_r'] / (features['C2X_r'] + 1e-5)
    feature_names.extend(['C2X_r/C2RCC', 'C2XC_r/C2RCC', 'C2XC_r/C2X_r'])

    # Interactions (4)
    features['C2X_r_x_C2RCC'] = features['C2X_r'] * features['C2RCC'] / 100
    features['C2XC_r_x_C2RCC'] = features['C2XC_r'] * features['C2RCC'] / 100
    features['C2RCC_squared'] = features['C2RCC'] ** 2
    features['C2RCC_x_log_C2X_r'] = features['C2RCC'] * features['log_C2X_r']
    feature_names.extend(['C2X_r×C2RCC', 'C2XC_r×C2RCC', 'C2RCC²', 'C2RCC×log(C2X_r)'])

    # Statistics (3)
    features['mean_all'] = features[['C2RCC', 'C2X_r', 'C2XC_r']].mean(axis=1)
    features['std_all'] = features[['C2RCC', 'C2X_r', 'C2XC_r']].std(axis=1)
    features['median_all'] = features[['C2RCC', 'C2X_r', 'C2XC_r']].median(axis=1)
    feature_names.extend(['Mean_r', 'Std_r', 'Median_r'])

    params = {
        'c2x_scale': c2x_scale,
        'c2x_offset': c2x_offset,
        'c2xc_scale': c2xc_scale,
        'c2xc_offset': c2xc_offset
    }

    return features, feature_names, params

def analyze_feature_importance(X, y, feature_names, approach_name):
    """Analyze feature importance using multiple methods"""
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE ANALYSIS: {approach_name}")
    print(f"{'='*70}")
    print(f"Number of features: {len(feature_names)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. ElasticNet coefficients
    print(f"\n1. ElasticNet Coefficient Analysis:")
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic.fit(X_train_scaled, y_train)

    # Get absolute coefficients as importance
    elastic_importance = np.abs(elastic.coef_)
    elastic_importance_normalized = elastic_importance / elastic_importance.sum()

    y_pred = elastic.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"   Model Performance: RMSE = {rmse:.2f}, R² = {r2:.3f}")
    print(f"\n   Top 10 Features by Coefficient Magnitude:")
    elastic_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': elastic.coef_,
        'Abs_Coeff': np.abs(elastic.coef_),
        'Importance': elastic_importance_normalized * 100
    }).sort_values('Abs_Coeff', ascending=False)

    for i, (_, row) in enumerate(elastic_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<20} coef={row['Coefficient']:>8.3f}  importance={row['Importance']:>6.2f}%")

    results['ElasticNet'] = {
        'importance': elastic_importance_normalized,
        'coefficients': elastic.coef_,
        'rmse': rmse,
        'r2': r2,
        'df': elastic_df
    }

    # 2. Random Forest feature importance
    print(f"\n2. Random Forest Feature Importance:")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train_scaled, y_train)

    rf_importance = rf.feature_importances_
    rf_importance_normalized = rf_importance / rf_importance.sum()

    y_pred_rf = rf.predict(X_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"   Model Performance: RMSE = {rmse_rf:.2f}, R² = {r2_rf:.3f}")
    print(f"\n   Top 10 Features by Importance:")
    rf_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importance_normalized * 100
    }).sort_values('Importance', ascending=False)

    for i, (_, row) in enumerate(rf_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<20} importance={row['Importance']:>6.2f}%")

    results['RandomForest'] = {
        'importance': rf_importance_normalized,
        'rmse': rmse_rf,
        'r2': r2_rf,
        'df': rf_df
    }

    # 3. Ridge coefficients (for comparison)
    print(f"\n3. Ridge Coefficient Analysis:")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)

    ridge_importance = np.abs(ridge.coef_)
    ridge_importance_normalized = ridge_importance / ridge_importance.sum()

    y_pred_ridge = ridge.predict(X_test_scaled)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)

    print(f"   Model Performance: RMSE = {rmse_ridge:.2f}, R² = {r2_ridge:.3f}")

    results['Ridge'] = {
        'importance': ridge_importance_normalized,
        'coefficients': ridge.coef_,
        'rmse': rmse_ridge,
        'r2': r2_ridge
    }

    # Summary statistics
    print(f"\n4. Feature Set Statistics:")
    print(f"   Total features: {len(feature_names)}")
    print(f"   Top 5 features account for (ElasticNet): {elastic_df.head(5)['Importance'].sum():.1f}%")
    print(f"   Top 5 features account for (RF): {rf_df.head(5)['Importance'].sum():.1f}%")
    print(f"   Features with >5% importance (ElasticNet): {(elastic_df['Importance'] > 5).sum()}")
    print(f"   Features with >5% importance (RF): {(rf_df['Importance'] > 5).sum()}")

    return results

def compare_approaches(df):
    """Compare all three approaches"""
    print(f"\n{'#'*70}")
    print(f"# COMPREHENSIVE FEATURE IMPORTANCE COMPARISON")
    print(f"{'#'*70}")

    y = df['Medicion'].values

    # Get training indices for consistent rescaling
    train_indices = df.index[train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=42
    )[0]]

    # Approach 1: Original 16 features (no rescaling)
    X_orig, names_orig = create_original_features(df)
    results_orig = analyze_feature_importance(
        X_orig.values, y, names_orig,
        "Approach 1: Original 16 Features (No Rescaling)"
    )

    # Approach 2: Simple rescaled (8 features)
    X_simple, names_simple, params_simple = create_rescaled_features_simple(df, train_indices)
    results_simple = analyze_feature_importance(
        X_simple.values, y, names_simple,
        "Approach 2: Simple Rescaled (8 Features)"
    )

    # Approach 3: Extended rescaled (16 features)
    X_extended, names_extended, params_extended = create_rescaled_features_extended(df, train_indices)
    results_extended = analyze_feature_importance(
        X_extended.values, y, names_extended,
        "Approach 3: Extended Rescaled (16 Features)"
    )

    # Print rescaling parameters
    print(f"\n{'='*70}")
    print(f"RESCALING PARAMETERS")
    print(f"{'='*70}")
    print(f"C2X scaling:  {params_simple['c2x_scale']:.6f}")
    print(f"C2X offset:   {params_simple['c2x_offset']:.2f} mg/m³")
    print(f"C2XC scaling: {params_simple['c2xc_scale']:.6f}")
    print(f"C2XC offset:  {params_simple['c2xc_offset']:.2f} mg/m³")

    # Compare performance
    print(f"\n{'='*70}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Approach':<40} {'Model':<15} {'RMSE':>10} {'R²':>10} {'Features':>10}")
    print(f"{'-'*87}")

    approaches = [
        ("Original 16 features (no rescaling)", results_orig, len(names_orig)),
        ("Simple rescaled (8 features)", results_simple, len(names_simple)),
        ("Extended rescaled (16 features)", results_extended, len(names_extended))
    ]

    for approach_name, results, n_features in approaches:
        for model_name in ['ElasticNet', 'Ridge', 'RandomForest']:
            rmse = results[model_name]['rmse']
            r2 = results[model_name]['r2']
            print(f"{approach_name:<40} {model_name:<15} {rmse:>10.2f} {r2:>10.3f} {n_features:>10}")

    # Create comparison visualizations
    create_comparison_plots(results_orig, results_simple, results_extended,
                           names_orig, names_simple, names_extended)

    return results_orig, results_simple, results_extended

def create_comparison_plots(results_orig, results_simple, results_extended,
                            names_orig, names_simple, names_extended):
    """Create comprehensive comparison plots"""

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: ElasticNet feature importance for all three approaches
    for i, (results, names, title) in enumerate([
        (results_orig, names_orig, "Original 16 Features\n(No Rescaling)"),
        (results_simple, names_simple, "Simple Rescaled\n(8 Features)"),
        (results_extended, names_extended, "Extended Rescaled\n(16 Features)")
    ]):
        ax = fig.add_subplot(gs[0, i])
        df = results['ElasticNet']['df'].head(10)
        colors = ['green' if x > 10 else 'orange' if x > 5 else 'gray'
                 for x in df['Importance']]
        ax.barh(range(len(df)), df['Importance'], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Feature'], fontsize=9)
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'{title}\nElasticNet: RMSE={results["ElasticNet"]["rmse"]:.2f}, R²={results["ElasticNet"]["r2"]:.3f}',
                    fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Add percentage labels
        for j, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['Importance'] + 0.5, j, f"{row['Importance']:.1f}%",
                   va='center', fontsize=8)

    # Row 2: Random Forest feature importance
    for i, (results, names, title) in enumerate([
        (results_orig, names_orig, "Original"),
        (results_simple, names_simple, "Simple Rescaled"),
        (results_extended, names_extended, "Extended Rescaled")
    ]):
        ax = fig.add_subplot(gs[1, i])
        df = results['RandomForest']['df'].head(10)
        colors = ['green' if x > 10 else 'orange' if x > 5 else 'gray'
                 for x in df['Importance']]
        ax.barh(range(len(df)), df['Importance'], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Feature'], fontsize=9)
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'{title}\nRandom Forest: RMSE={results["RandomForest"]["rmse"]:.2f}, R²={results["RandomForest"]["r2"]:.3f}',
                    fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        for j, (idx, row) in enumerate(df.iterrows()):
            ax.text(row['Importance'] + 0.5, j, f"{row['Importance']:.1f}%",
                   va='center', fontsize=8)

    # Row 3: Performance comparison and feature distribution
    # Performance comparison
    ax = fig.add_subplot(gs[2, 0])
    approaches = ['Original\n(16 feat)', 'Simple Resc.\n(8 feat)', 'Extended Resc.\n(16 feat)']
    rmse_elastic = [results_orig['ElasticNet']['rmse'],
                   results_simple['ElasticNet']['rmse'],
                   results_extended['ElasticNet']['rmse']]
    r2_elastic = [results_orig['ElasticNet']['r2'],
                 results_simple['ElasticNet']['r2'],
                 results_extended['ElasticNet']['r2']]

    x_pos = np.arange(len(approaches))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x_pos - width/2, rmse_elastic, width, label='RMSE', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, r2_elastic, width, label='R²', color='coral', alpha=0.8)

    ax.set_xlabel('Approach')
    ax.set_ylabel('RMSE (mg/m³)', color='steelblue')
    ax2.set_ylabel('R² Score', color='coral')
    ax.set_title('ElasticNet Performance Comparison', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, rmse_elastic)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for i, (bar, val) in enumerate(zip(bars2, r2_elastic)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Feature concentration (how many features account for 80% importance)
    ax = fig.add_subplot(gs[2, 1])

    def count_features_for_threshold(importance_array, threshold=0.8):
        sorted_imp = np.sort(importance_array)[::-1]
        cumsum = np.cumsum(sorted_imp)
        return np.argmax(cumsum >= threshold) + 1

    n_feat_80_orig_elastic = count_features_for_threshold(results_orig['ElasticNet']['importance'])
    n_feat_80_simple_elastic = count_features_for_threshold(results_simple['ElasticNet']['importance'])
    n_feat_80_extended_elastic = count_features_for_threshold(results_extended['ElasticNet']['importance'])

    n_feat_80_orig_rf = count_features_for_threshold(results_orig['RandomForest']['importance'])
    n_feat_80_simple_rf = count_features_for_threshold(results_simple['RandomForest']['importance'])
    n_feat_80_extended_rf = count_features_for_threshold(results_extended['RandomForest']['importance'])

    x_pos = np.arange(3)
    elastic_counts = [n_feat_80_orig_elastic, n_feat_80_simple_elastic, n_feat_80_extended_elastic]
    rf_counts = [n_feat_80_orig_rf, n_feat_80_simple_rf, n_feat_80_extended_rf]

    width = 0.35
    ax.bar(x_pos - width/2, elastic_counts, width, label='ElasticNet', color='steelblue', alpha=0.8)
    ax.bar(x_pos + width/2, rf_counts, width, label='Random Forest', color='coral', alpha=0.8)

    ax.set_ylabel('Number of Features')
    ax.set_xlabel('Approach')
    ax.set_title('Features Needed for 80% Cumulative Importance', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(elastic_counts):
        ax.text(i - width/2, v + 0.1, str(v), ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(rf_counts):
        ax.text(i + width/2, v + 0.1, str(v), ha='center', va='bottom', fontsize=9)

    # Cumulative importance curves
    ax = fig.add_subplot(gs[2, 2])

    for results, label, color in [
        (results_orig, 'Original (16 feat)', 'blue'),
        (results_simple, 'Simple Resc. (8 feat)', 'green'),
        (results_extended, 'Extended Resc. (16 feat)', 'red')
    ]:
        # ElasticNet cumulative importance
        sorted_imp = np.sort(results['ElasticNet']['importance'])[::-1]
        cumsum = np.cumsum(sorted_imp) * 100
        ax.plot(range(1, len(cumsum)+1), cumsum, marker='o', label=label,
               color=color, linewidth=2, markersize=4)

    ax.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='80% threshold')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cumulative Importance (%)')
    ax.set_title('Cumulative Feature Importance (ElasticNet)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(names_orig), len(names_extended)) + 1)

    plt.savefig('figures/feature_importance_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: figures/feature_importance_comparison.pdf")

def main():
    """Main execution"""
    print(f"\n{'#'*70}")
    print(f"# FEATURE IMPORTANCE ANALYSIS")
    print(f"{'#'*70}")

    # Load data
    df = load_data()
    print(f"\nDataset: {len(df)} samples from {df['Embalse'].nunique()} reservoirs")

    # Run comprehensive comparison
    results_orig, results_simple, results_extended = compare_approaches(df)

    # Generate recommendations
    print(f"\n{'#'*70}")
    print(f"# RECOMMENDATIONS")
    print(f"{'#'*70}")

    print(f"\n1. PERFORMANCE:")
    print(f"   Simple Rescaled (8 features) achieves:")
    print(f"   - RMSE = {results_simple['ElasticNet']['rmse']:.2f} mg/m³")
    print(f"   - R² = {results_simple['ElasticNet']['r2']:.3f}")
    print(f"   - Uses only 8 features vs 16")
    print(f"   → RECOMMENDED for simplicity and performance")

    print(f"\n2. FEATURE EFFICIENCY:")
    elastic_simple = results_simple['ElasticNet']['df']
    top5_simple = elastic_simple.head(5)['Importance'].sum()
    print(f"   Top 5 features account for {top5_simple:.1f}% of importance")
    print(f"   Top 5 features in Simple Rescaled approach:")
    for i, (_, row) in enumerate(elastic_simple.head(5).iterrows(), 1):
        print(f"   {i}. {row['Feature']:<25} {row['Importance']:>6.2f}%")

    print(f"\n3. COMPARISON WITH ORIGINAL:")
    improvement_rmse = ((results_orig['ElasticNet']['rmse'] - results_simple['ElasticNet']['rmse'])
                       / results_orig['ElasticNet']['rmse'] * 100)
    improvement_r2 = ((results_simple['ElasticNet']['r2'] - results_orig['ElasticNet']['r2'])
                     / results_orig['ElasticNet']['r2'] * 100)
    print(f"   RMSE improvement: {improvement_rmse:.1f}% reduction")
    print(f"   R² improvement: {improvement_r2:.1f}% increase")
    print(f"   Feature reduction: 50% fewer features (16 → 8)")

if __name__ == '__main__':
    main()
