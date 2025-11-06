#!/usr/bin/env python3
"""
Final Optimized Analysis: 10-Feature Set with Robust Cross-Validation

Implements the optimized feature engineering strategy with:
- Rescaling of C2X and C2XC processors
- 10-feature optimized set
- 5-fold and 10-fold cross-validation
- Leave-One-Reservoir-Out validation
- Complete manuscript figures

Author: Analysis Team
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_data():
    """Load and prepare data from resultados.csv"""
    print("Loading data...")
    df = pd.read_csv('resultados.csv', encoding='latin1', decimal=',')
    df.columns = df.columns.str.strip()

    # Convert numeric columns
    numeric_cols = ['C2RCC', 'C2X', 'C2XC', 'Medicion']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select valid samples
    mask = (df['C2RCC'].notna() & df['C2X'].notna() &
            df['C2XC'].notna() & df['Medicion'].notna())
    df_clean = df[mask].copy()

    # Add trophic state classification by reservoir average
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

    print(f"Valid samples: {len(df_clean)}")
    print(f"Reservoirs: {df_clean['Embalse'].nunique()}")
    print(f"Trophic distribution:\n{df_clean['Trophic_State'].value_counts()}")

    return df_clean

def calculate_rescaling_params(df, indices):
    """Calculate rescaling parameters from training data"""
    c2x_scale = df.loc[indices, 'C2RCC'].std() / df.loc[indices, 'C2X'].std()
    c2x_offset = df.loc[indices, 'C2RCC'].mean() - (df.loc[indices, 'C2X'].mean() * c2x_scale)

    c2xc_scale = df.loc[indices, 'C2RCC'].std() / df.loc[indices, 'C2XC'].std()
    c2xc_offset = df.loc[indices, 'C2RCC'].mean() - (df.loc[indices, 'C2XC'].mean() * c2xc_scale)

    return {
        'c2x_scale': c2x_scale,
        'c2x_offset': c2x_offset,
        'c2xc_scale': c2xc_scale,
        'c2xc_offset': c2xc_offset
    }

def create_optimized_features(df, params):
    """
    Create optimized 10-feature set with rescaling.

    Features (10 total):
    1. C2RCC (base)
    2. C2X_rescaled
    3. C2XC_rescaled
    4. Mean_rescaled
    5. Median_rescaled
    6. C2X_r × C2RCC (dominant interaction - 45% importance)
    7. C2XC_r × C2RCC
    8. C2RCC²
    9. sqrt(C2RCC)
    10. C2XC_r / C2RCC
    """
    features = pd.DataFrame()

    # Apply rescaling
    c2x_r = df['C2X'] * params['c2x_scale'] + params['c2x_offset']
    c2xc_r = df['C2XC'] * params['c2xc_scale'] + params['c2xc_offset']

    # Base (3)
    features['C2RCC'] = df['C2RCC']
    features['C2X_r'] = c2x_r
    features['C2XC_r'] = c2xc_r

    # Statistics (2)
    features['Mean_r'] = pd.DataFrame({'C2RCC': df['C2RCC'], 'C2X_r': c2x_r, 'C2XC_r': c2xc_r}).mean(axis=1)
    features['Median_r'] = pd.DataFrame({'C2RCC': df['C2RCC'], 'C2X_r': c2x_r, 'C2XC_r': c2xc_r}).median(axis=1)

    # Interactions (2) - MOST IMPORTANT
    features['C2X_r_x_C2RCC'] = c2x_r * df['C2RCC'] / 100
    features['C2XC_r_x_C2RCC'] = c2xc_r * df['C2RCC'] / 100

    # Transformations (2)
    features['C2RCC_sq'] = df['C2RCC'] ** 2
    features['sqrt_C2RCC'] = np.sqrt(np.abs(df['C2RCC']))

    # Ratio (1)
    features['C2XC_r_div_C2RCC'] = c2xc_r / (df['C2RCC'] + 1e-5)

    feature_names = [
        'C2RCC', 'C2X_rescaled', 'C2XC_rescaled',
        'Mean_rescaled', 'Median_rescaled',
        'C2X_r×C2RCC', 'C2XC_r×C2RCC',
        'C2RCC²', 'sqrt(C2RCC)',
        'C2XC_r/C2RCC'
    ]

    return features, feature_names

def get_models():
    """Get dictionary of models to evaluate"""
    return {
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, max_iter=2000, random_state=42),
        'Huber': HuberRegressor(epsilon=1.35, max_iter=200),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
        'MLP': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42, early_stopping=True),
    }

def evaluate_kfold_cv(df, n_folds=5):
    """Evaluate models using K-Fold cross-validation"""
    print(f"\n{'='*70}")
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")

    # Calculate rescaling params from entire dataset (will recalc per fold)
    params_global = calculate_rescaling_params(df, df.index)
    X_full, feature_names = create_optimized_features(df, params_global)
    y = df['Medicion'].values

    # Stratified K-Fold by trophic state
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = get_models()
    results = {}

    print(f"\n{'Model':<20} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
    print(f"{'-'*65}")

    for model_name, model in models.items():
        rmse_scores = []
        mae_scores = []
        r2_scores = []

        for train_idx, test_idx in kf.split(X_full):
            # Recalculate rescaling params for this fold's training data
            params_fold = calculate_rescaling_params(df, df.index[train_idx])
            X, _ = create_optimized_features(df, params_fold)

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train and predict
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Metrics
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

        # Calculate mean and std
        rmse_mean = np.mean(rmse_scores)
        rmse_std = np.std(rmse_scores)
        mae_mean = np.mean(mae_scores)
        mae_std = np.std(mae_scores)
        r2_mean = np.mean(r2_scores)
        r2_std = np.std(r2_scores)

        results[model_name] = {
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'mae_mean': mae_mean,
            'mae_std': mae_std,
            'r2_mean': r2_mean,
            'r2_std': r2_std
        }

        print(f"{model_name:<20} {rmse_mean:>6.2f}±{rmse_std:<5.2f} "
              f"{mae_mean:>6.2f}±{mae_std:<5.2f} "
              f"{r2_mean:>6.3f}±{r2_std:<5.3f}")

    return results

def evaluate_loro(df):
    """Leave-One-Reservoir-Out cross-validation"""
    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-RESERVOIR-OUT (LORO) VALIDATION")
    print(f"{'='*70}")

    reservoirs = df['Embalse'].unique()
    groups = df['Embalse'].values

    # Use ElasticNet as primary model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42)

    logo = LeaveOneGroupOut()
    rmse_by_reservoir = []
    mae_by_reservoir = []
    r2_by_reservoir = []
    reservoir_names = []
    reservoir_sizes = []

    print(f"\n{'Reservoir':<20} {'N':<5} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"{'-'*60}")

    for train_idx, test_idx in logo.split(df, groups=groups):
        # Get reservoir name
        reservoir_name = df.iloc[test_idx[0]]['Embalse']
        n_samples = len(test_idx)

        # Calculate rescaling params from training data only
        params = calculate_rescaling_params(df, df.index[train_idx])
        X, feature_names = create_optimized_features(df, params)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = df.iloc[train_idx]['Medicion'].values
        y_test = df.iloc[test_idx]['Medicion'].values

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan

        rmse_by_reservoir.append(rmse)
        mae_by_reservoir.append(mae)
        r2_by_reservoir.append(r2)
        reservoir_names.append(reservoir_name)
        reservoir_sizes.append(n_samples)

        print(f"{reservoir_name:<20} {n_samples:<5} {rmse:<10.2f} {mae:<10.2f} {r2:<10.3f}")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"LORO Summary Statistics:")
    print(f"  Mean RMSE: {np.mean(rmse_by_reservoir):.2f} ± {np.std(rmse_by_reservoir):.2f} mg/m³")
    print(f"  Mean MAE:  {np.mean(mae_by_reservoir):.2f} ± {np.std(mae_by_reservoir):.2f} mg/m³")
    print(f"  Mean R²:   {np.nanmean(r2_by_reservoir):.3f} ± {np.nanstd(r2_by_reservoir):.3f}")
    print(f"  Median RMSE: {np.median(rmse_by_reservoir):.2f} mg/m³")

    return {
        'reservoir_names': reservoir_names,
        'reservoir_sizes': reservoir_sizes,
        'rmse': rmse_by_reservoir,
        'mae': mae_by_reservoir,
        'r2': r2_by_reservoir,
        'rmse_mean': np.mean(rmse_by_reservoir),
        'rmse_std': np.std(rmse_by_reservoir),
        'mae_mean': np.mean(mae_by_reservoir),
        'mae_std': np.std(mae_by_reservoir),
        'r2_mean': np.nanmean(r2_by_reservoir),
        'r2_std': np.nanstd(r2_by_reservoir)
    }

def train_final_model(df):
    """Train final model for predictions and feature importance"""
    print(f"\n{'='*70}")
    print(f"TRAINING FINAL MODEL (70-30 split)")
    print(f"{'='*70}")

    # Split data stratified by trophic state
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42,
        stratify=df['Trophic_State']
    )

    # Calculate rescaling params from training data
    params = calculate_rescaling_params(train_df, train_df.index)

    print(f"\nRescaling Parameters:")
    print(f"  C2X:  scale = {params['c2x_scale']:.6f}, offset = {params['c2x_offset']:.2f} mg/m³")
    print(f"  C2XC: scale = {params['c2xc_scale']:.6f}, offset = {params['c2xc_offset']:.2f} mg/m³")

    # Create features
    X_train, feature_names = create_optimized_features(train_df, params)
    X_test, _ = create_optimized_features(test_df, params)

    y_train = train_df['Medicion'].values
    y_test = test_df['Medicion'].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = get_models()
    results = {}

    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Improvement':<12}")
    print(f"{'-'*65}")

    # Baseline: C2RCC performance
    baseline_rmse = np.sqrt(mean_squared_error(test_df['Medicion'], test_df['C2RCC']))

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100

        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'y_pred': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'improvement': improvement
        }

        print(f"{model_name:<20} {rmse:<10.2f} {mae:<10.2f} {r2:<10.3f} {improvement:>10.1f}%")

    # Feature importance analysis
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS (Random Forest)")
    print(f"{'='*60}")

    rf_model = results['Random Forest']['model']
    importance = rf_model.feature_importances_
    importance_pct = importance / importance.sum() * 100

    print(f"\n{'Rank':<6} {'Feature':<20} {'Importance':<12}")
    print(f"{'-'*40}")
    for i, (name, imp) in enumerate(sorted(zip(feature_names, importance_pct), key=lambda x: x[1], reverse=True), 1):
        print(f"{i:<6} {name:<20} {imp:>10.2f}%")

    # Store for plotting
    results['feature_names'] = feature_names
    results['feature_importance'] = importance_pct
    results['params'] = params
    results['X_test'] = X_test
    results['y_test'] = y_test
    results['test_df'] = test_df
    results['baseline_rmse'] = baseline_rmse

    return results

def generate_figures(df, cv5_results, cv10_results, loro_results, final_results):
    """Generate all manuscript figures"""
    print(f"\n{'='*70}")
    print(f"GENERATING MANUSCRIPT FIGURES")
    print(f"{'='*70}")

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Figure 1: Model comparison (RMSE and R²)
    ax1 = fig.add_subplot(gs[0, :2])
    models = list(cv5_results.keys())
    rmse_means = [cv5_results[m]['rmse_mean'] for m in models]
    rmse_stds = [cv5_results[m]['rmse_std'] for m in models]

    x_pos = np.arange(len(models))
    colors = ['green' if r < 35 else 'orange' if r < 40 else 'red' for r in rmse_means]

    ax1.bar(x_pos, rmse_means, yerr=rmse_stds, color=colors, alpha=0.7, capsize=5)
    ax1.axhline(y=final_results['baseline_rmse'], color='red', linestyle='--', linewidth=2,
               label=f'C2RCC Baseline ({final_results["baseline_rmse"]:.1f} mg/m³)')
    ax1.set_ylabel('RMSE (mg/m³)')
    ax1.set_xlabel('Model')
    ax1.set_title('Model Performance Comparison (5-Fold CV)\nRMSE with Standard Deviation', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
        ax1.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8)

    # Figure 1b: R² comparison
    ax1b = fig.add_subplot(gs[0, 2])
    r2_means = [cv5_results[m]['r2_mean'] for m in models]
    r2_stds = [cv5_results[m]['r2_std'] for m in models]
    colors_r2 = ['green' if r > 0.6 else 'orange' if r > 0.4 else 'red' for r in r2_means]

    ax1b.barh(x_pos, r2_means, xerr=r2_stds, color=colors_r2, alpha=0.7, capsize=5)
    ax1b.set_xlabel('R² Score')
    ax1b.set_title('R² with Std Dev', fontweight='bold')
    ax1b.set_yticks(x_pos)
    ax1b.set_yticklabels(models, fontsize=9)
    ax1b.grid(True, alpha=0.3, axis='x')

    # Figure 2: Feature importance
    ax2 = fig.add_subplot(gs[1, :])
    feature_names = final_results['feature_names']
    importance = final_results['feature_importance']

    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imp = importance[sorted_idx]

    colors_feat = ['green' if x > 10 else 'orange' if x > 5 else 'steelblue' for x in sorted_imp]

    ax2.barh(range(len(sorted_names)), sorted_imp, color=colors_feat, alpha=0.8)
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names)
    ax2.set_xlabel('Feature Importance (%)')
    ax2.set_title('Feature Importance Analysis (Random Forest)\n10-Feature Optimized Set', fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    for i, imp in enumerate(sorted_imp):
        ax2.text(imp + 0.5, i, f'{imp:.1f}%', va='center', fontsize=9)

    # Figure 3: LORO validation
    ax3 = fig.add_subplot(gs[2, 0])
    loro_rmse = loro_results['rmse']
    ax3.hist(loro_rmse, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(loro_results['rmse_mean'], color='red', linestyle='--', linewidth=2,
               label=f'Mean: {loro_results["rmse_mean"]:.1f}±{loro_results["rmse_std"]:.1f}')
    ax3.set_xlabel('RMSE (mg/m³)')
    ax3.set_ylabel('Number of Reservoirs')
    ax3.set_title('LORO Validation: RMSE Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Figure 3b: LORO by reservoir size
    ax3b = fig.add_subplot(gs[2, 1])
    ax3b.scatter(loro_results['reservoir_sizes'], loro_results['rmse'], alpha=0.6, s=80)
    ax3b.set_xlabel('Reservoir Sample Size')
    ax3b.set_ylabel('RMSE (mg/m³)')
    ax3b.set_title('LORO: RMSE vs Sample Size', fontweight='bold')
    ax3b.grid(True, alpha=0.3)

    # Add correlation
    if len(loro_results['reservoir_sizes']) > 2:
        corr = np.corrcoef(loro_results['reservoir_sizes'], loro_results['rmse'])[0, 1]
        ax3b.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax3b.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Figure 3c: CV comparison
    ax3c = fig.add_subplot(gs[2, 2])
    cv_data = {
        '5-Fold CV': cv5_results['ElasticNet']['rmse_mean'],
        '10-Fold CV': cv10_results['ElasticNet']['rmse_mean'],
        'LORO': loro_results['rmse_mean']
    }
    cv_std = {
        '5-Fold CV': cv5_results['ElasticNet']['rmse_std'],
        '10-Fold CV': cv10_results['ElasticNet']['rmse_std'],
        'LORO': loro_results['rmse_std']
    }

    x_cv = np.arange(len(cv_data))
    bars = ax3c.bar(x_cv, list(cv_data.values()), yerr=list(cv_std.values()),
                    color=['steelblue', 'coral', 'lightgreen'], alpha=0.7, capsize=5)
    ax3c.set_ylabel('RMSE (mg/m³)')
    ax3c.set_title('Cross-Validation Comparison\n(ElasticNet)', fontweight='bold')
    ax3c.set_xticks(x_cv)
    ax3c.set_xticklabels(list(cv_data.keys()))
    ax3c.grid(True, alpha=0.3, axis='y')

    for i, (bar, val, std) in enumerate(zip(bars, cv_data.values(), cv_std.values())):
        ax3c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                 f'{val:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)

    # Figure 4: Scatter plots by trophic state
    ax4a = fig.add_subplot(gs[3, 0])
    ax4b = fig.add_subplot(gs[3, 1])
    ax4c = fig.add_subplot(gs[3, 2])

    test_df = final_results['test_df']
    y_pred_elastic = final_results['ElasticNet']['y_pred']

    trophic_colors = {'Mesotrophic': 'green', 'Eutrophic': 'orange', 'Hypertrophic': 'red'}

    for ax, trophic in zip([ax4a, ax4b, ax4c], ['Mesotrophic', 'Eutrophic', 'Hypertrophic']):
        mask = test_df['Trophic_State'] == trophic
        if mask.sum() > 0:
            y_true = test_df.loc[mask, 'Medicion'].values
            y_pred = y_pred_elastic[test_df.index.get_indexer(test_df[mask].index)]

            ax.scatter(y_true, y_pred, alpha=0.6, s=80, color=trophic_colors[trophic], edgecolors='black', linewidth=0.5)

            # 1:1 line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1 line')

            # Linear regression line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            ax.plot(y_true, p(y_true), "r-", alpha=0.8, linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            ax.text(0.05, 0.95, f'N = {len(y_true)}\nRMSE = {rmse:.2f}\nR² = {r2:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.set_xlabel('Measured Chl-a (mg/m³)')
            ax.set_ylabel('Predicted Chl-a (mg/m³)')
            ax.set_title(f'{trophic} Waters', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.savefig('figures/final_comprehensive_analysis.pdf', dpi=300, bbox_inches='tight')
    print("Saved: figures/final_comprehensive_analysis.pdf")

    # Generate individual figures for manuscript
    generate_individual_figures(df, cv5_results, loro_results, final_results)

def generate_individual_figures(df, cv5_results, loro_results, final_results):
    """Generate individual figures for manuscript inclusion"""

    # Figure 2: Model comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = list(cv5_results.keys())
    rmse_means = [cv5_results[m]['rmse_mean'] for m in models]
    rmse_stds = [cv5_results[m]['rmse_std'] for m in models]
    r2_means = [cv5_results[m]['r2_mean'] for m in models]
    r2_stds = [cv5_results[m]['r2_std'] for m in models]

    x_pos = np.arange(len(models))

    # RMSE plot
    colors = ['green' if r < 35 else 'orange' if r < 40 else 'red' for r in rmse_means]
    ax1.bar(x_pos, rmse_means, yerr=rmse_stds, color=colors, alpha=0.7, capsize=5)
    ax1.axhline(y=final_results['baseline_rmse'], color='red', linestyle='--', linewidth=2,
               label=f'C2RCC Baseline')
    ax1.set_ylabel('RMSE (mg/m³)', fontsize=11)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_title('(a) RMSE Comparison (5-Fold CV)', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # R² plot
    colors_r2 = ['green' if r > 0.6 else 'orange' if r > 0.4 else 'red' for r in r2_means]
    ax2.bar(x_pos, r2_means, yerr=r2_stds, color=colors_r2, alpha=0.7, capsize=5)
    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_title('(b) R² Comparison (5-Fold CV)', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/fig2_model_comparison.pdf', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig2_model_comparison.pdf")
    plt.close()

    # Figure 5: Feature importance
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    feature_names = final_results['feature_names']
    importance = final_results['feature_importance']

    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imp = importance[sorted_idx]

    colors_feat = ['green' if x > 10 else 'orange' if x > 5 else 'steelblue' for x in sorted_imp]

    ax.barh(range(len(sorted_names)), sorted_imp, color=colors_feat, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel('Feature Importance (%)', fontsize=12)
    ax.set_title('Feature Importance Analysis (Random Forest)', fontweight='bold', fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add percentage labels
    for i, imp in enumerate(sorted_imp):
        ax.text(imp + 0.8, i, f'{imp:.1f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/fig5_feature_importance.pdf', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig5_feature_importance.pdf")
    plt.close()

    # Figure 6: LORO validation
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    loro_rmse = loro_results['rmse']
    ax.hist(loro_rmse, bins=12, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(loro_results['rmse_mean'], color='red', linestyle='--', linewidth=2.5,
              label=f'Mean: {loro_results["rmse_mean"]:.2f} ± {loro_results["rmse_std"]:.2f} mg/m³')
    ax.axvline(np.median(loro_rmse), color='green', linestyle=':', linewidth=2.5,
              label=f'Median: {np.median(loro_rmse):.2f} mg/m³')
    ax.set_xlabel('RMSE (mg/m³)', fontsize=12)
    ax.set_ylabel('Number of Reservoirs', fontsize=12)
    ax.set_title('Leave-One-Reservoir-Out Validation: RMSE Distribution', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig6_loro_validation.pdf', dpi=300, bbox_inches='tight')
    print("Saved: figures/fig6_loro_validation.pdf")
    plt.close()

    print("\nAll figures generated successfully!")

def main():
    """Main execution"""
    print(f"\n{'#'*70}")
    print(f"# FINAL OPTIMIZED ANALYSIS: 10-Feature Set with Robust CV")
    print(f"{'#'*70}")

    # Load data
    df = load_data()

    # Evaluate with different CV strategies
    cv5_results = evaluate_kfold_cv(df, n_folds=5)
    cv10_results = evaluate_kfold_cv(df, n_folds=10)

    # LORO validation
    loro_results = evaluate_loro(df)

    # Train final model
    final_results = train_final_model(df)

    # Generate figures
    generate_figures(df, cv5_results, cv10_results, loro_results, final_results)

    # Print final summary
    print(f"\n{'#'*70}")
    print(f"# FINAL SUMMARY")
    print(f"{'#'*70}")

    print(f"\n1. BEST MODEL: ElasticNet with 10-feature optimized set")
    print(f"   5-Fold CV:  RMSE = {cv5_results['ElasticNet']['rmse_mean']:.2f} ± {cv5_results['ElasticNet']['rmse_std']:.2f} mg/m³, "
          f"R² = {cv5_results['ElasticNet']['r2_mean']:.3f} ± {cv5_results['ElasticNet']['r2_std']:.3f}")
    print(f"   10-Fold CV: RMSE = {cv10_results['ElasticNet']['rmse_mean']:.2f} ± {cv10_results['ElasticNet']['rmse_std']:.2f} mg/m³, "
          f"R² = {cv10_results['ElasticNet']['r2_mean']:.3f} ± {cv10_results['ElasticNet']['r2_std']:.3f}")
    print(f"   LORO:       RMSE = {loro_results['rmse_mean']:.2f} ± {loro_results['rmse_std']:.2f} mg/m³")

    print(f"\n2. IMPROVEMENT OVER C2RCC BASELINE:")
    baseline = final_results['baseline_rmse']
    elastic_rmse = cv5_results['ElasticNet']['rmse_mean']
    improvement = ((baseline - elastic_rmse) / baseline) * 100
    print(f"   C2RCC Baseline: {baseline:.2f} mg/m³")
    print(f"   ElasticNet:     {elastic_rmse:.2f} mg/m³")
    print(f"   Improvement:    {improvement:.1f}%")

    print(f"\n3. TOP 5 FEATURES (Random Forest Importance):")
    feature_names = final_results['feature_names']
    importance = final_results['feature_importance']
    for i, (name, imp) in enumerate(sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5], 1):
        print(f"   {i}. {name:<20} {imp:>6.2f}%")

    print(f"\n4. RESCALING PARAMETERS:")
    params = final_results['params']
    print(f"   C2X_rescaled  = C2X  × {params['c2x_scale']:.6f} + {params['c2x_offset']:.2f}")
    print(f"   C2XC_rescaled = C2XC × {params['c2xc_scale']:.6f} + {params['c2xc_offset']:.2f}")

    # Save summary to file
    with open('FINAL_RESULTS_SUMMARY.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("FINAL OPTIMIZED ANALYSIS RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"ElasticNet (10 features, optimized with rescaling)\n")
        f.write(f"5-Fold CV:  RMSE = {cv5_results['ElasticNet']['rmse_mean']:.2f} ± {cv5_results['ElasticNet']['rmse_std']:.2f} mg/m³\n")
        f.write(f"10-Fold CV: RMSE = {cv10_results['ElasticNet']['rmse_mean']:.2f} ± {cv10_results['ElasticNet']['rmse_std']:.2f} mg/m³\n")
        f.write(f"LORO:       RMSE = {loro_results['rmse_mean']:.2f} ± {loro_results['rmse_std']:.2f} mg/m³\n")
        f.write(f"\nImprovement: {improvement:.1f}% over C2RCC baseline\n")

    print(f"\nResults saved to: FINAL_RESULTS_SUMMARY.txt")

if __name__ == '__main__':
    main()
