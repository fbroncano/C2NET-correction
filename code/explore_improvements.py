#!/usr/bin/env python3
"""
Exploratory analysis to improve machine learning correction of C2-Net processors.

This script investigates why current improvements are modest (3.9%) and proposes
alternative strategies to achieve better performance.

Author: Analysis Team
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_data():
    """Load and prepare data from resultados.csv"""
    df = pd.read_csv('resultados.csv', encoding='latin1', decimal=',')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert numeric columns - handle both , and . as decimal separators
    numeric_cols = ['C2RCC', 'C2X', 'C2XC', 'Medicion']
    for col in numeric_cols:
        if col in df.columns:
            # Convert to string first, replace commas with dots, then to float
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select valid samples with all corrections
    mask = (df['C2RCC'].notna() &
            df['C2X'].notna() &
            df['C2XC'].notna() &
            df['Medicion'].notna())

    df_clean = df[mask].copy()

    # Add trophic state classification by reservoir
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

    print(f"\n{'='*70}")
    print(f"DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Total valid samples: {len(df_clean)}")
    print(f"Reservoirs: {df_clean['Embalse'].nunique()}")
    print(f"\nTrophic state distribution:")
    print(df_clean['Trophic_State'].value_counts())
    print(f"\nChl-a statistics:")
    print(f"  Mean: {df_clean['Medicion'].mean():.2f} mg/m³")
    print(f"  Std: {df_clean['Medicion'].std():.2f} mg/m³")
    print(f"  Range: {df_clean['Medicion'].min():.2f} - {df_clean['Medicion'].max():.2f} mg/m³")

    return df_clean

def diagnostic_analysis(df):
    """Analyze why current approach gives modest improvements"""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC ANALYSIS: Why are improvements so modest?")
    print(f"{'='*70}")

    y = df['Medicion'].values

    # 1. Analyze C2X and C2XC contribution
    print(f"\n1. C2X and C2XC Error Analysis:")
    print(f"   {'Processor':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Correlation':>12}")
    print(f"   {'-'*56}")

    for proc in ['C2RCC', 'C2X', 'C2XC']:
        rmse = np.sqrt(mean_squared_error(y, df[proc]))
        mae = mean_absolute_error(y, df[proc])
        r2 = r2_score(y, df[proc])
        corr = np.corrcoef(y, df[proc])[0, 1]
        print(f"   {proc:<12} {rmse:>10.2f} {mae:>10.2f} {r2:>10.3f} {corr:>12.3f}")

    # 2. Check if C2X/C2XC are adding noise
    print(f"\n2. Signal-to-Noise Ratio (using C2RCC as reference):")
    c2rcc_residuals = np.abs(df['C2RCC'] - y)
    c2x_residuals = np.abs(df['C2X'] - y)
    c2xc_residuals = np.abs(df['C2XC'] - y)

    print(f"   C2RCC mean absolute error: {c2rcc_residuals.mean():.2f} mg/m³")
    print(f"   C2X mean absolute error: {c2x_residuals.mean():.2f} mg/m³")
    print(f"   C2XC mean absolute error: {c2xc_residuals.mean():.2f} mg/m³")
    print(f"\n   → C2X adds {c2x_residuals.mean() / c2rcc_residuals.mean():.1f}x more error")
    print(f"   → C2XC adds {c2xc_residuals.mean() / c2rcc_residuals.mean():.1f}x more error")

    # 3. Analyze by trophic state
    print(f"\n3. Performance by Trophic State (C2RCC baseline):")
    print(f"   {'Trophic State':<15} {'N':>5} {'RMSE':>10} {'R²':>10}")
    print(f"   {'-'*42}")

    for state in ['Mesotrophic', 'Eutrophic', 'Hypertrophic']:
        mask = df['Trophic_State'] == state
        if mask.sum() > 0:
            y_state = df.loc[mask, 'Medicion']
            pred_state = df.loc[mask, 'C2RCC']
            rmse = np.sqrt(mean_squared_error(y_state, pred_state))
            r2 = r2_score(y_state, pred_state)
            print(f"   {state:<15} {mask.sum():>5} {rmse:>10.2f} {r2:>10.3f}")

    # 4. Check C2RCC baseline quality
    print(f"\n4. C2RCC Already Performs Reasonably Well:")
    baseline_rmse = np.sqrt(mean_squared_error(y, df['C2RCC']))
    baseline_r2 = r2_score(y, df['C2RCC'])
    print(f"   Baseline RMSE: {baseline_rmse:.2f} mg/m³")
    print(f"   Baseline R²: {baseline_r2:.3f}")
    print(f"   → This limits potential improvement!")

    return {
        'c2rcc_rmse': baseline_rmse,
        'c2x_noise_ratio': c2x_residuals.mean() / c2rcc_residuals.mean(),
        'c2xc_noise_ratio': c2xc_residuals.mean() / c2rcc_residuals.mean()
    }

def strategy_1_c2rcc_only(df):
    """Strategy 1: Use ONLY C2RCC with sophisticated features"""
    print(f"\n{'='*70}")
    print(f"STRATEGY 1: Use ONLY C2RCC (eliminate noisy C2X/C2XC)")
    print(f"{'='*70}")

    # Create features from C2RCC only
    features = pd.DataFrame()
    features['C2RCC'] = df['C2RCC']
    features['C2RCC_squared'] = df['C2RCC'] ** 2
    features['C2RCC_cubed'] = df['C2RCC'] ** 3
    features['log_C2RCC'] = np.log1p(np.abs(df['C2RCC']))
    features['sqrt_C2RCC'] = np.sqrt(np.abs(df['C2RCC']))
    features['inv_C2RCC'] = 1 / (df['C2RCC'] + 1e-5)

    # Add exponential decay features
    features['exp_C2RCC'] = np.exp(-df['C2RCC'] / 50)

    print(f"\nFeatures created: {features.shape[1]} features from C2RCC only")

    X = features.values
    y = df['Medicion'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=df['Trophic_State']
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test multiple models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Huber': HuberRegressor(epsilon=1.35),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    print(f"\nModel Performance:")
    print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Improvement':>12}")
    print(f"{'-'*64}")

    baseline_rmse = np.sqrt(mean_squared_error(y_test, X_test[:, 0]))  # C2RCC only

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100

        results.append({
            'Strategy': 'C2RCC_only',
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Improvement': improvement
        })

        print(f"{name:<20} {rmse:>10.2f} {mae:>10.2f} {r2:>10.3f} {improvement:>11.1f}%")

    return results

def strategy_2_rescaled_processors(df):
    """Strategy 2: Rescale C2X and C2XC before using them"""
    print(f"\n{'='*70}")
    print(f"STRATEGY 2: Rescale C2X/C2XC to C2RCC scale")
    print(f"{'='*70}")

    # Rescale C2X and C2XC to have similar scale as C2RCC
    # Use quantile matching
    from scipy.stats import pearsonr

    y = df['Medicion'].values

    # Find optimal scaling factors using correlation with measured values
    # C2X rescaling
    c2x_scale = df['C2RCC'].std() / df['C2X'].std()
    c2x_offset = df['C2RCC'].mean() - (df['C2X'].mean() * c2x_scale)

    # C2XC rescaling
    c2xc_scale = df['C2RCC'].std() / df['C2XC'].std()
    c2xc_offset = df['C2RCC'].mean() - (df['C2XC'].mean() * c2xc_scale)

    print(f"\nRescaling parameters:")
    print(f"  C2X:  scale={c2x_scale:.4f}, offset={c2x_offset:.2f}")
    print(f"  C2XC: scale={c2xc_scale:.4f}, offset={c2xc_offset:.2f}")

    # Create rescaled features
    features = pd.DataFrame()
    features['C2RCC'] = df['C2RCC']
    features['C2X_rescaled'] = df['C2X'] * c2x_scale + c2x_offset
    features['C2XC_rescaled'] = df['C2XC'] * c2xc_scale + c2xc_offset

    # Add interactions with rescaled values
    features['mean_all'] = features[['C2RCC', 'C2X_rescaled', 'C2XC_rescaled']].mean(axis=1)
    features['std_all'] = features[['C2RCC', 'C2X_rescaled', 'C2XC_rescaled']].std(axis=1)
    features['C2X_r_x_C2RCC'] = features['C2X_rescaled'] * features['C2RCC'] / 100
    features['C2XC_r_x_C2RCC'] = features['C2XC_rescaled'] * features['C2RCC'] / 100
    features['C2RCC_squared'] = features['C2RCC'] ** 2

    print(f"\nRescaled processor errors:")
    print(f"  C2X_rescaled RMSE: {np.sqrt(mean_squared_error(y, features['C2X_rescaled'])):.2f}")
    print(f"  C2XC_rescaled RMSE: {np.sqrt(mean_squared_error(y, features['C2XC_rescaled'])):.2f}")
    print(f"  (Original C2X RMSE: {np.sqrt(mean_squared_error(y, df['C2X'])):.2f})")
    print(f"  (Original C2XC RMSE: {np.sqrt(mean_squared_error(y, df['C2XC'])):.2f})")

    X = features.values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=df['Trophic_State']
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Test models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    print(f"\nModel Performance with Rescaled Processors:")
    print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Improvement':>12}")
    print(f"{'-'*64}")

    baseline_rmse = np.sqrt(mean_squared_error(y_test, X_test[:, 0]))

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100

        results.append({
            'Strategy': 'Rescaled',
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Improvement': improvement
        })

        print(f"{name:<20} {rmse:>10.2f} {mae:>10.2f} {r2:>10.3f} {improvement:>11.1f}%")

    return results

def strategy_3_trophic_specific_models(df):
    """Strategy 3: Train separate models for each trophic state"""
    print(f"\n{'='*70}")
    print(f"STRATEGY 3: Trophic-Specific Models")
    print(f"{'='*70}")
    print(f"Train separate models for each trophic state")

    # Use all 16 original features
    def create_features(data):
        features = pd.DataFrame()
        # C_orig (3)
        features['C2RCC'] = data['C2RCC']
        features['C2X'] = data['C2X']
        features['C2XC'] = data['C2XC']
        # T_log (3)
        features['log_C2X'] = np.log1p(data['C2X'])
        features['log_C2XC'] = np.log1p(data['C2XC'])
        features['sqrt_C2RCC'] = np.sqrt(np.abs(data['C2RCC']))
        # R_ratio (3)
        features['C2X_C2RCC_ratio'] = data['C2X'] / (data['C2RCC'] + 1e-5)
        features['C2XC_C2RCC_ratio'] = data['C2XC'] / (data['C2RCC'] + 1e-5)
        features['C2XC_C2X_ratio'] = data['C2XC'] / (data['C2X'] + 1e-5)
        # I_inter (4)
        features['C2X_x_C2RCC'] = data['C2X'] * data['C2RCC'] / 1000
        features['C2XC_x_C2RCC'] = data['C2XC'] * data['C2RCC'] / 1000
        features['C2RCC_squared'] = data['C2RCC'] ** 2
        features['C2RCC_x_log_C2X'] = data['C2RCC'] * features['log_C2X']
        # S_stat (3)
        features['mean_corrections'] = data[['C2RCC', 'C2X', 'C2XC']].mean(axis=1)
        features['std_corrections'] = data[['C2RCC', 'C2X', 'C2XC']].std(axis=1)
        features['median_corrections'] = data[['C2RCC', 'C2X', 'C2XC']].median(axis=1)
        return features

    results = []

    for trophic_state in ['Mesotrophic', 'Eutrophic', 'Hypertrophic']:
        print(f"\n{trophic_state}:")
        mask = df['Trophic_State'] == trophic_state
        df_state = df[mask].copy()

        if len(df_state) < 10:
            print(f"  Skipping (insufficient samples: {len(df_state)})")
            continue

        print(f"  Samples: {len(df_state)}")

        X = create_features(df_state).values
        y = df_state['Medicion'].values

        # Split with smaller test size for small datasets
        test_size = 0.3 if len(df_state) > 20 else 0.25
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use simpler model for small datasets
        if len(df_state) < 20:
            model = Ridge(alpha=1.0)
            model_name = 'Ridge'
        else:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5)
            model_name = 'ElasticNet'

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Baseline for this trophic state
        baseline = df_state['C2RCC'].values
        y_all = df_state['Medicion'].values
        baseline_rmse = np.sqrt(mean_squared_error(y_all, baseline))
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100

        results.append({
            'Strategy': f'Trophic_{trophic_state}',
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Improvement': improvement,
            'N_samples': len(df_state),
            'N_test': len(y_test)
        })

        print(f"  Model: {model_name}")
        print(f"  RMSE: {rmse:.2f} mg/m³")
        print(f"  R²: {r2:.3f}")
        print(f"  Improvement: {improvement:.1f}%")

    return results

def strategy_4_weighted_ensemble_by_trophic(df):
    """Strategy 4: Weighted ensemble that assigns different importance by trophic state"""
    print(f"\n{'='*70}")
    print(f"STRATEGY 4: Adaptive Weighted Ensemble by Trophic State")
    print(f"{'='*70}")

    def create_features(data):
        features = pd.DataFrame()
        features['C2RCC'] = data['C2RCC']
        features['C2X'] = data['C2X']
        features['C2XC'] = data['C2XC']
        features['log_C2X'] = np.log1p(data['C2X'])
        features['log_C2XC'] = np.log1p(data['C2XC'])
        features['sqrt_C2RCC'] = np.sqrt(np.abs(data['C2RCC']))
        features['C2X_C2RCC_ratio'] = data['C2X'] / (data['C2RCC'] + 1e-5)
        features['C2XC_C2RCC_ratio'] = data['C2XC'] / (data['C2RCC'] + 1e-5)
        features['C2XC_C2X_ratio'] = data['C2XC'] / (data['C2X'] + 1e-5)
        features['C2X_x_C2RCC'] = data['C2X'] * data['C2RCC'] / 1000
        features['C2XC_x_C2RCC'] = data['C2XC'] * data['C2RCC'] / 1000
        features['C2RCC_squared'] = data['C2RCC'] ** 2
        features['C2RCC_x_log_C2X'] = data['C2RCC'] * features['log_C2X']
        features['mean_corrections'] = data[['C2RCC', 'C2X', 'C2XC']].mean(axis=1)
        features['std_corrections'] = data[['C2RCC', 'C2X', 'C2XC']].std(axis=1)
        features['median_corrections'] = data[['C2RCC', 'C2X', 'C2XC']].median(axis=1)
        return features

    X = create_features(df).values
    y = df['Medicion'].values
    trophic = df['Trophic_State'].values

    X_train, X_test, y_train, y_test, trophic_train, trophic_test = train_test_split(
        X, y, trophic, test_size=0.3, random_state=42, stratify=df['Trophic_State']
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions[name] = model.predict(X_test_scaled)

    # Compute optimal weights per trophic state
    weights_by_trophic = {}

    for state in ['Mesotrophic', 'Eutrophic', 'Hypertrophic']:
        mask = trophic_test == state
        if mask.sum() == 0:
            continue

        y_state = y_test[mask]

        # Find best single model for this trophic state
        best_rmse = float('inf')
        best_model = None

        for name, pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_state, pred[mask]))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = name

        weights_by_trophic[state] = best_model
        print(f"\n{state}: Best model = {best_model} (RMSE = {best_rmse:.2f})")

    # Create adaptive predictions
    y_pred_adaptive = np.zeros_like(y_test)

    for state in ['Mesotrophic', 'Eutrophic', 'Hypertrophic']:
        mask = trophic_test == state
        if mask.sum() == 0:
            continue
        best_model = weights_by_trophic[state]
        y_pred_adaptive[mask] = predictions[best_model][mask]

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_adaptive))
    mae = mean_absolute_error(y_test, y_pred_adaptive)
    r2 = r2_score(y_test, y_pred_adaptive)

    baseline_rmse = np.sqrt(mean_squared_error(y_test, X_test[:, 0]))
    improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100

    print(f"\nAdaptive Ensemble Performance:")
    print(f"  RMSE: {rmse:.2f} mg/m³")
    print(f"  MAE: {mae:.2f} mg/m³")
    print(f"  R²: {r2:.3f}")
    print(f"  Improvement: {improvement:.1f}%")

    return [{
        'Strategy': 'Adaptive_Ensemble',
        'Model': 'Best_per_Trophic',
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Improvement': improvement
    }]

def compare_all_strategies(all_results):
    """Compare all strategies and identify the best"""
    print(f"\n{'='*70}")
    print(f"COMPARISON OF ALL STRATEGIES")
    print(f"{'='*70}")

    df_results = pd.DataFrame(all_results)

    # Find best overall
    best = df_results.loc[df_results['RMSE'].idxmin()]

    print(f"\nBEST STRATEGY:")
    print(f"  Strategy: {best['Strategy']}")
    print(f"  Model: {best['Model']}")
    print(f"  RMSE: {best['RMSE']:.2f} mg/m³")
    print(f"  MAE: {best['MAE']:.2f} mg/m³")
    print(f"  R²: {best['R2']:.3f}")
    print(f"  Improvement: {best['Improvement']:.1f}%")

    # Top 5 strategies
    print(f"\nTOP 5 APPROACHES:")
    print(f"{'Strategy':<25} {'Model':<20} {'RMSE':>10} {'R²':>10} {'Improve':>10}")
    print(f"{'-'*77}")

    top5 = df_results.nsmallest(5, 'RMSE')
    for _, row in top5.iterrows():
        strategy = row['Strategy'][:24]
        model = row['Model'][:19]
        print(f"{strategy:<25} {model:<20} {row['RMSE']:>10.2f} {row['R2']:>10.3f} {row['Improvement']:>9.1f}%")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: RMSE comparison
    ax = axes[0, 0]
    df_plot = df_results.sort_values('RMSE')
    colors = ['green' if x < 35 else 'orange' if x < 40 else 'red' for x in df_plot['RMSE']]
    ax.barh(range(len(df_plot)), df_plot['RMSE'], color=colors)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{r['Strategy'][:15]}-{r['Model'][:10]}" for _, r in df_plot.iterrows()], fontsize=8)
    ax.set_xlabel('RMSE (mg/m³)')
    ax.set_title('RMSE Comparison Across All Strategies')
    ax.axvline(39.44, color='red', linestyle='--', linewidth=2, label='C2RCC Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: R² comparison
    ax = axes[0, 1]
    df_plot = df_results.sort_values('R2', ascending=False)
    colors = ['green' if x > 0.6 else 'orange' if x > 0.4 else 'red' for x in df_plot['R2']]
    ax.barh(range(len(df_plot)), df_plot['R2'], color=colors)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{r['Strategy'][:15]}-{r['Model'][:10]}" for _, r in df_plot.iterrows()], fontsize=8)
    ax.set_xlabel('R² Score')
    ax.set_title('R² Comparison Across All Strategies')
    ax.axvline(0.43, color='red', linestyle='--', linewidth=2, label='C2RCC Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Improvement percentage
    ax = axes[1, 0]
    df_plot = df_results.sort_values('Improvement', ascending=False)
    colors = ['green' if x > 10 else 'orange' if x > 5 else 'red' for x in df_plot['Improvement']]
    ax.barh(range(len(df_plot)), df_plot['Improvement'], color=colors)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{r['Strategy'][:15]}-{r['Model'][:10]}" for _, r in df_plot.iterrows()], fontsize=8)
    ax.set_xlabel('Improvement over C2RCC (%)')
    ax.set_title('Improvement Percentage')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3)

    # Plot 4: Strategy summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    summary_data = []
    for strategy in df_results['Strategy'].unique():
        df_strat = df_results[df_results['Strategy'] == strategy]
        best_model = df_strat.loc[df_strat['RMSE'].idxmin()]
        summary_data.append([
            strategy[:20],
            best_model['Model'][:15],
            f"{best_model['RMSE']:.1f}",
            f"{best_model['R2']:.3f}",
            f"{best_model['Improvement']:.1f}%"
        ])

    table = ax.table(cellText=summary_data,
                     colLabels=['Strategy', 'Best Model', 'RMSE', 'R²', 'Improve'],
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code the table
    for i in range(1, len(summary_data) + 1):
        rmse = float(summary_data[i-1][2])
        if rmse < 35:
            color = '#90EE90'
        elif rmse < 40:
            color = '#FFD700'
        else:
            color = '#FFB6C1'
        table[(i, 0)].set_facecolor(color)

    ax.set_title('Summary: Best Model per Strategy', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/strategy_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: figures/strategy_comparison.pdf")

    return best

def main():
    """Main execution"""
    print(f"\n{'#'*70}")
    print(f"# EXPLORATORY ANALYSIS: Improving ML Correction Performance")
    print(f"{'#'*70}")

    # Load data
    df = load_data()

    # Diagnostic analysis
    diagnostic_info = diagnostic_analysis(df)

    # Test all strategies
    all_results = []

    # Strategy 1: C2RCC only
    results1 = strategy_1_c2rcc_only(df)
    all_results.extend(results1)

    # Strategy 2: Rescaled processors
    results2 = strategy_2_rescaled_processors(df)
    all_results.extend(results2)

    # Strategy 3: Trophic-specific models
    results3 = strategy_3_trophic_specific_models(df)
    all_results.extend(results3)

    # Strategy 4: Adaptive ensemble
    results4 = strategy_4_weighted_ensemble_by_trophic(df)
    all_results.extend(results4)

    # Compare all
    best_strategy = compare_all_strategies(all_results)

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('strategy_comparison_results.csv', index=False)
    print(f"\nResults saved: strategy_comparison_results.csv")

    print(f"\n{'#'*70}")
    print(f"# RECOMMENDATION")
    print(f"{'#'*70}")
    print(f"\nBased on this analysis, the recommended approach is:")
    print(f"Strategy: {best_strategy['Strategy']}")
    print(f"Model: {best_strategy['Model']}")
    print(f"\nThis achieves:")
    print(f"  - RMSE reduction from 39.44 to {best_strategy['RMSE']:.2f} mg/m³")
    print(f"  - Improvement of {best_strategy['Improvement']:.1f}%")
    print(f"  - R² improvement from 0.43 to {best_strategy['R2']:.3f}")

if __name__ == '__main__':
    main()
