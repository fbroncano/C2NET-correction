#!/usr/bin/env python3
"""
Generate Figure 7: Performance by Trophic State

Creates a 4-panel figure showing model performance across trophic states:
(a) RMSE comparison by trophic state
(b) R² comparison by trophic state  
(c) Heatmap of RMSE values
(d) Performance summary table

Based on final_optimized_analysis.py results
Author: Analysis Team
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configuración para publicación (ajustes de legibilidad)
FONT_BASE = 18  # aumento del 30% para mejor legibilidad (14 -> 18)
plt.rcParams['font.size'] = FONT_BASE
plt.rcParams['axes.labelsize'] = FONT_BASE + 1
plt.rcParams['axes.titlesize'] = FONT_BASE + 2
plt.rcParams['xtick.labelsize'] = FONT_BASE - 2
plt.rcParams['ytick.labelsize'] = FONT_BASE - 2
plt.rcParams['legend.fontsize'] = FONT_BASE - 1
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15

# Set style
sns.set_style("ticks")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

TROPHIC_COLORS = {'Mesotrophic': '#2E7D32', 'Eutrophic': '#F57C00', 'Hypertrophic': '#C62828'}

def load_data():
    """Load and prepare data"""
    print("Loading data...")
    df = pd.read_csv('data/resultados.csv', encoding='latin1', decimal=',')
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
    """Create optimized 10-feature set with rescaling"""
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
    
    # Interactions (2) - divided by 100
    features['C2X_r_x_C2RCC'] = c2x_r * df['C2RCC'] / 100
    features['C2XC_r_x_C2RCC'] = c2xc_r * df['C2RCC'] / 100
    
    # Transformations (2)
    features['C2RCC_sq'] = df['C2RCC'] ** 2
    features['sqrt_C2RCC'] = np.sqrt(np.abs(df['C2RCC']))
    
    # Ratio (1)
    features['C2XC_r_div_C2RCC'] = c2xc_r / (df['C2RCC'] + 1e-5)
    
    return features

def get_models():
    """Get top 4 models for comparison"""
    return {
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        #'Ridge': Ridge(alpha=1.0, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42, early_stopping=True)
    }

def train_and_evaluate_by_trophic(df):
    """Train models and evaluate by trophic state"""
    print("\nTraining models and evaluating by trophic state...")
    
    # Split data (stratified by trophic state)
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42,
        stratify=df['Trophic_State']
    )
    
    # Calculate rescaling params from training data
    params = calculate_rescaling_params(train_df, train_df.index)
    
    # Create features
    X_train = create_optimized_features(train_df, params)
    X_test = create_optimized_features(test_df, params)
    
    y_train = train_df['Medicion'].values
    y_test = test_df['Medicion'].values
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models and get predictions
    models = get_models()
    predictions = {}
    
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        predictions[model_name] = y_pred
    
    # Calculate metrics by trophic state
    trophic_states = ['Mesotrophic', 'Eutrophic', 'Hypertrophic']
    results = []
    
    for trophic in trophic_states:
        mask = test_df['Trophic_State'] == trophic
        if mask.sum() > 0:
            y_true = test_df.loc[mask, 'Medicion'].values
            
            for model_name, y_pred_all in predictions.items():
                y_pred = y_pred_all[test_df.index.get_indexer(test_df[mask].index)]
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
                
                results.append({
                    'Trophic_State': trophic,
                    'Model': model_name,
                    'N': mask.sum(),
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                })
    
    results_df = pd.DataFrame(results)
    
    print("\nResults by trophic state:")
    print(results_df.to_string(index=False))
    
    return results_df, test_df

def generate_figure(results_df, test_df):
    """Generate 4-panel figure"""
    print("\nGenerating Figure 7...")
    
    fig = plt.figure(figsize=(7, 15))
    gs = fig.add_gridspec(3, 1, hspace=0.4, wspace=0.3)
    
    trophic_order = ['Mesotrophic', 'Eutrophic', 'Hypertrophic']
    trophic_order = TROPHIC_COLORS
    model_order = ['MLP', 'Random Forest','ElasticNet']
    colors = {"MLP": "#F4C50B", "Random Forest": "#079C43", "ElasticNet": "#099BC8"}
    
    # Panel (a): RMSE by trophic state
    ax1 = fig.add_subplot(gs[0, 0])
    
    rmse_pivot = results_df.pivot(index='Trophic_State', columns='Model', values='RMSE')
    rmse_pivot = rmse_pivot.reindex(trophic_order)
    
    x = np.arange(len(trophic_order))
    width = 0.2
    
    for i, model in enumerate(model_order):
        offset = (i - 1.5) * width
        values = rmse_pivot[model].values
        bars = ax1.bar(x + offset, values, width, label=model, alpha=0.8, color=colors[model])
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Trophic State', fontsize=11, fontweight='bold')
    ax1.set_ylabel('RMSE (mg/m³)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) RMSE Comparison Across Trophic States', fontsize=FONT_BASE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(trophic_order, fontsize=10)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(rmse_pivot.max()) * 1.2)
    
    # Panel (b): R² by trophic state
    ax2 = fig.add_subplot(gs[1, 0])
    
    r2_pivot = results_df.pivot(index='Trophic_State', columns='Model', values='R2')
    r2_pivot = r2_pivot.reindex(trophic_order)
    
    for i, model in enumerate(model_order):
        offset = (i - 1.5) * width
        values = r2_pivot[model].values
        bars = ax2.bar(x + offset, values, width, label=model, alpha=0.8, color=colors[model])
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                y_pos = max(val - 0.15, 0.15) if val > 0 else val + 0.12
                ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', 
                        fontsize=8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('Trophic State', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax2.set_title('(b) R² Comparison Across Trophic States', fontsize=FONT_BASE, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(trophic_order, fontsize=10)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel (c): Heatmap of RMSE values
    ax3 = fig.add_subplot(gs[2, 0])
    
    rmse_heatmap = rmse_pivot[model_order].T
    
    im = ax3.imshow(rmse_heatmap.values, cmap='YlOrRd', aspect='auto', 
                    vmin=0, vmax=rmse_heatmap.max().max())
    
    ax3.set_xticks(np.arange(len(trophic_order)))
    ax3.set_yticks(np.arange(len(model_order)))
    ax3.set_xticklabels(trophic_order, fontsize=10)
    ax3.set_yticklabels(["MLP", "RF", "E-N"], fontsize=10)
    
    # Add text annotations
    for i in range(len(model_order)):
        for j in range(len(trophic_order)):
            val = rmse_heatmap.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > rmse_heatmap.max().max() * 0.6 else 'black'
                ax3.text(j, i, f'{val:.1f}',
                        ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
    
    ax3.set_title('(c) RMSE Heatmap: All Models Show\nElevated Errors in Hypertrophic Waters', 
                  fontsize=FONT_BASE, fontweight='bold')
    ax3.set_xlabel('Trophic State', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('RMSE (mg/m³)', fontsize=10)
    
    # Panel (d): Performance summary table
    # ax4 = fig.add_subplot(gs[1, 1])
    # ax4.axis('off')
    
    # # Create summary statistics
    # summary_data = []
    # for trophic in trophic_order:
    #     trophic_results = results_df[results_df['Trophic_State'] == trophic]
        
    #     n = trophic_results['N'].iloc[0]
    #     rmse_mean = trophic_results['RMSE'].mean()
    #     rmse_std = trophic_results['RMSE'].std()
    #     mae_mean = trophic_results['MAE'].mean()
    #     r2_mean = trophic_results['R2'].mean()
        
    #     summary_data.append([
    #         trophic,
    #         f'N={n}',
    #         f'{rmse_mean:.1f}±{rmse_std:.1f}',
    #         f'{mae_mean:.1f}',
    #         f'{r2_mean:.2f}'
    #     ])
    
    # # Create table
    # col_labels = ['Trophic State', 'Samples', 'RMSE\n(mg/m³)', 'MAE\n(mg/m³)', 'R²\n(mean)']
    
    # table = ax4.table(cellText=summary_data, colLabels=col_labels,
    #                  cellLoc='center', loc='center',
    #                  colWidths=[0.24, 0.15, 0.22, 0.19, 0.20])
    
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)
    # table.scale(1, 2.5)
    
    # # Style header
    # for i in range(len(col_labels)):
    #     cell = table[(0, i)]
    #     cell.set_facecolor('#34495e')
    #     cell.set_text_props(weight='bold', color='white')
    
    # # Style rows by trophic state
    # for i, trophic in enumerate(trophic_order):
    #     for j in range(len(col_labels)):
    #         cell = table[(i+1, j)]
    #         cell.set_facecolor(trophic_colors[trophic])
    #         cell.set_alpha(0.3)
    #         if j == 0:
    #             cell.set_text_props(weight='bold')
    
    # ax4.set_title('(d) Performance Summary: Mesotrophic Waters\nAchieve Lowest Errors Despite Negative R²', 
    #               fontsize=12, fontweight='bold', pad=20)
    
    # # Add interpretation text
    # interpretation = (
    #     "Key Finding: Mesotrophic and eutrophic waters show negative R² due to\n"
    #     "narrow concentration ranges (low variance), where small prediction errors\n"
    #     "exceed group variance. However, low MAE/RMSE indicate good absolute accuracy.\n"
    #     "Hypertrophic waters show positive R² but higher absolute errors due to\n"
    #     "increased optical complexity and natural variability."
    # )
    
    # ax4.text(0.5, 0.02, interpretation, transform=ax4.transAxes,
    #         fontsize=9, ha='center', va='bottom',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
    #         style='italic')
    
    # Save figure
    plt.savefig('figures/fig7_performance_trophic.pdf', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/fig7_performance_trophic.pdf")
    
    plt.savefig('figures/fig7_performance_trophic.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: figures/fig7_performance_trophic.png")
    
    plt.close()

def main():
    """Main execution"""
    print("="*70)
    print("GENERATING FIGURE 7: PERFORMANCE BY TROPHIC STATE")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Train models and evaluate by trophic state
    results_df, test_df = train_and_evaluate_by_trophic(df)
    
    # Generate figure
    generate_figure(results_df, test_df)
    
    print("\n" + "="*70)
    print("FIGURE 7 GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nThe figure includes:")
    print("  (a) RMSE comparison across trophic states")
    print("  (b) R² comparison across trophic states")
    print("  (c) Heatmap showing consistent error patterns")
    print("  (d) Performance summary with interpretation")
    print("\nFiles saved:")
    print("  - figures/fig7_performance_trophic.pdf")
    print("  - figures/fig7_performance_trophic.png")

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    main()
