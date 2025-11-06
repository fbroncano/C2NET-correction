"""Verify performance by trophic state using the optimized 10-feature model."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    """Load and prepare data"""
    df = pd.read_csv('resultados.csv')

    # Filter valid samples
    required_cols = ['Medicion', 'C2RCC', 'C2X', 'C2XC', 'Embalse']
    df_clean = df.dropna(subset=required_cols).copy()

    # Add trophic state classification by reservoir average
    print("Calculating trophic classification by reservoir average...")
    trophic_map = {}
    for reservoir in df_clean['Embalse'].unique():
        reservoir_mean = df_clean[df_clean['Embalse'] == reservoir]['Medicion'].mean()
        if reservoir_mean <= 25:
            trophic_map[reservoir] = 'Mesotrophic'
        elif reservoir_mean <= 75:
            trophic_map[reservoir] = 'Eutrophic'
        else:
            trophic_map[reservoir] = 'Hypertrophic'

    df_clean['Trophic_State'] = df_clean['Embalse'].map(trophic_map)

    print(f"\nTrophic distribution:")
    print(df_clean['Trophic_State'].value_counts())

    return df_clean

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
    temp_df = pd.DataFrame({'C2RCC': df['C2RCC'], 'C2X_r': c2x_r, 'C2XC_r': c2xc_r})
    features['Mean_r'] = temp_df.mean(axis=1)
    features['Median_r'] = temp_df.median(axis=1)

    # Interactions (2)
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

def evaluate_by_trophic_state(df):
    """Evaluate model performance by trophic state"""

    # Calculate rescaling parameters from full training data
    c2x_scale = df['C2RCC'].std() / df['C2X'].std()
    c2x_offset = df['C2RCC'].mean() - (df['C2X'].mean() * c2x_scale)

    c2xc_scale = df['C2RCC'].std() / df['C2XC'].std()
    c2xc_offset = df['C2RCC'].mean() - (df['C2XC'].mean() * c2xc_scale)

    params = {
        'c2x_scale': c2x_scale,
        'c2x_offset': c2x_offset,
        'c2xc_scale': c2xc_scale,
        'c2xc_offset': c2xc_offset
    }

    print(f"\nRescaling parameters:")
    print(f"  C2X:  scale={c2x_scale:.6f}, offset={c2x_offset:.2f}")
    print(f"  C2XC: scale={c2xc_scale:.6f}, offset={c2xc_offset:.2f}")

    # Create features
    X, feature_names = create_optimized_features(df, params)
    y = df['Medicion']

    # Train-test split (70-30) stratified by trophic state
    X_train, X_test, y_train, y_test, trophic_train, trophic_test = train_test_split(
        X, y, df['Trophic_State'],
        test_size=0.3,
        random_state=42,
        stratify=df['Trophic_State']
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ElasticNet model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Overall performance
    print(f"\n{'='*70}")
    print(f"OVERALL TEST SET PERFORMANCE (ElasticNet)")
    print(f"{'='*70}")
    print(f"RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} mg/m³")
    print(f"MAE:   {mean_absolute_error(y_test, y_pred):.2f} mg/m³")
    print(f"R²:    {r2_score(y_test, y_pred):.3f}")

    # Performance by trophic state
    print(f"\n{'='*70}")
    print(f"PERFORMANCE BY TROPHIC STATE (Test Set)")
    print(f"{'='*70}")
    print(f"\n{'Trophic State':<20} {'N':<6} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
    print(f"{'-'*70}")

    results = {}
    for trophic in ['Mesotrophic', 'Eutrophic', 'Hypertrophic']:
        mask = trophic_test == trophic
        if mask.sum() > 0:
            y_true_trophic = y_test[mask]
            y_pred_trophic = y_pred[mask]

            rmse = np.sqrt(mean_squared_error(y_true_trophic, y_pred_trophic))
            mae = mean_absolute_error(y_true_trophic, y_pred_trophic)
            r2 = r2_score(y_true_trophic, y_pred_trophic)

            results[trophic] = {'n': mask.sum(), 'rmse': rmse, 'mae': mae, 'r2': r2}

            print(f"{trophic:<20} {mask.sum():<6} {rmse:<12.2f} {mae:<12.2f} {r2:<10.3f}")

    print(f"\n{'='*70}")

    # Save results
    with open('TROPHIC_STATE_PERFORMANCE.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("PERFORMANCE BY TROPHIC STATE (ElasticNet with 10 optimized features)\n")
        f.write("="*70 + "\n\n")
        f.write("Test Set (70-30 split, stratified)\n\n")
        f.write(f"{'Trophic State':<20} {'N':<6} {'RMSE':<12} {'MAE':<12} {'R²':<10}\n")
        f.write("-"*70 + "\n")
        for trophic in ['Mesotrophic', 'Eutrophic', 'Hypertrophic']:
            if trophic in results:
                r = results[trophic]
                f.write(f"{trophic:<20} {r['n']:<6} {r['rmse']:<12.2f} {r['mae']:<12.2f} {r['r2']:<10.3f}\n")

    print(f"\nResults saved to: TROPHIC_STATE_PERFORMANCE.txt")

    return results

def main():
    print(f"\n{'#'*70}")
    print(f"# VERIFY TROPHIC STATE PERFORMANCE")
    print(f"# ElasticNet with 10 Optimized Features")
    print(f"{'#'*70}")

    df = load_data()
    results = evaluate_by_trophic_state(df)

    print(f"\n{'#'*70}")
    print(f"# VERIFICATION COMPLETE")
    print(f"{'#'*70}\n")

if __name__ == '__main__':
    main()
