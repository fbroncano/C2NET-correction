#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Completo de Modelos Ensemble para Estimación de Chl-a
Autor: Sistema de análisis para correcciones atmosféricas C2-Net
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   KFold, LeaveOneGroupOut, GridSearchCV)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                           r2_score, mean_absolute_percentage_error)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                            StackingRegressor, VotingRegressor)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                 ElasticNet, HuberRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

def load_and_prepare_data(filepath='resultados.csv'):
    """
    Carga y prepara los datos para el modelado
    """
    print("="*80)
    print("1. CARGA Y PREPARACIÓN DE DATOS")
    print("="*80)
    
    # Cargar datos
    df = pd.read_csv(filepath)
    print(f"\n✓ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # Limpiar datos nulos
    df_clean = df.dropna(subset=['C2RCC', 'C2X', 'C2XC', 'Medicion'])
    print(f"✓ Registros con datos completos: {df_clean.shape[0]}")
    
    # Análisis estadístico básico
    print("\nEstadísticas descriptivas:")
    print(df_clean[['C2RCC', 'C2X', 'C2XC', 'Medicion']].describe())
    
    # Clasificación por estado trófico
    df_clean['Estado_Trofico'] = pd.cut(df_clean['Medicion'], 
                                         bins=[0, 25, 75, np.inf],
                                         labels=['Mesotrófico', 'Eutrófico', 'Hipertrófico'])
    
    print(f"\nDistribución por estado trófico:")
    print(df_clean['Estado_Trofico'].value_counts())
    
    return df_clean

# ==============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================

def feature_engineering(df):
    """
    Crea características adicionales para mejorar el modelado
    """
    print("\n" + "="*80)
    print("2. INGENIERÍA DE CARACTERÍSTICAS")
    print("="*80)
    
    features = pd.DataFrame()
    
    # Características originales
    features['C2RCC'] = df['C2RCC']
    features['C2X'] = df['C2X']
    features['C2XC'] = df['C2XC']
    
    # Transformaciones logarítmicas (para manejar valores extremos)
    features['log_C2X'] = np.log1p(df['C2X'])
    features['log_C2XC'] = np.log1p(df['C2XC'])
    features['sqrt_C2RCC'] = np.sqrt(np.abs(df['C2RCC']))
    
    # Ratios
    features['C2X_C2RCC_ratio'] = df['C2X'] / (df['C2RCC'] + 1e-5)
    features['C2XC_C2RCC_ratio'] = df['C2XC'] / (df['C2RCC'] + 1e-5)
    features['C2XC_C2X_ratio'] = df['C2XC'] / (df['C2X'] + 1e-5)
    
    # Interacciones
    features['C2RCC_x_log_C2X'] = df['C2RCC'] * features['log_C2X']
    features['C2RCC_x_log_C2XC'] = df['C2RCC'] * features['log_C2XC']
    features['C2RCC_squared'] = df['C2RCC'] ** 2
    
    # Indicadores de confiabilidad
    features['all_agree'] = ((df['C2RCC'] < 50) & 
                             (df['C2X'] < 500) & 
                             (df['C2XC'] < 1000)).astype(int)
    
    # Estadísticas entre las tres correcciones
    features['mean_corrections'] = df[['C2RCC', 'C2X', 'C2XC']].mean(axis=1)
    features['std_corrections'] = df[['C2RCC', 'C2X', 'C2XC']].std(axis=1)
    features['median_corrections'] = df[['C2RCC', 'C2X', 'C2XC']].median(axis=1)
    
    print(f"\n✓ Características creadas: {features.shape[1]}")
    print(f"✓ Características: {list(features.columns)}")
    
    return features

# ==============================================================================
# 3. MÉTRICAS DE EVALUACIÓN
# ==============================================================================

def calculate_metrics(y_true, y_pred, name="Model"):
    """
    Calcula métricas completas de evaluación
    """
    # Evitar divisiones por cero
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'Model': name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'MBE': np.mean(y_pred - y_true),  # Mean Bias Error
        'MaxError': np.max(np.abs(y_true - y_pred))
    }
    
    return metrics

def evaluate_by_trophic_state(y_true, y_pred, trophic_states):
    """
    Evalúa el modelo por estado trófico
    """
    results = {}
    
    for state in ['Mesotrófico', 'Eutrófico', 'Hipertrófico']:
        mask = trophic_states == state
        if mask.sum() > 0:
            results[state] = {
                'n': mask.sum(),
                'RMSE': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                'MAE': mean_absolute_error(y_true[mask], y_pred[mask])
            }
    
    return results

# ==============================================================================
# 4. MODELOS BASE
# ==============================================================================

def get_base_models():
    """
    Define todos los modelos base a evaluar
    """
    models = {
        # Modelos lineales
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Huber': HuberRegressor(epsilon=1.35),
        
        # Modelos basados en árboles
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbosity=0
        ),
        
        # Otros modelos
        'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
    }
    
    return models

# ==============================================================================
# 5. MODELOS ENSEMBLE PERSONALIZADOS
# ==============================================================================

class WeightedEnsemble:
    """
    Ensemble con pesos optimizados
    """
    def __init__(self, models):
        self.models = models
        self.weights = None
        self.fitted_models = []
        
    def fit(self, X, y):
        from scipy.optimize import minimize
        
        # Entrenar todos los modelos
        self.fitted_models = []
        predictions = []
        
        for name, model in self.models.items():
            model_clone = model
            model_clone.fit(X, y)
            self.fitted_models.append((name, model_clone))
            predictions.append(model_clone.predict(X))
        
        predictions = np.array(predictions)
        
        # Optimizar pesos
        def objective(weights):
            weighted_pred = np.average(predictions, axis=0, weights=weights)
            return np.mean((weighted_pred - y) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(
            objective,
            x0=np.ones(len(self.models))/len(self.models),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x
        return self
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for _, model in self.fitted_models])
        return np.average(predictions, axis=0, weights=self.weights)

class AdaptiveCorrection:
    """
    Modelo adaptativo que usa diferentes estrategias según el rango
    """
    def __init__(self, threshold_c2x=500, threshold_c2xc=1000):
        self.threshold_c2x = threshold_c2x
        self.threshold_c2xc = threshold_c2xc
        self.model_reliable = None
        self.model_extreme = None
        self.bias = 0
        
    def fit(self, X, y):
        X_df = pd.DataFrame(X, columns=['C2RCC', 'C2X', 'C2XC'])
        
        # Identificar zonas confiables y extremas
        reliable_mask = (X_df['C2X'] < self.threshold_c2x) & (X_df['C2XC'] < self.threshold_c2xc)
        
        # Modelo para zona confiable
        if reliable_mask.sum() > 10:
            self.model_reliable = LinearRegression()
            X_reliable = X_df[reliable_mask][['C2RCC', 'C2X', 'C2XC']]
            # Aplicar transformación log a C2X y C2XC
            X_reliable_trans = X_reliable.copy()
            X_reliable_trans['C2X'] = np.log1p(X_reliable['C2X'])
            X_reliable_trans['C2XC'] = np.log1p(X_reliable['C2XC'])
            self.model_reliable.fit(X_reliable_trans, y[reliable_mask])
        
        # Para zona extrema, usar solo C2RCC con corrección de sesgo
        extreme_mask = ~reliable_mask
        if extreme_mask.sum() > 0:
            self.bias = np.mean(y[extreme_mask] - X_df[extreme_mask]['C2RCC'])
        
        return self
    
    def predict(self, X):
        X_df = pd.DataFrame(X, columns=['C2RCC', 'C2X', 'C2XC'])
        predictions = np.zeros(len(X))
        
        # Identificar zonas
        reliable_mask = (X_df['C2X'] < self.threshold_c2x) & (X_df['C2XC'] < self.threshold_c2xc)
        extreme_mask = ~reliable_mask
        
        # Predicciones para zona confiable
        if reliable_mask.sum() > 0 and self.model_reliable is not None:
            X_reliable = X_df[reliable_mask][['C2RCC', 'C2X', 'C2XC']]
            X_reliable_trans = X_reliable.copy()
            X_reliable_trans['C2X'] = np.log1p(X_reliable['C2X'])
            X_reliable_trans['C2XC'] = np.log1p(X_reliable['C2XC'])
            predictions[reliable_mask] = self.model_reliable.predict(X_reliable_trans)
        
        # Predicciones para zona extrema
        if extreme_mask.sum() > 0:
            predictions[extreme_mask] = X_df[extreme_mask]['C2RCC'] + self.bias
        
        return predictions

# ==============================================================================
# 6. EVALUACIÓN COMPLETA
# ==============================================================================

def evaluate_all_models(X, y, features_df, trophic_states):
    """
    Evalúa todos los modelos y compara resultados
    """
    print("\n" + "="*80)
    print("6. EVALUACIÓN DE MODELOS")
    print("="*80)
    
    # División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=trophic_states
    )
    
    # También dividir características extendidas
    X_feat_train, X_feat_test = train_test_split(
        features_df, test_size=0.3, random_state=42, stratify=trophic_states
    )
    
    # Estados tróficos para test
    _, trophic_test = train_test_split(
        trophic_states, test_size=0.3, random_state=42, stratify=trophic_states
    )
    
    print(f"\nTamaño conjunto entrenamiento: {len(X_train)}")
    print(f"Tamaño conjunto prueba: {len(X_test)}")
    
    results = []
    
    # 1. Baseline: Usar solo cada corrección individual
    print("\n--- Evaluando Baselines (Correcciones Individuales) ---")
    
    for i, col in enumerate(['C2RCC', 'C2X', 'C2XC']):
        y_pred = X_test[:, i]
        metrics = calculate_metrics(y_test, y_pred, f"{col} Solo")
        results.append(metrics)
        print(f"{col}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
    
    # 2. C2RCC con corrección de sesgo
    print("\n--- C2RCC con Corrección de Sesgo ---")
    bias = np.mean(y_train - X_train[:, 0])
    y_pred_corrected = X_test[:, 0] + bias
    metrics = calculate_metrics(y_test, y_pred_corrected, "C2RCC Corregido")
    results.append(metrics)
    print(f"Sesgo detectado: {bias:.2f} mg/m³")
    print(f"RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
    
    # 3. Modelos con características originales (3 variables)
    print("\n--- Modelos con Características Originales ---")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = get_base_models()
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            metrics = calculate_metrics(y_test, y_pred, name)
            results.append(metrics)
            print(f"{name}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error en {name}: {e}")
    
    # 4. Modelos con características extendidas
    print("\n--- Modelos con Características Extendidas ---")
    
    scaler_feat = StandardScaler()
    X_feat_train_scaled = scaler_feat.fit_transform(X_feat_train)
    X_feat_test_scaled = scaler_feat.transform(X_feat_test)
    
    best_extended_models = {
        'RF Extended': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        'XGB Extended': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42, verbosity=0)
    }
    
    for name, model in best_extended_models.items():
        try:
            model.fit(X_feat_train_scaled, y_train)
            y_pred = model.predict(X_feat_test_scaled)
            metrics = calculate_metrics(y_test, y_pred, name)
            results.append(metrics)
            print(f"{name}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
        except Exception as e:
            print(f"Error en {name}: {e}")
    
    # 5. Modelos Ensemble
    print("\n--- Modelos Ensemble ---")
    
    # Stacking
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
        ('ridge', Ridge())
    ]
    
    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=LinearRegression(),
        cv=5
    )
    
    stacking.fit(X_train_scaled, y_train)
    y_pred = stacking.predict(X_test_scaled)
    metrics = calculate_metrics(y_test, y_pred, "Stacking Ensemble")
    results.append(metrics)
    print(f"Stacking: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
    
    # Voting
    voting = VotingRegressor(base_estimators)
    voting.fit(X_train_scaled, y_train)
    y_pred = voting.predict(X_test_scaled)
    metrics = calculate_metrics(y_test, y_pred, "Voting Ensemble")
    results.append(metrics)
    print(f"Voting: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
    
    # Weighted Ensemble
    print("\n--- Weighted Ensemble Optimizado ---")
    weighted_models = {
        'RF': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGB': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'Ridge': Ridge()
    }
    
    weighted_ensemble = WeightedEnsemble(weighted_models)
    weighted_ensemble.fit(X_train_scaled, y_train)
    y_pred = weighted_ensemble.predict(X_test_scaled)
    metrics = calculate_metrics(y_test, y_pred, "Weighted Ensemble")
    results.append(metrics)
    print(f"Pesos óptimos: {weighted_ensemble.weights}")
    print(f"RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
    
    # Modelo Adaptativo
    print("\n--- Modelo Adaptativo ---")
    adaptive = AdaptiveCorrection()
    adaptive.fit(X_train, y_train)
    y_pred = adaptive.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "Adaptive Correction")
    results.append(metrics)
    print(f"RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    return results_df, stacking, trophic_test

# ==============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ==============================================================================

def plot_results(results_df, y_test, y_pred_best, model_name, trophic_test):
    """
    Genera visualizaciones completas de los resultados
    """
    print("\n" + "="*80)
    print("7. VISUALIZACIÓN DE RESULTADOS")
    print("="*80)
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Comparación de modelos - RMSE
    ax1 = plt.subplot(2, 3, 1)
    results_plot = results_df.head(15)  # Top 15 modelos
    colors = ['green' if x < 40 else 'orange' if x < 100 else 'red' 
              for x in results_plot['RMSE']]
    ax1.barh(range(len(results_plot)), results_plot['RMSE'], color=colors)
    ax1.set_yticks(range(len(results_plot)))
    ax1.set_yticklabels(results_plot['Model'])
    ax1.set_xlabel('RMSE (mg/m³)')
    ax1.set_title('Comparación de Modelos - RMSE')
    ax1.grid(True, alpha=0.3)
    
    # 2. Comparación de modelos - R²
    ax2 = plt.subplot(2, 3, 2)
    colors = ['green' if x > 0.8 else 'orange' if x > 0.5 else 'red' 
              for x in results_plot['R2']]
    ax2.barh(range(len(results_plot)), results_plot['R2'], color=colors)
    ax2.set_yticks(range(len(results_plot)))
    ax2.set_yticklabels(results_plot['Model'])
    ax2.set_xlabel('R²')
    ax2.set_title('Comparación de Modelos - R²')
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Predicho vs Real (mejor modelo)
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_test, y_pred_best, alpha=0.5, c=trophic_test.cat.codes, cmap='viridis')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_xlabel('Valor Real (mg/m³)')
    ax3.set_ylabel('Valor Predicho (mg/m³)')
    ax3.set_title(f'Predicción vs Real - {model_name}')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribución de residuos
    ax4 = plt.subplot(2, 3, 4)
    residuals = y_test - y_pred_best
    ax4.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residuos (mg/m³)')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Residuos')
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuos vs Predichos
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(y_pred_best, residuals, alpha=0.5, c=trophic_test.cat.codes, cmap='viridis')
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Valor Predicho (mg/m³)')
    ax5.set_ylabel('Residuos (mg/m³)')
    ax5.set_title('Residuos vs Predichos')
    ax5.grid(True, alpha=0.3)
    
    # 6. Box plot por estado trófico
    ax6 = plt.subplot(2, 3, 6)
    residuals_by_state = [residuals[trophic_test == state] 
                          for state in ['Mesotrófico', 'Eutrófico', 'Hipertrófico']]
    bp = ax6.boxplot(residuals_by_state, labels=['Meso', 'Eu', 'Hyper'])
    ax6.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax6.set_ylabel('Residuos (mg/m³)')
    ax6.set_title('Residuos por Estado Trófico')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis Completo de Resultados', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Tabla resumen de mejores modelos
    print("\n" + "="*50)
    print("TABLA RESUMEN - TOP 10 MODELOS")
    print("="*50)
    print(results_df.head(10).to_string(index=False))
    
    return fig

# ==============================================================================
# 8. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# ==============================================================================

def analyze_feature_importance(model, feature_names, X_test):
    """
    Analiza la importancia de las características
    """
    print("\n" + "="*80)
    print("8. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
    print("="*80)
    
    # Si el modelo tiene feature_importances_ (árboles)
    if hasattr(model, 'estimators_'):
        # Es un ensemble, obtener el primer estimador de tipo árbol
        for name, estimator in model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                break
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("El modelo no tiene información de importancia de características")
        return None
    
    # Crear DataFrame de importancias
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
    plt.xlabel('Importancia')
    plt.title('Top 10 Características Más Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 características más importantes:")
    print(importance_df.head(10).to_string(index=False))
    
    return importance_df

# ==============================================================================
# 9. VALIDACIÓN CRUZADA ESPACIAL
# ==============================================================================

def spatial_cross_validation(X, y, groups, model):
    """
    Realiza validación cruzada dejando un embalse fuera (Leave-One-Reservoir-Out)
    """
    print("\n" + "="*80)
    print("9. VALIDACIÓN CRUZADA ESPACIAL")
    print("="*80)
    
    logo = LeaveOneGroupOut()
    scores = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar y predecir
        model_clone = model
        model_clone.fit(X_train_scaled, y_train)
        y_pred = model_clone.predict(X_test_scaled)
        
        # Calcular RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(rmse)
    
    print(f"RMSE promedio (Leave-One-Reservoir-Out): {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"RMSE mínimo: {np.min(scores):.2f}")
    print(f"RMSE máximo: {np.max(scores):.2f}")
    
    return scores

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("\n" + "="*80)
    print(" ANÁLISIS COMPLETO DE MODELOS ENSEMBLE PARA ESTIMACIÓN DE CHL-A")
    print("="*80)
    
    # 1. Cargar y preparar datos
    df_clean = load_and_prepare_data('resultados.csv')
    
    # 2. Preparar variables
    X = df_clean[['C2RCC', 'C2X', 'C2XC']].values
    y = df_clean['Medicion'].values
    trophic_states = df_clean['Estado_Trofico']
    groups = df_clean['Embalse'].values
    
    # 3. Ingeniería de características
    features_df = feature_engineering(df_clean)
    
    # 4. Evaluación de todos los modelos
    results_df, best_model, trophic_test = evaluate_all_models(X, y, features_df, trophic_states)
    
    # 5. Obtener predicciones del mejor modelo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=trophic_states
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_pred_best = best_model.predict(X_test_scaled)
    
    # 6. Visualización
    fig = plot_results(results_df, y_test, y_pred_best, "Stacking Ensemble", trophic_test)
    
    # 7. Análisis de importancia
    analyze_feature_importance(best_model, ['C2RCC', 'C2X', 'C2XC'], X_test)
    
    # 8. Validación cruzada espacial
    spatial_scores = spatial_cross_validation(X, y, groups, best_model)
    
    # 9. Evaluación por estado trófico
    print("\n" + "="*80)
    print("10. EVALUACIÓN POR ESTADO TRÓFICO")
    print("="*80)
    
    trophic_results = evaluate_by_trophic_state(y_test, y_pred_best, trophic_test)
    for state, metrics in trophic_results.items():
        print(f"\n{state} (n={metrics['n']}):")
        print(f"  RMSE: {metrics['RMSE']:.2f} mg/m³")
        print(f"  MAE: {metrics['MAE']:.2f} mg/m³")
    
    # 10. Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    
    best_result = results_df.iloc[0]
    print(f"\n🏆 MEJOR MODELO: {best_result['Model']}")
    print(f"   RMSE: {best_result['RMSE']:.2f} mg/m³")
    print(f"   MAE: {best_result['MAE']:.2f} mg/m³")
    print(f"   R²: {best_result['R2']:.4f}")
    print(f"   MAPE: {best_result['MAPE']:.1f}%")
    
    # Comparación con baseline
    c2rcc_baseline = results_df[results_df['Model'] == 'C2RCC Solo'].iloc[0]
    improvement = ((c2rcc_baseline['RMSE'] - best_result['RMSE']) / c2rcc_baseline['RMSE']) * 100
    
    print(f"\n📈 MEJORA sobre C2RCC solo: {improvement:.1f}%")
    
    # Guardar resultados
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\n✓ Resultados guardados en 'model_comparison_results.csv'")
    
    return results_df, best_model

if __name__ == "__main__":
    results, model = main()
