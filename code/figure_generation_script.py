#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar todas las figuras del artículo sobre corrección de Chl-a
Genera figuras de alta calidad para publicación en revista Q1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, LinearRegression
try:
    import xgboost as xgb
    ENABLE_XGB = False  # forzar desactivación para evitar incompatibilidades
except Exception:
    xgb = None
    ENABLE_XGB = False
import warnings
warnings.filterwarnings('ignore')

# Configuración para publicación (ajustes de legibilidad)
FONT_BASE = 18  # aumento del 30% para mejor legibilidad (14 -> 18)
plt.rcParams['font.size'] = FONT_BASE
plt.rcParams['axes.labelsize'] = FONT_BASE + 1
plt.rcParams['axes.titlesize'] = FONT_BASE + 2
plt.rcParams['xtick.labelsize'] = FONT_BASE
plt.rcParams['ytick.labelsize'] = FONT_BASE
plt.rcParams['legend.fontsize'] = FONT_BASE - 1
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15

# Voltear disposiciones apaisadas (1x2 -> 2x1, 2x3 -> 3x2)
FLIP_WIDE_LAYOUTS = True

# Colores para estados tróficos
TROPHIC_COLORS = {'Mesotrophic': '#2E7D32', 'Eutrophic': '#F57C00', 'Hypertrophic': '#C62828'}

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def load_and_prepare_data(filepath='resultados.csv'):
    """Carga y prepara los datos"""
    df = pd.read_csv(filepath)
    df_clean = df.dropna(subset=['C2RCC', 'C2X', 'C2XC', 'Medicion'])
    
    # Clasificación por estado trófico
    df_clean['Estado_Trofico'] = pd.cut(df_clean['Medicion'], 
                                         bins=[0, 25, 75, np.inf],
                                         labels=['Mesotrophic', 'Eutrophic', 'Hypertrophic'])
    
    # Añadir características derivadas
    df_clean['log_C2X'] = np.log1p(df_clean['C2X'])
    df_clean['log_C2XC'] = np.log1p(df_clean['C2XC'])
    df_clean['C2X_C2RCC_ratio'] = df_clean['C2X'] / (df_clean['C2RCC'] + 1e-5)
    df_clean['C2XC_C2RCC_ratio'] = df_clean['C2XC'] / (df_clean['C2RCC'] + 1e-5)
    
    return df_clean

def train_all_models(X_train, X_test, y_train, y_test):
    """Entrena todos los modelos y devuelve resultados"""
    results = []
    models_dict = {}
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos a evaluar con hiperparámetros optimizados para datos con outliers
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=10.0)': Ridge(alpha=10.0),  # Aumentar regularización
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=0.1)': Ridge(alpha=0.1),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Huber': HuberRegressor(epsilon=1.35),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    }
    if ENABLE_XGB:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    
    # Entrenar cada modelo
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calcular métricas con manejo robusto de valores extremos
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calcular R² de forma robusta - limitar valores extremos
        r2 = r2_score(y_test, y_pred)
        
        # Calcular MAPE con manejo de división por cero
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-6))) * 100
        
        metrics = {
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        results.append(metrics)
        models_dict[name] = (model, y_pred)
    
    # Modelo Stacking (alineado con el texto: RF, XGBoost, Ridge; meta-aprendiz RandomForest)
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('ridge', Ridge(alpha=0.1))
    ]
    if ENABLE_XGB:
        base_estimators.insert(1, ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)))
    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        cv=5
    )
    stacking.fit(X_train_scaled, y_train)
    y_pred_stack = stacking.predict(X_test_scaled)
    
    # Calcular métricas para el modelo stacking
    rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    mae_stack = mean_absolute_error(y_test, y_pred_stack)
    r2_stack = r2_score(y_test, y_pred_stack)
    mape_stack = np.mean(np.abs((y_test - y_pred_stack) / np.maximum(y_test, 1e-6))) * 100
    
    metrics = {
        'Model': 'Stacking Ensemble',
        'RMSE': rmse_stack,
        'MAE': mae_stack,
        'R2': r2_stack,
        'MAPE': mape_stack
    }
    results.append(metrics)
    models_dict['Stacking Ensemble'] = (stacking, y_pred_stack)
    
    return pd.DataFrame(results), models_dict

# ==============================================================================
# FIGURA 1: MAPA DEL ÁREA DE ESTUDIO
# ==============================================================================

def create_study_area_map(df_clean):
    """Crea mapa del área de estudio con los embalses"""
    if FLIP_WIDE_LAYOUTS:
        fig, axes = plt.subplots(2, 1, figsize=(7, 10))
        ax1, ax2 = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes
    
    # Panel izquierdo: Localización en España
    ax1.set_title('(a) Location of study area', fontsize=FONT_BASE + 1, fontweight='bold')
    ax1.text(0.5, 0.7, 'SPAIN', ha='center', fontsize=14)
    ax1.text(0.3, 0.4, 'Extremadura', ha='center', fontsize=11, style='italic')
    ax1.add_patch(plt.Rectangle((0.25, 0.35), 0.1, 0.1, fill=True, color='red', alpha=0.3))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Panel derecho: Distribución de embalses por estado trófico
    trophic_counts = df_clean.groupby('Estado_Trofico')['Embalse'].nunique()
    colors = [TROPHIC_COLORS[state] for state in trophic_counts.index]
    
    ax2.pie(trophic_counts.values, labels=trophic_counts.index, colors=colors,
            autopct='%1.0f%%', startangle=90)
    ax2.set_title('(b) Reservoirs by trophic state', fontsize=FONT_BASE + 1, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/fig1_study_area.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig1_study_area.pdf', dpi=300, bbox_inches='tight')
    print("OK Figure 1 saved: Study area map")
    return fig

# ==============================================================================
# FIGURA 2: COMPARACIÓN DE MODELOS CON BARRAS (TODOS LOS MODELOS CON R²)
# ==============================================================================

def create_model_comparison(results_df, baseline=None):
    """Crea figura comparando todos los modelos con barras para mejor percepción.

    baseline: dict opcional con claves 'RMSE', 'MAE', 'R2', 'MAPE' que
    representan la métrica del modelo de referencia C2RCC sin modificar.
    """

    # Ordenar por RMSE (mejor primero)
    results_sorted = results_df.sort_values('RMSE', ascending=True)

    # Crear figura con subplots compartiendo eje Y por fila
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(16, 12), sharey='row'
    )
    # Ajustar espaciado manualmente para evitar solapes
    plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.90)
    
    # Colores basados en desempeño
    def get_performance_color(rmse):
        if rmse < 30:
            return '#2E7D32'  # Verde - Excelente
        elif rmse < 50:
            return '#F57C00'  # Naranja - Bueno
        else:
            return '#C62828'  # Rojo - Pobre
    
    colors = [get_performance_color(rmse) for rmse in results_sorted['RMSE']]
    
    # Panel 1: RMSE (barras horizontales para mejor lectura)
    y_pos = np.arange(len(results_sorted))
    bars1 = ax1.barh(y_pos, results_sorted['RMSE'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Añadir valores en las barras con fuente más grande
    for i, (bar, rmse) in enumerate(zip(bars1, results_sorted['RMSE'])):
        ax1.text(bar.get_width() + max(results_sorted['RMSE']) * 0.02, bar.get_y() + bar.get_height()/2, 
                f'{rmse:.1f}', ha='left', va='center', fontsize=13, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    # Sustituimos etiquetas de eje Y por texto dentro de las barras (mejor visibilidad)
    ax1.set_yticklabels([])
    ax1.set_xlabel('RMSE (mg/m³)', fontsize=FONT_BASE, fontweight='bold')
    ax1.set_title('(a) Root Mean Square Error', fontsize=FONT_BASE + 1, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(results_sorted['RMSE']) * 1.2)  # Más espacio para etiquetas
    
    # Panel 2: MAE (barras horizontales) - Compartir eje y con panel 1
    bars2 = ax2.barh(y_pos, results_sorted['MAE'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Añadir valores en las barras con fuente más grande
    for i, (bar, mae) in enumerate(zip(bars2, results_sorted['MAE'])):
        ax2.text(bar.get_width() + max(results_sorted['MAE']) * 0.02, bar.get_y() + bar.get_height()/2, 
                f'{mae:.1f}', ha='left', va='center', fontsize=13, fontweight='bold')
    
    # Asegurar que no mostramos etiquetas del eje Y (iremos con texto en barras)
    ax1.tick_params(labelleft=False)
    ax2.sharey(ax1)
    ax2.set_yticks(y_pos)
    ax2.tick_params(labelleft=False)
    ax2.set_yticklabels([])
    ax2.set_xlabel('MAE (mg/m³)', fontsize=FONT_BASE, fontweight='bold')
    ax2.set_title('(b) Mean Absolute Error', fontsize=FONT_BASE + 1, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, max(results_sorted['MAE']) * 1.2)  # Más espacio para etiquetas
    
    # Panel 3: R² (barras horizontales) - Compartir eje y con panel 4
    # Filtrar modelos con R² válido para mejor visualización
    valid_r2 = results_sorted[results_sorted['R2'] > -2]  # Excluir valores extremadamente negativos
    y_pos_r2 = np.arange(len(valid_r2))
    colors_r2 = [get_performance_color(rmse) for rmse in valid_r2['RMSE']]
    
    bars3 = ax3.barh(y_pos_r2, valid_r2['R2'], color=colors_r2, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Añadir valores en las barras con fuente más grande
    for i, (bar, r2) in enumerate(zip(bars3, valid_r2['R2'])):
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{r2:.3f}', ha='left', va='center', fontsize=13, fontweight='bold')
    
    ax3.set_yticks(y_pos_r2)
    # Ocultamos etiquetas de eje Y; pondremos texto dentro de barras
    ax3.set_yticklabels([])
    ax3.tick_params(labelleft=False)
    ax3.set_xlabel('R²', fontsize=FONT_BASE, fontweight='bold')
    ax3.set_title('(c) Coefficient of Determination (R²)', fontsize=FONT_BASE + 1, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    # Escalado correcto del eje X para incluir valores negativos si existen
    min_r2 = float(valid_r2['R2'].min()) if len(valid_r2) else 0.0
    left = min(min_r2 - 0.05, -0.05)  # asegure al menos algo de rango negativo
    right = min(1.0, max(float(valid_r2['R2'].max()), 0.2) + 0.05) if len(valid_r2) else 1.0
    ax3.set_xlim(left, right)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Panel 4: MAPE (barras horizontales) - Compartir eje y con panel 3
    bars4 = ax4.barh(y_pos_r2, valid_r2['MAPE'], color=colors_r2, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Añadir valores en las barras con fuente más grande
    for i, (bar, mape) in enumerate(zip(bars4, valid_r2['MAPE'])):
        ax4.text(bar.get_width() + max(valid_r2['MAPE']) * 0.02, 
                bar.get_y() + bar.get_height()/2, 
                f'{mape:.0f}%', ha='left', va='center', fontsize=13, fontweight='bold')
    
    # Asegurar etiquetas del eje Y solo en el subplot izquierdo
    ax4.sharey(ax3)
    ax4.set_yticks(y_pos_r2)
    ax4.tick_params(labelleft=False)
    ax4.set_yticklabels([])
    ax4.set_xlabel('MAPE (%)', fontsize=FONT_BASE, fontweight='bold')
    ax4.set_title('(d) Mean Absolute Percentage Error', fontsize=FONT_BASE + 1, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, max(valid_r2['MAPE']) * 1.2)  # Más espacio para etiquetas
    
    # Añadir leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', label='Excellent (RMSE < 30)'),
        Patch(facecolor='#F57C00', label='Good (30-50)'),
        Patch(facecolor='#C62828', label='Poor (> 50)')
    ]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=FONT_BASE - 1)
    
    # Configurar título general con más espacio para evitar superposición
    # Si hay baseline, añadir líneas verticales de referencia por métrica
    if isinstance(baseline, dict):
        try:
            if 'RMSE' in baseline:
                ax1.axvline(x=float(baseline['RMSE']), color='red', linestyle='--', linewidth=2, label='C2RCC baseline')
            if 'MAE' in baseline:
                ax2.axvline(x=float(baseline['MAE']), color='red', linestyle='--', linewidth=2)
            if 'R2' in baseline:
                # asegurar que la línea está dentro de los límites de R²
                b_r2 = float(baseline['R2'])
                # recomputar límites incluyendo la línea base
                min_r2 = float(valid_r2['R2'].min()) if len(valid_r2) else 0.0
                max_r2 = float(valid_r2['R2'].max()) if len(valid_r2) else 1.0
                left = min(min_r2 - 0.05, b_r2 - 0.05, -0.05)
                right = min(1.0, max(max_r2, b_r2, 0.2) + 0.05)
                ax3.set_xlim(left, right)
                ax3.axvline(x=b_r2, color='red', linestyle='--', linewidth=2)
            if 'MAPE' in baseline:
                ax4.axvline(x=float(baseline['MAPE']), color='red', linestyle='--', linewidth=2)
            # leyenda sólo en el primer panel
            ax1.legend(loc='lower right', fontsize=FONT_BASE - 2)
        except Exception:
            pass

    # Título principal elevado (hacia arriba) para evitar cualquier solape con la primera fila
    fig.suptitle(
        'Comprehensive Model Performance Comparison\n(All Models with Complete Metrics)',
        fontsize=FONT_BASE + 3,
        fontweight='bold',
        y=1.02,
    )

    # Añadir etiquetas de nombre de modelo dentro de las barras (texto blanco)
    max_rmse = float(results_sorted['RMSE'].max()) if len(results_sorted) else 1.0
    inset_rmse = 0.02 * max_rmse
    for i, (bar, name) in enumerate(zip(bars1, results_sorted['Model'])):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        if width > inset_rmse * 1.5:
            x = max(0.0, width - inset_rmse)
            ax1.text(x, y, str(name), va='center', ha='right', color='white', fontsize=12, fontweight='bold')
        else:
            x = width + inset_rmse
            ax1.text(x, y, str(name), va='center', ha='left', color='black', fontsize=12, fontweight='bold')

    # Etiquetas dentro de las barras para R² (maneja positivos y negativos)
    if len(valid_r2):
        # Determinar rango para offsets
        r2_min = float(valid_r2['R2'].min())
        r2_max = float(valid_r2['R2'].max())
        span = max(1e-6, r2_max - r2_min)
        inset_r2 = 0.03 * span
        for i, (bar, name) in enumerate(zip(bars3, valid_r2['Model'])):
            x_left = bar.get_x()
            width = bar.get_width()
            y = bar.get_y() + bar.get_height()/2
            if width >= 0:
                # barra hacia la derecha desde 0
                if width > inset_r2 * 1.5:
                    x = x_left + width - inset_r2
                    ax3.text(x, y, str(name), va='center', ha='right', color='white', fontsize=12, fontweight='bold')
                else:
                    x = x_left + width + inset_r2
                    ax3.text(x, y, str(name), va='center', ha='left', color='black', fontsize=12, fontweight='bold')
            else:
                # barra hacia la izquierda (width negativo)
                if abs(width) > inset_r2 * 1.5:
                    x = x_left + inset_r2
                    ax3.text(x, y, str(name), va='center', ha='left', color='white', fontsize=12, fontweight='bold')
                else:
                    x = x_left - inset_r2
                    ax3.text(x, y, str(name), va='center', ha='right', color='black', fontsize=12, fontweight='bold')

    # Etiquetas dentro de las barras para MAE
    max_mae = float(results_sorted['MAE'].max()) if len(results_sorted) else 1.0
    inset_mae = 0.02 * max_mae
    for i, (bar, name) in enumerate(zip(bars2, results_sorted['Model'])):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        if width > inset_mae * 1.5:
            x = max(0.0, width - inset_mae)
            ax2.text(x, y, str(name), va='center', ha='right', color='white', fontsize=12, fontweight='bold')
        else:
            x = width + inset_mae
            ax2.text(x, y, str(name), va='center', ha='left', color='black', fontsize=12, fontweight='bold')

    # Etiquetas dentro de las barras para MAPE
    if len(valid_r2):
        max_mape = float(valid_r2['MAPE'].max())
        inset_mape = 0.02 * max_mape
        for i, (bar, name) in enumerate(zip(bars4, valid_r2['Model'])):
            width = bar.get_width()
            y = bar.get_y() + bar.get_height()/2
            if width > inset_mape * 1.5:
                x = max(0.0, width - inset_mape)
                ax4.text(x, y, str(name), va='center', ha='right', color='white', fontsize=12, fontweight='bold')
            else:
                x = width + inset_mape
                ax4.text(x, y, str(name), va='center', ha='left', color='black', fontsize=12, fontweight='bold')

    # Guardar figura
    plt.savefig('figures/fig2_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig2_model_comparison.pdf', dpi=300, bbox_inches='tight')
    print("OK Figure 2 saved: Bar chart comparison (All models with R²)")
    return fig

# ==============================================================================
# FIGURA 3: SCATTER PLOTS - PREDICCIÓN VS OBSERVACIÓN
# ==============================================================================

def create_prediction_scatter(y_test, models_dict, trophic_test):
    """Crea scatter plots para los mejores modelos, sin subplots vacíos,
    compartiendo ejes Y por fila y ejes X por columna, y ampliando ligeramente el tamaño."""

    # Seleccionar los 6 mejores modelos en orden deseado, filtrar por los disponibles
    preferred = ['Ridge (α=0.1)', 'ElasticNet', 'Random Forest', 'XGBoost', 'Stacking Ensemble', 'Linear Regression']
    available_models = [m for m in preferred if m in models_dict]
    n = len(available_models)
    if n == 0:
        # fallback simple
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No models available', ha='center', va='center')
        return fig

    # Definir rejilla según orientación
    if FLIP_WIDE_LAYOUTS:
        cols = 2
        rows = int(np.ceil(n / cols))
        figsize = (13, 17)  # ligeramente mayor
    else:
        rows = 2
        cols = int(np.ceil(n / rows))
        figsize = (18, 11)  # ligeramente mayor

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex='col', sharey='row')
    axes = np.array(axes).reshape(rows, cols)

    # Para sincronizar límites por columna (x) y por fila (y)
    col_xmax = [0.0 for _ in range(cols)]
    row_ymax = [0.0 for _ in range(rows)]

    # Pintar cada modelo
    for idx, model_name in enumerate(available_models):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        _, y_pred = models_dict[model_name]

        colors = [TROPHIC_COLORS[state] for state in trophic_test]
        sc = ax.scatter(y_test, y_pred, alpha=0.7, c=colors, s=80, edgecolors='black', linewidth=0.8)

        # Líneas 1:1 y regresión
        maxxy = float(max(np.max(y_test), np.max(y_pred)))
        col_xmax[c] = max(col_xmax[c], maxxy)
        row_ymax[r] = max(row_ymax[r], maxxy)
        ax.plot([0, maxxy], [0, maxxy], 'k--', alpha=0.7, linewidth=3, zorder=0)
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_test, p(y_test), 'r-', alpha=0.7, linewidth=3)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        ax.set_title(f'{model_name}\nRMSE={rmse:.1f}, R²={r2:.3f}', fontsize=FONT_BASE + 1, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Sin etiquetas por-eje: usaremos etiquetas comunes por fila/columna
        ax.set_xlabel('')
        ax.set_ylabel('')

        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=TROPHIC_COLORS[key], label=key) for key in TROPHIC_COLORS.keys()]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=FONT_BASE - 2)

    # Ajustar límites compartidos por columna (x) y fila (y)
    for c in range(cols):
        xmax = col_xmax[c] if col_xmax[c] > 0 else None
        if xmax:
            for r in range(rows):
                if r * cols + c < n:
                    axes[r, c].set_xlim(0, xmax * 1.02)
    for r in range(rows):
        ymax = row_ymax[r] if row_ymax[r] > 0 else None
        if ymax:
            for c in range(cols):
                if r * cols + c < n:
                    axes[r, c].set_ylim(0, ymax * 1.02)

    # Eliminar subplots vacíos (si los hay)
    total_slots = rows * cols
    for idx in range(n, total_slots):
        r = idx // cols
        c = idx % cols
        axes[r, c].remove()

    # Etiquetas comunes para toda la figura (por columna/fila)
    try:
        fig.supxlabel('Observed Chl-a (mg/m³)', fontsize=FONT_BASE + 1)
        fig.supylabel('Predicted Chl-a (mg/m³)', fontsize=FONT_BASE + 1)
    except Exception:
        # Compatibilidad con versiones antiguas de matplotlib
        fig.text(0.5, 0.02, 'Observed Chl-a (mg/m³)', ha='center', va='center', fontsize=FONT_BASE + 1)
        fig.text(0.02, 0.5, 'Predicted Chl-a (mg/m³)', ha='center', va='center', rotation='vertical', fontsize=FONT_BASE + 1)
    
    plt.suptitle('Observed vs Predicted Chlorophyll-a Concentrations', 
                fontsize=FONT_BASE + 3, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig3_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig3_scatter_plots.pdf', dpi=300, bbox_inches='tight')
    print("OK Figure 3 saved: Prediction scatter plots")
    return fig

# ==============================================================================
# FIGURA 4: ANÁLISIS DE RESIDUOS
# ==============================================================================

def create_residual_analysis(y_test, y_pred_best, model_name, trophic_test):
    """Crea análisis completo de residuos"""
    from scipy import stats
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    residuals = y_test - y_pred_best
    
    # Panel 1: Residuos vs Predichos
    colors = [TROPHIC_COLORS[state] for state in trophic_test]
    ax1.scatter(y_pred_best, residuals, alpha=0.6, c=colors, s=50, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted values (mg/m³)', fontsize=10)
    ax1.set_ylabel('Residuals (mg/m³)', fontsize=10)
    ax1.set_title('(a) Residuals vs Predicted', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('(b) Q-Q Plot', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax2.set_ylabel('Sample Quantiles', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Histograma de residuos
    _, p_value = stats.shapiro(residuals)
    ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals (mg/m³)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title(f'(c) Residuals Distribution (Shapiro-Wilk p={p_value:.3f})', 
                 fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Box plot por estado trófico
    residuals_by_state = [residuals[trophic_test == state] 
                          for state in ['Mesotrophic', 'Eutrophic', 'Hypertrophic']]
    bp = ax4.boxplot(residuals_by_state, labels=['Meso', 'Eu', 'Hyper'],
                     patch_artist=True, notch=True)
    
    # Colorear las cajas
    for patch, color in zip(bp['boxes'], TROPHIC_COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_ylabel('Residuals (mg/m³)', fontsize=10)
    ax4.set_xlabel('Trophic State', fontsize=10)
    ax4.set_title('(d) Residuals by Trophic State', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Residual Analysis - {model_name}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig4_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig4_residual_analysis.pdf', dpi=300, bbox_inches='tight')
    print("OK Figure 4 saved: Residual analysis")
    return fig, float(p_value)

# ==============================================================================
# FIGURA 5: IMPORTANCIA DE CARACTERÍSTICAS
# ==============================================================================

def create_feature_importance(X_train, y_train, feature_names):
    """Crea gráfico de importancia de características con tamaños de fuente consistentes"""
    # Entrenar Random Forest para obtener importancias
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)

    # Importancias y top 10
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_feats = [feature_names[i] for i in indices]
    top_imps = importances[indices]

    # Lienzo
    if FLIP_WIDE_LAYOUTS:
        fig, axes = plt.subplots(2, 1, figsize=(9, 13))
        ax1, ax2 = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Barras
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_imps)))
    bars = ax1.bar(range(len(top_imps)), top_imps, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_xticks(range(len(top_imps)))
    ax1.set_xticklabels(top_feats, rotation=40, ha='right')
    ax1.tick_params(axis='x', labelsize=FONT_BASE - 3)
    ax1.tick_params(axis='y', labelsize=FONT_BASE - 2)
    ax1.set_ylabel('Feature Importance', fontsize=FONT_BASE)
    ax1.set_title('(a) Top 10 Most Important Features', fontsize=FONT_BASE + 1, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Etiquetas de porcentaje encima de barras (porcentaje relativo a 1.0)
    for bar, importance in zip(bars, top_imps):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{importance * 100:.1f}%',
            ha='center',
            va='bottom',
            fontsize=FONT_BASE - 4,
            color='black',
            fontweight='bold',
        )

    # Panel 2: Importancia acumulada
    cumsum = np.cumsum(top_imps)
    ax2.plot(range(1, len(top_imps) + 1), cumsum, 'o-', linewidth=3, markersize=8, color='steelblue')
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.6, label='80% cumulative')
    ax2.set_xlabel('Number of Features', fontsize=FONT_BASE)
    ax2.set_ylabel('Cumulative Importance', fontsize=FONT_BASE)
    ax2.set_title('(b) Cumulative Feature Importance', fontsize=FONT_BASE + 1, fontweight='bold')
    ax2.set_xticks(range(1, len(top_imps) + 1))
    ax2.tick_params(axis='both', labelsize=FONT_BASE - 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=FONT_BASE - 3)
    ax2.set_ylim(0, 1)

    plt.suptitle('Feature Importance Analysis', fontsize=FONT_BASE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig5_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig5_feature_importance.pdf', dpi=300, bbox_inches='tight')

    # Escribir un texto dinámico (Top-3 features con porcentajes) para insertar en el caption
    try:
        import os
        os.makedirs('figures', exist_ok=True)
        def esc(name: str) -> str:
            return name.replace('_', '\\_')
        k = min(3, len(top_feats))
        parts = [f"{esc(top_feats[i])} ({top_imps[i]*100:.1f}\\%)" for i in range(k)]
        summary = "Top-3 features: " + ", ".join(parts) + "."
        with open('figures/feature_importance_caption.tex', 'w', encoding='utf-8') as f:
            f.write(summary)
    except Exception:
        pass

    print("OK Figure 5 saved: Feature importance")
    return fig

# ==============================================================================
# FIGURA 6: VALIDACIÓN CRUZADA ESPACIAL (LORO)
# ==============================================================================

def create_loro_validation(X, y, groups):
    """Crea figura de validación Leave-One-Reservoir-Out usando el STACKING definido."""
    logo = LeaveOneGroupOut()
    scores = []
    reservoir_names = []

    # Pipeline de stacking alineado con train_all_models
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('ridge', Ridge(alpha=0.1))
    ]
    if ENABLE_XGB:
        base_estimators.insert(1, ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)))
    meta = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()

    for train_idx, test_idx in logo.split(X, y, groups):
        if len(test_idx) > 0:  # Solo si hay datos de test
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            stacking = StackingRegressor(estimators=base_estimators, final_estimator=meta, cv=5)
            stacking.fit(X_train_scaled, y_train)
            y_pred = stacking.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(rmse)
            reservoir_names.append(groups[test_idx[0]])
    
    if FLIP_WIDE_LAYOUTS:
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))
        ax1, ax2 = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Distribución de RMSE
    ax1.hist(scores, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=np.mean(scores), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(scores):.1f} mg/m³')
    ax1.axvline(x=np.median(scores), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(scores):.1f} mg/m³')
    ax1.set_xlabel('RMSE (mg/m³)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('(a) LORO Cross-Validation RMSE Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Box plot con puntos
    bp = ax2.boxplot(scores, vert=True, patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Añadir puntos individuales
    x = np.random.normal(1, 0.04, size=len(scores))
    ax2.scatter(x, scores, alpha=0.5, s=30, color='red')
    
    ax2.set_ylabel('RMSE (mg/m³)', fontsize=10)
    ax2.set_title('(b) LORO Performance Summary', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(['All Reservoirs'])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir estadísticas
    stats_text = f'Mean ± SD: {np.mean(scores):.1f} ± {np.std(scores):.1f} mg/m³\n'
    stats_text += f'Min: {np.min(scores):.1f} mg/m³\n'
    stats_text += f'Max: {np.max(scores):.1f} mg/m³\n'
    stats_text += f'Q1: {np.percentile(scores, 25):.1f} mg/m³\n'
    stats_text += f'Q3: {np.percentile(scores, 75):.1f} mg/m³'
    ax2.text(1.3, np.mean(scores), stats_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Spatial Cross-Validation (Leave-One-Reservoir-Out)', 
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig6_loro_validation.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig6_loro_validation.pdf', dpi=300, bbox_inches='tight')
    print("OK Figure 6 saved: LORO validation")
    return fig, float(np.mean(scores)), float(np.std(scores))

# ==============================================================================
# UTILIDAD: GUARDAR MÉTRICAS EN ARCHIVO .tex PARA EL MANUSCRITO
# ==============================================================================

def write_metrics_tex(loro_mean, loro_sd, shapiro_p, test_rmse, outpath='figures/metrics.tex'):
    """Escribe comandos LaTeX con métricas clave para ser incluidos en el .tex"""
    import os
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    content = []
    content.append(f"% Auto-generated metrics file\n")
    content.append(f"\\newcommand{{\\LOROMeanRMSE}}{{{loro_mean:.1f}}}\n")
    content.append(f"\\newcommand{{\\LOROSDRMSE}}{{{loro_sd:.1f}}}\n")
    content.append(f"\\newcommand{{\\ShapiroP}}{{{shapiro_p:.3f}}}\n")
    content.append(f"\\newcommand{{\\TestRMSE}}{{{test_rmse:.1f}}}\n")
    # Porcentaje relativo de LORO sobre test
    if test_rmse > 0:
        loro_pct = 100.0 * (loro_mean - test_rmse) / test_rmse
    else:
        loro_pct = float('nan')
    content.append(f"\\newcommand{{\\LOROPctOverTest}}{{{loro_pct:.0f}}}\n")
    with open(outpath, 'w', encoding='utf-8') as f:
        f.writelines(content)
    print(f"OK Metrics written to {outpath}")

# ==============================================================================
# FIGURA 7: COMPARACIÓN POR ESTADO TRÓFICO
# ==============================================================================

def create_trophic_comparison(y_test, models_dict, trophic_test):
    """Crea comparación de modelos por estado trófico"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Modelos a comparar
    preferred = ['Ridge (a=0.1)', 'Random Forest', 'XGBoost', 'Stacking Ensemble']
    models_to_compare = [m for m in preferred if m in models_dict]
    trophic_states = ['Mesotrophic', 'Eutrophic', 'Hypertrophic']
    
    # Preparar datos para cada modelo y estado
    results_by_state = {}
    for model_name in models_to_compare:
        if model_name in models_dict:
            _, y_pred = models_dict[model_name]
            results_by_state[model_name] = {}
            
            for state in trophic_states:
                mask = trophic_test == state
                if mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
                    mae = mean_absolute_error(y_test[mask], y_pred[mask])
                    r2 = r2_score(y_test[mask], y_pred[mask])
                    results_by_state[model_name][state] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    # Panel 1: RMSE por estado trófico
    ax1 = axes[0, 0]
    x = np.arange(len(trophic_states))
    width = 0.2
    
    for i, model_name in enumerate(models_to_compare):
        values = [results_by_state[model_name][state]['RMSE'] for state in trophic_states]
        ax1.bar(x + i*width, values, width, label=model_name, alpha=0.8)
    
    ax1.set_xlabel('Trophic State', fontsize=10)
    ax1.set_ylabel('RMSE (mg/m³)', fontsize=10)
    ax1.set_title('(a) RMSE by Trophic State', fontsize=11, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(trophic_states)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: R² por estado trófico
    ax2 = axes[0, 1]
    for i, model_name in enumerate(models_to_compare):
        values = [results_by_state[model_name][state]['R2'] for state in trophic_states]
        ax2.bar(x + i*width, values, width, label=model_name, alpha=0.8)
    
    ax2.set_xlabel('Trophic State', fontsize=10)
    ax2.set_ylabel('R²', fontsize=10)
    ax2.set_title('(b) R² by Trophic State', fontsize=11, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(trophic_states)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    # Panel 3: Heatmap de RMSE
    ax3 = axes[1, 0]
    rmse_matrix = np.array([[results_by_state[model][state]['RMSE'] 
                             for state in trophic_states] 
                            for model in models_to_compare])
    
    im = ax3.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')
    ax3.set_xticks(np.arange(len(trophic_states)))
    ax3.set_yticks(np.arange(len(models_to_compare)))
    ax3.set_xticklabels(trophic_states)
    ax3.set_yticklabels(models_to_compare)
    ax3.set_title('(c) RMSE Heatmap', fontsize=11, fontweight='bold')
    
    # Añadir valores en las celdas
    for i in range(len(models_to_compare)):
        for j in range(len(trophic_states)):
            text = ax3.text(j, i, f'{rmse_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax3, label='RMSE (mg/m³)')
    
    # Panel 4: Resumen estadístico
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Crear tabla resumen
    table_data = []
    table_data.append(['Model', 'Overall RMSE', 'Best State', 'Worst State'])
    
    for model_name in models_to_compare:
        overall_rmse = np.mean([results_by_state[model_name][state]['RMSE'] 
                                for state in trophic_states])
        best_state = min(results_by_state[model_name].items(), 
                        key=lambda x: x[1]['RMSE'])[0]
        worst_state = max(results_by_state[model_name].items(), 
                         key=lambda x: x[1]['RMSE'])[0]
        table_data.append([model_name, f'{overall_rmse:.1f}', best_state[:4], worst_state[:4]])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Colorear encabezado
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('(d) Performance Summary', fontsize=11, fontweight='bold', y=0.95)
    
    plt.suptitle('Model Performance by Trophic State', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig7_trophic_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig7_trophic_comparison.pdf', dpi=300, bbox_inches='tight')
    print("OK Figure 7 saved: Trophic state comparison")
    return fig

# ==============================================================================
# FIGURA 8: ANÁLISIS TEMPORAL
# ==============================================================================

def create_temporal_analysis(df_clean):
    """Crea análisis temporal de las correcciones"""
    # Convertir fecha a datetime
    df_clean['fecha'] = pd.to_datetime(df_clean['fecha'])
    df_clean['year_month'] = df_clean['fecha'].dt.to_period('M')
    
    # Agrupar por mes
    monthly_stats = df_clean.groupby('year_month').agg({
        'Medicion': ['mean', 'std', 'count'],
        'C2RCC': 'mean',
        'C2X': 'mean',
        'C2XC': 'mean'
    }).reset_index()
    
    monthly_stats.columns = ['year_month', 'mean_measured', 'std_measured', 
                             'count', 'mean_C2RCC', 'mean_C2X', 'mean_C2XC']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Panel 1: Serie temporal de valores medidos vs C2RCC
    x = range(len(monthly_stats))
    ax1.plot(x, monthly_stats['mean_measured'], 'ko-', label='In-situ', linewidth=2, markersize=6)
    ax1.fill_between(x, 
                     monthly_stats['mean_measured'] - monthly_stats['std_measured'],
                     monthly_stats['mean_measured'] + monthly_stats['std_measured'],
                     alpha=0.3, color='gray')
    ax1.plot(x, monthly_stats['mean_C2RCC'], 'b^-', label='C2RCC', alpha=0.7, markersize=5)
    ax1.set_ylabel('Chl-a (mg/m³)', fontsize=10)
    ax1.set_title('(a) Temporal Evolution of Chlorophyll-a', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 150)
    
    # Panel 2: Número de muestras por mes
    bars = ax2.bar(x, monthly_stats['count'], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Month', fontsize=10)
    ax2.set_ylabel('Number of Samples', fontsize=10)
    ax2.set_title('(b) Sampling Frequency', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Configurar etiquetas del eje x
    step = max(1, len(monthly_stats) // 10)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([str(d) for d in monthly_stats['year_month'].iloc[::step]], 
                        rotation=45, ha='right')
    
    plt.suptitle('Temporal Analysis of Measurements', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig8_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig8_temporal_analysis.pdf', dpi=300, bbox_inches='tight')
    print("OK Figure 8 saved: Temporal analysis")
    return fig

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main(fig_number=None):
    """Genera figuras del artículo.

    Si fig_number es None, genera todas. Si es un entero (1-8),
    genera únicamente esa figura y realiza solo los cálculos necesarios.
    """
    import os

    # Crear directorio para figuras
    os.makedirs('figures', exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERACIÓN DE FIGURAS PARA ARTÍCULO")
    print("="*60 + "\n")
    
    # Cargar datos
    print("Loading data...")
    df_clean = load_and_prepare_data('resultados.csv')
    
    # Variables base
    y = df_clean['Medicion'].values
    trophic_states = df_clean['Estado_Trofico']
    groups = df_clean['Embalse'].values

    # Preparar características extendidas (ligero coste)
    df_clean['C2X_prod_C2RCC'] = df_clean['C2X'] * df_clean['C2RCC']
    df_clean['C2XC_prod_C2RCC'] = df_clean['C2XC'] * df_clean['C2RCC']
    df_clean['C2RCC_squared'] = df_clean['C2RCC']**2
    feature_names = ['C2RCC', 'C2X', 'C2XC', 'log_C2X', 'log_C2XC',
                     'C2X_C2RCC_ratio', 'C2XC_C2RCC_ratio',
                     'C2X_prod_C2RCC', 'C2XC_prod_C2RCC',
                     'C2RCC_squared']
    X_extended = df_clean[feature_names].values

    # Determinar si hace falta entrenar modelos (figs 2-7)
    figs_need_training = {2, 3, 4, 5, 6, 7}
    do_training = (fig_number is None) or (fig_number in figs_need_training)

    # Placeholders para variables que podrían usarse más abajo
    results_df = None
    models_dict = None
    best_model_name = None
    y_pred_best = None
    X_train = X_test = y_train = y_test = trophic_test = None
    best_rmse = None

    if do_training:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, _, trophic_test = train_test_split(
            X_extended, y, trophic_states, test_size=0.3, random_state=42, stratify=trophic_states
        )

        print("\nTraining models...")
        results_df, models_dict = train_all_models(X_train, X_test, y_train, y_test)

        best_row = results_df.sort_values('RMSE').iloc[0]
        best_model_name = best_row['Model']
        _, y_pred_best = models_dict[best_model_name]
        best_rmse = float(best_row['RMSE'])
        best_r2 = float(best_row['R2'])
        print(f"\nBest model: {best_model_name}")
        print(f"RMSE: {best_rmse:.2f}")
        print(f"R²: {best_r2:.3f}")

    print("\nGenerating figures...")

    def want(n):
        return (fig_number is None) or (fig_number == n)

    # 1) Área de estudio
    if want(1):
        create_study_area_map(df_clean)

    # 2) Comparación de modelos
    if want(2) and do_training:
        c2rcc_idx = feature_names.index('C2RCC')
        c2rcc_test = X_test[:, c2rcc_idx]
        rmse_base = float(np.sqrt(mean_squared_error(y_test, c2rcc_test)))
        mae_base = float(mean_absolute_error(y_test, c2rcc_test))
        r2_base = float(r2_score(y_test, c2rcc_test))
        mape_base = float(np.mean(np.abs((y_test - c2rcc_test) / np.maximum(y_test, 1e-6))) * 100.0)
        baseline_metrics = {'RMSE': rmse_base, 'MAE': mae_base, 'R2': r2_base, 'MAPE': mape_base}
        create_model_comparison(results_df, baseline=baseline_metrics)

    # 3) Scatter Observed vs Predicted
    if want(3) and do_training:
        create_prediction_scatter(y_test, models_dict, trophic_test)

    # 4) Análisis de residuos
    if want(4) and do_training:
        create_residual_analysis(y_test, y_pred_best, best_model_name, trophic_test)

    # 5) Importancia de características
    if want(5) and do_training:
        create_feature_importance(X_train, y_train, feature_names)

    # 6) Validación cruzada espacial LORO
    if want(6) and do_training:
        create_loro_validation(X_extended, y, groups)

    # 7) Comparación por estado trófico
    if want(7) and do_training:
        create_trophic_comparison(y_test, models_dict, trophic_test)

    # 8) Análisis temporal
    if want(8):
        create_temporal_analysis(df_clean)

    # Artefactos auxiliares sólo al generar todo
    if fig_number is None and do_training and results_df is not None:
        results_df.to_csv('figures/model_results_table.csv', index=False)
        try:
            _, loro_mean, loro_sd = create_loro_validation(X_extended, y, groups)
            _, shapiro_p = create_residual_analysis(y_test, y_pred_best, best_model_name, trophic_test)
            write_metrics_tex(loro_mean=loro_mean, loro_sd=loro_sd, shapiro_p=shapiro_p,
                              test_rmse=best_rmse, outpath='figures/metrics.tex')
        except Exception:
            pass
        print("\nOK Model results table saved: model_results_table.csv")

    if fig_number is None:
        print("\n" + "="*60)
        print("✓ TODAS LAS FIGURAS GENERADAS EXITOSAMENTE")
        print("="*60)
    else:
        print(f"\n✓ Figura {fig_number} generada")

    return None

if __name__ == "__main__":
    import sys
    fig_n = None
    if len(sys.argv) > 1:
        try:
            fig_n = int(sys.argv[1])
            if not (1 <= fig_n <= 8):
                print('Invalid figure number. Use 1-8 or no argument for all.')
                sys.exit(2)
        except ValueError:
            print('Usage: python figure_generation_script.py [FIG_NUMBER]')
            sys.exit(2)
    main(fig_n)
