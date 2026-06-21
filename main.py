# -*- coding: utf-8 -*-
"""
TRABAJO DE LABORATORIO 1: DATA SCIENCE - MACHINE LEARNING
Modelo Random Forest OPTIMIZADO para detección de enfermedades cardiovasculares
"""

# 1. IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, learning_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    classification_report, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from scipy.stats import randint

# 2. CARGA Y PREPARACIÓN DEL DATASET

df = pd.read_csv("cardio_train.csv", sep=";")

print("=== INFO DEL DATASET ===")
print(df.info())
print(f"\nForma del dataset: {df.shape}")
print(f"\nValores nulos:\n{df.isnull().sum()}")
print(f"\nDistribución del target:\n{df['cardio'].value_counts(normalize=True).round(3)}")

# 3. LIMPIEZA DE DATOS

# Remover valores fisiológicamente imposibles
df = df[df['ap_hi'] > 0]       # presión sistólica positiva
df = df[df['ap_lo'] > 0]       # presión diastólica positiva
df = df[df['ap_hi'] < 300]     # cap superior razonable
df = df[df['ap_lo'] < 200]
df = df[df['height'] > 100]    # altura mínima razonable (cm)
df = df[df['height'] < 250]
df = df[df['weight'] > 30]     # peso mínimo razonable (kg)
df = df[df['weight'] < 250]

print(f"\nForma tras limpieza: {df.shape}")

# 4. INGENIERÍA DE FEATURES 

# IMC
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# Presión de pulso 
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# Edad en años 
df['age_years'] = (df['age'] / 365).astype(int)

print(f"\nFeatures finales: {list(df.columns)}")

# 5. SEPARACIÓN TRAIN / TEST

X = df.drop(labels='cardio', axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  
)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# 6. PIPELINE CON SCALER INTEGRADO

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
])

# 7. BÚSQUEDA DE HIPERPARÁMETROS 

param_dist = {
    'clf__n_estimators': randint(50, 200),
    'clf__max_depth': randint(5, 20),
    'clf__min_samples_split': randint(10, 60),
    'clf__min_samples_leaf': randint(2, 25),
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_features': ['sqrt', 'log2']
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=30,              
    cv=cv_strategy,
    scoring='f1_macro',    
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\n=== BUSCANDO MEJORES HIPERPARÁMETROS ===")
search.fit(X_train, y_train)

print(f"\nMejor F1 en validación cruzada: {search.best_score_:.4f}")
print(f"Mejores hiperparámetros: {search.best_params_}")

# Guardar resultados en CSV
results_df = pd.DataFrame(search.cv_results_)[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
].sort_values('rank_test_score')
results_df.to_csv('resultados_busqueda.csv', index=False)

# 8. EVALUACIÓN EN TEST

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # probabilidades para ROC/PR

print("\n=== MÉTRICAS EN TEST ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nReporte completo:\n{classification_report(y_test, y_pred)}")

# 9. VISUALIZACIONES

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Evaluación del Modelo Random Forest - Enfermedades Cardiovasculares", fontsize=14, fontweight='bold')

# 9.1 Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=axes[0, 0],
            annot_kws={"size": 18}, cbar=False)
axes[0, 0].set_title("Matriz de Confusión")
axes[0, 0].set_xlabel("Predicción")
axes[0, 0].set_ylabel("Real")

# 9.2 Curva ROC 
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_val = roc_auc_score(y_test, y_prob)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.4f}')
axes[0, 1].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
axes[0, 1].set_title("Curva ROC")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right")

# 9.3 Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
axes[0, 2].plot(recall, precision, color='green', lw=2, label=f'AP = {ap:.4f}')
axes[0, 2].axhline(y=y.mean(), color='navy', linestyle='--', label='Baseline')
axes[0, 2].set_title("Curva Precision-Recall")
axes[0, 2].set_xlabel("Recall")
axes[0, 2].set_ylabel("Precision")
axes[0, 2].legend()

# 9.4 Importancia de features 
rf_model = best_model.named_steps['clf']
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title("Importancia de Features")
axes[1, 0].set_xlabel("Importancia")

# 9.5 Curva de aprendizaje (NUEVA)
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train,
    cv=5, scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_std    = val_scores.std(axis=1)

axes[1, 1].plot(train_sizes, train_mean, 'o-', color='blue', label='Train')
axes[1, 1].plot(train_sizes, val_mean,   'o-', color='orange', label='Validación')
axes[1, 1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
axes[1, 1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
axes[1, 1].set_title("Curva de Aprendizaje")
axes[1, 1].set_xlabel("Tamaño del conjunto de entrenamiento")
axes[1, 1].set_ylabel("F1-score")
axes[1, 1].legend()

# 9.6 Métricas resumen 
tp_v, fn_v, fp_v, tn_v = confusion_matrix(y_test, y_pred).ravel()
metricas = {
    'Accuracy':    accuracy_score(y_test, y_pred),
    'Precision':   tp_v / (tp_v + fp_v),
    'Recall':      tp_v / (tp_v + fn_v),
    'Specificity': tn_v / (tn_v + fp_v),
    'F1-macro':    f1_score(y_test, y_pred, average='macro'),
    'AUC-ROC':     auc_val,
}
names  = list(metricas.keys())
values = list(metricas.values())
bars = axes[1, 2].barh(names, values, color='mediumseagreen')
axes[1, 2].set_xlim(0, 1.1)
axes[1, 2].set_title("Resumen de Métricas")
for bar, val in zip(bars, values):
    axes[1, 2].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("evaluacion_modelo.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico guardado como 'evaluacion_modelo.png'")

# 10. GUARDAR EL MODELO 

joblib.dump(best_model, 'modelo_cardio_rf.pkl')
joblib.dump(X.columns.tolist(), 'features_cardio.pkl')
print("\nModelo guardado como 'modelo_cardio_rf.pkl'")
print("Features guardados como 'features_cardio.pkl'")

# 11. FUNCIÓN DE PREDICCIÓN PARA NUEVOS PACIENTES 

def predecir_paciente(datos_paciente: dict) -> dict:
    """
    Predice el riesgo cardiovascular para un nuevo paciente.

    Parámetros esperados en datos_paciente:
        id, age (días), gender (1/2), height (cm), weight (kg),
        ap_hi, ap_lo, cholesterol (1/2/3), gluc (1/2/3),
        smoke (0/1), alco (0/1), active (0/1)

    Retorna: dict con predicción y probabilidad
    """
    modelo   = joblib.load('modelo_cardio_rf.pkl')
    features = joblib.load('features_cardio.pkl')

    df_pac = pd.DataFrame([datos_paciente])

    # Aplicar misma ingeniería de features
    df_pac['bmi']            = df_pac['weight'] / ((df_pac['height'] / 100) ** 2)
    df_pac['pulse_pressure'] = df_pac['ap_hi'] - df_pac['ap_lo']
    df_pac['age_years']      = (df_pac['age'] / 365).astype(int)

    df_pac = df_pac[features]  # orden correcto de columnas

    pred      = modelo.predict(df_pac)[0]
    prob      = modelo.predict_proba(df_pac)[0][1]
    riesgo    = "ALTO" if pred == 1 else "BAJO"

    print(f"\n=== PREDICCIÓN ===")
    print(f"Riesgo cardiovascular: {riesgo}")
    print(f"Probabilidad de enfermedad: {prob:.1%}")

    return {"prediccion": int(pred), "probabilidad": float(prob), "riesgo": riesgo}


# Ejemplo de uso con un paciente ficticio:
paciente_ejemplo = {
    'id': 999, 'age': 19000, 'gender': 1, 'height': 168, 'weight': 80,
    'ap_hi': 130, 'ap_lo': 85, 'cholesterol': 2, 'gluc': 1,
    'smoke': 0, 'alco': 0, 'active': 1
}
resultado = predecir_paciente(paciente_ejemplo)