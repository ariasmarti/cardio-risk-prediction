#  CardioRisk — Predicción temprana de riesgo cardiovascular

Aplicación de Machine Learning que estima el riesgo de enfermedad cardiovascular a partir de datos clínicos del paciente, utilizando un modelo Random Forest entrenado sobre más de 68.000 registros médicos.

** Demo en vivo:** https://cardio-risk-prediction-ariasmarti.streamlit.app/

>
---

##  Descripción

El proyecto fue desarrollado como Trabajo de Laboratorio para la materia *Data Science - Machine Learning*. El objetivo es proponer una arquitectura de IA que colabore con la automatización de una alerta temprana de riesgo cardiovascular, a partir de variables clínicas simples y accesibles (presión arterial, colesterol, glucosa, peso, altura, hábitos).

##  Modelo

- **Algoritmo:** Random Forest Classifier
- **Dataset:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (Kaggle, ~70.000 registros)
- **Pipeline:** escalado Min-Max + Random Forest, optimizado con `RandomizedSearchCV` y validación cruzada estratificada (5 folds)
- **Feature engineering:** IMC, presión de pulso, edad en años derivada
- **Métricas en test:**

| Métrica | Valor |
|---|---|
| Accuracy | 0.73 |
| F1-score (macro) | 0.73 |
| AUC-ROC | 0.80 |

##  Funcionalidades de la app

- Formulario interactivo con datos personales, signos vitales y hábitos del paciente
- Predicción de riesgo (alto/bajo) con probabilidad asociada
- Indicadores clínicos calculados automáticamente (IMC, presión de pulso)
- Visualización de importancia de variables del modelo
- Aviso médico: la herramienta es un prototipo académico, no reemplaza diagnóstico profesional

##  Tecnologías

- **Python** — pandas, numpy, scikit-learn, joblib
- **Streamlit** — interfaz web interactiva
- **Matplotlib / Seaborn** — visualizaciones

##  Estructura del repositorio

```
├── main.py                    # Entrenamiento, optimización y guardado del modelo
├── app.py                     # Interfaz Streamlit
├── modelo_cardio_rf.pkl       # Modelo entrenado (serializado)
├── features_cardio.pkl        # Lista de features usadas por el modelo
├── evaluacion_modelo.png      # Gráficos de evaluación (matriz de confusión, ROC, etc.)
├── resultados_busqueda.csv    # Resultados de la búsqueda de hiperparámetros
└── requirements.txt           # Dependencias del proyecto
```

##  Cómo correrlo localmente

```bash
# Clonar el repositorio
git clone https://github.com/ariasmarti/cardio-risk-prediction.git
cd cardio-risk-prediction

# Instalar dependencias
pip install -r requirements.txt

# (Opcional) Reentrenar el modelo desde cero
python main.py

# Levantar la app
streamlit run app.py
```
