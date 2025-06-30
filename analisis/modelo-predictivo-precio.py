import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor

# Cargar los datos
df = pd.read_csv('../datasets/processed/dataset-vehiculos-limpio.csv')

# Convertir publish_date y extraer características temporales
df['publish_date'] = pd.to_datetime(df['publish_date'])
df['publish_year'] = df['publish_date'].dt.year
df['publish_month'] = df['publish_date'].dt.month
df['publish_dayofweek'] = df['publish_date'].dt.dayofweek
df.drop(columns=['publish_date'], inplace=True)

# Separar variables predictoras y target
X = df.drop(columns=['price'])
y = df['price']

# Identificar variables categóricas y numéricas
cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(exclude='object').columns.tolist()

# Preprocesamiento: codificación one-hot
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
], remainder='passthrough')

# Pipeline con XGBoost (GPU)
model = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', XGBRegressor(tree_method='hist', device='cuda', n_estimators=1000, random_state=42))
])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Importancia de variables originales
onehot = model.named_steps['preprocessing'].named_transformers_['cat']
encoded_cat_names = onehot.get_feature_names_out(cat_features)
all_feature_names = list(encoded_cat_names) + num_features

importances = model.named_steps['regressor'].feature_importances_

# Asociar cada feature codificado con su variable original
orig_features = []
for name in all_feature_names:
    for cat in cat_features:
        if name.startswith(cat + '_'):
            orig_features.append(cat)
            break
    else:
        orig_features.append(name)

# Agrupar por variable original
feat_df = pd.DataFrame({
    'Variables': orig_features,
    'Importancia en el precio': importances
})
feat_importance_df = feat_df.groupby('Variables').sum().sort_values(by='Importancia en el precio', ascending=False).reset_index()

# Gráfico de importancia
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_importance_df.head(15), x='Importancia en el precio', y='Variables')
plt.title("Importancia para predecir el precio")
plt.tight_layout()
plt.savefig('../graficos/modelo-predictivo/importancia_variables.png', dpi=300)
plt.show()

# Comparación precios reales vs. estimados por marca
X_test_copy = X_test.copy()
X_test_copy['precio_real'] = y_test
X_test_copy['precio_estimado'] = y_pred
grouped = X_test_copy.groupby('make')[['precio_real', 'precio_estimado']].mean().sort_values(by='precio_real', ascending=False)

grouped.plot(kind='bar', figsize=(14, 8))
plt.ylabel("Precio medio (€)")
plt.title("Comparación del precio real vs. estimado por marca")
plt.xticks(rotation=90)
plt.tight_layout()
plt.legend(title='Precio', labels=['Real', 'Estimado'])
plt.savefig('../graficos/modelo-predictivo/precio_real_vs_estimado_por_marca.png', dpi=300)
plt.show()

# Nuevos vehículos
nuevos_vehiculos = pd.DataFrame([
    {
        'make': 'Dacia',
        'model': 'Sandero',
        'version': '1.0 TCe Stepway Expression',
        'fuel': 'Gasolina',
        'year': 2025,
        'kms': 1672.0,
        'power': 91.0,
        'shift': 'automatic',
        'location': 'Madrid',
        'publish_date': '2025-01-01',
        'dealer_name': 'Autohero'
    },
    {
        'make': 'TOYOTA',
        'model': 'C-HR',
        'version': '2.0 GR Sport AWDi Hybrid 200',
        'fuel': 'Híbrido',
        'year': 2025,
        'kms': 5613.0,
        'power': 196.0,
        'shift': 'automatic',
        'location': 'Terrassa',
        'publish_date': '2025-06-04',
        'dealer_name': 'CSM Grup'
    },
    {
        'make': 'SEAT',
        'model': 'Ibiza',
        'version': '1.0 MPI Reference Salta',
        'fuel': 'Gasolina',
        'year': 2025,
        'kms': 10000.0,
        'power': 80.0,
        'shift': 'manual',
        'location': 'Castellón',
        'publish_date': '2025-06-10',
        'dealer_name': 'Seat Auto Esteller'
    },
    {
        'make': 'CITROEN',
        'model': 'C3',
        'version': 'PureTech Feel',
        'fuel': 'Gasolina',
        'year': 2022,
        'kms': 61352.0,
        'power': 83.0,
        'shift': 'manual',
        'location': 'Madrid',
        'publish_date': '2025-06-20',
        'dealer_name': 'AutosMadriD'
    },
    {
        'make': 'CITROEN',
        'model': 'Berlingo',
        'version': 'Talla M BlueHDi 130 SS 6v FEEL',
        'fuel': 'Diésel',
        'year': 2019,
        'kms': 80217.0,
        'power': 130.0,
        'shift': 'manual',
        'location': 'Barcelona',
        'publish_date': '2025-05-21',
        'dealer_name': 'HRmotor'
    },
    {
        'make': 'AUDI',
        'model': 'A5',
        'version': 'Avant TDI 204 CV Black Line',
        'fuel': 'Híbrido',
        'year': 2024,
        'kms': 100.0,
        'power': 204.0,
        'shift': 'automatic',
        'location': 'Barcelona',
        'publish_date': '2025-06-10',
        'dealer_name': 'Fecosauto'
    },
    {
        'make': 'AUDI',
        'model': 'A3',
        'version': 'RS3 Sportback TFSI quattro S tron',
        'fuel': 'Gasolina',
        'year': 2022,
        'kms': 54500.0,
        'power': 400.0,
        'shift': 'automatic',
        'location': 'Alicante',
        'publish_date': '2025-06-20',
        'dealer_name': 'Massini cars'
    }
])

# Procesar publish_date en nuevos datos
nuevos_vehiculos['publish_date'] = pd.to_datetime(nuevos_vehiculos['publish_date'])
nuevos_vehiculos['publish_year'] = nuevos_vehiculos['publish_date'].dt.year
nuevos_vehiculos['publish_month'] = nuevos_vehiculos['publish_date'].dt.month
nuevos_vehiculos['publish_dayofweek'] = nuevos_vehiculos['publish_date'].dt.dayofweek
nuevos_vehiculos.drop(columns=['publish_date'], inplace=True)

# Predicción
precios_estimados = model.predict(nuevos_vehiculos)
nuevos_vehiculos['precio_estimado'] = precios_estimados

# Mostrar resultados
print(nuevos_vehiculos[['make', 'model', 'year', 'precio_estimado']])