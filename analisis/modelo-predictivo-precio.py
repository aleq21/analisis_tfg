import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor

# Cargar los datos
df = pd.read_csv('../datasets/processed/dataset-vehiculos-limpio.csv')

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
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Importancia de variables originales
# Agrupar importancias por variable original (no codificada)
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

# Agrupar por marca y comparar medias
grouped = X_test_copy.groupby('make')[['precio_real', 'precio_estimado']].mean().sort_values(by='precio_real', ascending=False)

# Gráfico de comparación completo
plt.figure(figsize=(14, 8))
ax = grouped.plot(kind='bar', figsize=(14, 8))
plt.ylabel("Precio medio (€)")
plt.title("Comparación del precio real vs. estimado por marca")
plt.xticks(rotation=90)
plt.tight_layout()
plt.legend(title='Precio', labels=['Real', 'Estimado'])
plt.savefig('../graficos/modelo-predictivo/precio_real_vs_estimado_por_marca.png', dpi=300)
plt.show()

