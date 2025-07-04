import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px

ruta_del_archivo = '../datasets/processed/dataset-vehiculos-limpio.csv'

try:
    # Cargar datos del csv limpio
    df = pd.read_csv(ruta_del_archivo)
    print("Dataset cargado.")

    # Mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    print("Primeras 5 filas del dataset:")
    print(df.head())

    print("\nÚltimas 5 filas del dataset:")
    print(df.tail())
    pd.reset_option('display.max_columns')

    print(df.describe())

    # Análisis EDA con el dataset limpio
    print("\nInformación general del dataset:")
    df.info()
    print("\nNúmero de valores únicos por columna:")
    print(df.nunique())

    # Visualización de la distribución de las variables numéricas
    # Distribución de Precios
    fig = px.box(df, y='price', title='Distribución de precios')
    fig.update_layout(
        yaxis_title='Precio',
    )
    fig.write_image('../graficos/eda/distribucion_precios.png', scale=3)  # Guarda el gráfico
    fig.show()

    # Marca y modelo con el precio más alto
    marca_modelo_precio_max = df.loc[df['price'].idxmax(), ['make', 'model', 'price']]
    print(f"Marca y modelo con el precio más alto: {marca_modelo_precio_max['make']} {marca_modelo_precio_max['model']} - Precio: {marca_modelo_precio_max['price']} €")

    # Distribución de Kilometraje

    print(df['kms'].max())
    porcentaje_menor_200k = (df[df['kms'] < 200000].shape[0] / df.shape[0]) * 100
    print(f"Porcentaje de vehículos con menos de 200,000 km: {porcentaje_menor_200k:.2f}%")
    porcentaje_menor_130k = (df[df['kms'] < 130000].shape[0] / df.shape[0]) * 100
    print(f"Porcentaje de vehículos con menos de 130,000 km: {porcentaje_menor_130k:.2f}%")

    fig = px.box(df, x='kms', title='Distribución del kilometraje',
                 color_discrete_sequence=['salmon'])

    fig.update_layout(
        xaxis_title='Kilómetros',
        yaxis_title='',
        template='simple_white'
    )

    fig.write_image('../graficos/eda/boxplot_kilometraje_plotly.png', scale=3)
    fig.show()


    # Distribución de Potencia
    plt.figure(figsize=(10, 5))
    sns.histplot(df['power'], bins=40, color='green')
    plt.title('Distribución de potencia (CV)')
    plt.xlabel('Potencia')
    plt.ylabel('Frecuencia')
    plt.savefig('../graficos/eda/distribucion_potencia.png', dpi=300)
    plt.show()

    # Distribución de Año de Fabricación
    plt.figure(figsize=(10, 5))
    sns.histplot(df['year'], kde=False, color='gold', discrete=True)
    plt.title('Distribución de los años de fabricación')
    plt.xlabel('Año')
    plt.ylabel('Frecuencia')
    plt.savefig('../graficos/eda/distribucion_anno_fabricacion.png', dpi=300)
    plt.show()

    # Visualización de la frecuencia de las variables categóricas
    # Número de Vehículos por Marca
    plt.figure(figsize=(12, 10))
    sns.countplot(y='make', data=df, order=df['make'].value_counts().index, hue='make', palette='viridis', legend=False)
    plt.title('Número de vehículos por marca')
    plt.xlabel('Frecuencia')
    plt.ylabel('Marca')
    plt.savefig('../graficos/eda/numero_vehiculos_por_marca.png', dpi=300)
    plt.show()

    # Ver 8 marcas con más vehículos en venta con su numero y porcentaje
    marcas_mas_ventas = df['make'].value_counts().head(8)
    print("\nMarcas con más vehículos en venta:")
    print(marcas_mas_ventas)
    porcentaje_ventas = (marcas_mas_ventas / df.shape[0]) * 100
    print("\nPorcentaje de vehículos en venta por marca:")
    print(porcentaje_ventas)


    # Número de Vehículos por Combustible
    plt.figure(figsize=(10, 5))
    sns.countplot(x='fuel', data=df, order=df['fuel'].value_counts().index, hue='fuel', palette='plasma', legend=False)
    plt.title('Número de vehículos por tipo de combustible')
    plt.xlabel('Tipo de combustible')
    plt.ylabel('Frecuencia')
    plt.savefig('../graficos/eda/numero_vehiculos_por_combustible.png', dpi=300)
    plt.show()

    # Numero de vehiculos hibridos
    numero_hibridos = df[df['fuel'] == 'Híbrido'].shape[0]
    print(f"Número de vehículos híbridos: {numero_hibridos}")

    # unknown en combustible
    numero_unknown_combustible = df[df['fuel'] == 'unknown'].shape[0]
    print(f"Número de vehículos con combustible 'unknown': {numero_unknown_combustible}")

    # Número de Vehículos por Tipo de Transmisión
    plt.figure(figsize=(10, 5))
    sns.countplot(x='shift', data=df, order=df['shift'].value_counts().index, hue='shift', palette='magma', legend=False)
    plt.title('Número de vehículos por tipo de transmisión')
    plt.xlabel('Transmisión')
    plt.ylabel('Frecuencia')
    plt.savefig('../graficos/eda/numero_vehiculos_por_transmision.png', dpi=300)
    plt.show()

    # Relación entre variables numéricas
    # Calcular la matriz de correlación solo para columnas numéricas
    corr_matrix = df[['price', 'kms', 'power', 'year']].corr()

    # Visualizar la matriz de correlación con un mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Mapa de calor de correlación de las variables numéricas')
    plt.savefig('../graficos/eda/mapa_calor_correlacion.png', dpi=300)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='fuel', y='price', hue='shift', data=df, estimator='mean', errorbar='sd', palette='pastel')
    plt.title('Precio medio según combustible y transmisión')
    plt.xlabel('Tipo de combustible')
    plt.ylabel('Precio medio')
    plt.legend(title='Transmisión')
    plt.tight_layout()
    plt.savefig('../graficos/eda/precio_medio_combustible_transmision.png', dpi=300)
    plt.show()

    # ¿Cuál es el precio medio de venta por marca?
    # Agrupar por 'make' y calcular la media de 'price'
    media_precio_por_marca = df.groupby('make')['price'].mean().reset_index()

    # Ordenar los valores para una mejor visualización posterior
    media_precio_por_marca = media_precio_por_marca.sort_values(by='price', ascending=False)

    # Mostrar el resultado
    print("Precio medio de venta por marca:")
    print(media_precio_por_marca)

    # Preparar etiquetas para el gráfico (Marca y Precio)
    labels = [f"{row['make']}\n({row['price']:,.0f} €)" for index, row in media_precio_por_marca.iterrows()]

    # Crear el gráfico
    plt.figure(figsize=(26, 16))
    squarify.plot(sizes=media_precio_por_marca['price'],
                  label=labels,
                  alpha=0.8,
                  color=sns.color_palette("Paired", len(media_precio_por_marca)),
                  text_kwargs={'fontsize': 20}
                  )

    plt.title('Precio medio por marca', fontsize=24)
    plt.axis('off')
    plt.savefig('../graficos/eda/precio_medio_por_marca.png', dpi=300)
    plt.show()

    # Fecha media de fabricación de los vehículos por marca
    # Calcular la fecha media de fabricación
    media_anno_fabricacion = df.groupby('make')['year'].mean().reset_index()
    media_anno_fabricacion = media_anno_fabricacion.sort_values(by='year', ascending=False)

    # Crear el gráfico de barras vertical
    plt.figure(figsize=(16, 8))
    barplot = sns.barplot(x='make', y='year', hue='make', data=media_anno_fabricacion, palette='bright', legend=False)
    plt.title('Fecha media de fabricación por marca de vehículo', fontsize=16)
    plt.xlabel('Marca', fontsize=11)
    plt.ylabel('Año medio de fabricación', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    plt.ylim(bottom=media_anno_fabricacion['year'].min() - 1, top=media_anno_fabricacion['year'].max() + 1)

    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.0f}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom',
                         xytext=(0, 9),
                         textcoords='offset points',
                         rotation=90)
    plt.savefig('../graficos/eda/fecha_media_fabricacion_por_marca.png', dpi=300)
    plt.show()

    # Top 10 de modelos de coches en venta (marca + modelo)
    # Crear una nueva columna combinando marca y modelo para evitar ambigüedades
    df['modeloEntero'] = df['make'] + ' ' + df['model']

    print("DataFrame con la nueva columna 'modeloEntero':")
    print(df.head())

    # Contar la frecuencia de cada modelo completo y obtener el top 10
    top_10_models = df['modeloEntero'].value_counts().head(10).reset_index()

    # Renombrar las columnas para mayor claridad
    top_10_models.columns = ['modeloEntero', 'count']

    plt.figure(figsize=(12, 8))

    barplot = sns.barplot(x='count', y='modeloEntero', hue='modeloEntero', data=top_10_models, palette='rocket', legend=False)

    plt.title('Top 10 modelos de vehículos más frecuentes en venta', fontsize=16)
    plt.xlabel('Número de vehículos en venta', fontsize=12)
    plt.ylabel('Modelo del vehículo', fontsize=12)

    for index, value in enumerate(top_10_models['count']):
        plt.text(value, index, f' {value}', va='center')

    plt.tight_layout()
    plt.savefig('../graficos/eda/top_10_modelos_vehiculos.png', dpi=300)
    plt.show()

except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{ruta_del_archivo}'. Asegúrate de que la ruta es correcta.")