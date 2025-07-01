import pandas as pd
import numpy as np

def imputar_por_remuestreo(df, columna_objetivo, columna_grupo, semilla=42):

    np.random.seed(semilla)

    # Agrupar por la columna indicada
    for valor_grupo, grupo in df.groupby(columna_grupo):
        # Máscara para ubicar los valores faltantes en este grupo
        mascara_faltantes = (df[columna_grupo] == valor_grupo) & (df[columna_objetivo].isna())
        n_faltantes = mascara_faltantes.sum()
        valores_disponibles = grupo[columna_objetivo].dropna().values

        # Si hay valores para remuestrear e imputar
        if n_faltantes > 0 and len(valores_disponibles) > 0:
            valores_imputados = np.random.choice(valores_disponibles, size=n_faltantes, replace=True)
            df.loc[mascara_faltantes, columna_objetivo] = valores_imputados

    return df

# Ruta del dataset a analizar
ruta_del_archivo = '../datasets/raw/vehiculos-de-segunda-mano-sample.csv'

try:
    # Primera lectura
    # Cargar datos del csv
    df = pd.read_csv(ruta_del_archivo)
    print("Dataset cargado.")

    # Mostrar todas las columnas
    pd.set_option('display.max_columns', None)

    print("Primeras 5 filas del dataset:")
    print(df.head())

    pd.reset_option('display.max_columns')

    print("\nInformación general del dataset:")
    df.info()
    print(df['power'].describe())
    print(df['kms'].describe())

    # Mostrar la información de los vehículos de la marca KTM
    print("\nInformación de los vehículos de la marca KTM:")
    print(df[df['make'] == 'KTM'].info())
    print("\nPrecio medio de los vehículos de la marca KTM:")
    print(df[df['make'] == 'KTM']['price'].mean())
    print("\nNúmero de vehículos de la marca KTM:")
    print(df[df['make'] == 'KTM'].shape[0])
    print("\nNombres del modelo de KTM:")
    print(df[df['make'] == 'KTM']['model'].unique())



    # Mostrar el número de valores únicos por columna
    print("\nNúmero de valores únicos por columna:")
    print(df.nunique())

    # Eliminar columnas completamente nulas
    df.drop(columns=['doors', 'color'], inplace=True)

    # Eliminar columnas irrelevantes o con poco valor para el análisis
    df.drop(columns=['vehicle_type', 'photos', 'description', 'currency', 'update_date', 'dealer_description', 'dealer_address', 'dealer_zip_code', 'dealer_country_code', 'dealer_is_professional', 'dealer_website', 'dealer_registered_at', 'dealer_city', 'date'], inplace=True)

    # Trasformación de varias columnas a datetime
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')

    # Imputar unknown en las columnas categoricas
    df['version'] = df['version'].fillna('unknown')
    df['fuel'] = df['fuel'].fillna('unknown')
    df['shift'] = df['shift'].fillna('unknown')
    df['dealer_name'] = df['dealer_name'].fillna('unknown')

    # Imputar kms por año de fabricación
    df = imputar_por_remuestreo(df, columna_objetivo='kms', columna_grupo='year')

    # Imputar power por modelo
    df = imputar_por_remuestreo(df, columna_objetivo='power', columna_grupo='model')

    # Después de la primera limpieza
    print("\nInformación general del dataset:")
    df.info()

    # Ver modelos cuyos valores de 'power' son nulos
    print("\nModelos con 'power' nulo:")
    modelos_con_power_nulo = df[df['power'].isna()]['model'].unique()
    print(modelos_con_power_nulo)

    # Añadir valor manualmente al modelo con valor nulo
    df.loc[df['model'] == 'ë-Jumpy', 'power'] = 136
    df.loc[(df['model'] == 'Wrangler') & (df['year'] == 2006), 'power'] = 177
    df.loc[(df['model'] == 'Model S') & (df['year'] == 2019), 'power'] = 333
    df.loc[(df['model'] == 'A110') & (df['year'] == 1980), 'power'] = 80
    df.loc[(df['model'] == 'S8') & (df['year'] == 2000), 'power'] = 571

    df['fuel'] = df['fuel'].replace('Otros', 'Híbrido')

    keywords = ['PHEV', 'HEV', 'EV', 'Hybrid', 'mhev', 'kw', 'kwh', '0h', '5h', 'electric', 'hibrido', 'híbrido']
    pattern = '|'.join(keywords)
    mask = (df['fuel'] == 'unknown') & (df['version'].str.contains(pattern, case=False, na=False))
    df.loc[mask, 'fuel'] = 'Híbrido'

    # Verificar que se ha añadido correctamente
    print("\nModelos con 'power' nulo después de la imputación manual:")
    modelos_con_power_nulo = df[df['power'].isna()]['model'].unique()
    print(modelos_con_power_nulo)

    print("\nInformación general del dataset final:")
    df.info()

    # Mostrar el número de valores únicos por columna
    print("\nNúmero de valores únicos por columna después de la limpieza:")
    print(df.nunique())

    # Guardar el DataFrame limpio en un nuevo archivo CSV
    print("\nGuardando el dataset limpio...")
    df.to_csv('../datasets/processed/dataset-vehiculos-limpio.csv', index=False)

except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{ruta_del_archivo}'. Asegúrate de que la ruta es correcta.")



