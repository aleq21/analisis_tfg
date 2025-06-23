import os

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import folium
from folium.plugins import HeatMap

ruta_del_archivo = '../datasets/processed/dataset-vehiculos-limpio.csv'

try:
    # Cargar datos del csv limpio
    df = pd.read_csv(ruta_del_archivo)
    print("Dataset cargado.")

    # Contar el número de vehículos por ubicación
    location_counts = df['location'].value_counts().reset_index()
    location_counts.columns = ['ciudad', 'count']

    archivo_cache = '../coordenadas_cache.csv'

    print("Concentración de vehículos por ciudad:")
    print(location_counts)

    if os.path.exists(archivo_cache):
        print(f"Cargando caché desde '{archivo_cache}'...")
        cache_df = pd.read_csv(archivo_cache)
    else:
        print("No se encontró archivo de caché. Se creará uno nuevo.")
        cache_df = pd.DataFrame(columns=['ciudad', 'latitude', 'longitude'])

    cache_dict = cache_df.set_index('ciudad').T.to_dict('list')

    # Inicializar el geocodificador
    geolocator = Nominatim(user_agent="vehiculo_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Listas para guardar las nuevas coordenadas
    latitudes = []
    longitudes = []
    nuevas_coordenadas = False

    print("\nProcesando ciudades...")
    for ciudad in location_counts['ciudad']:
        if ciudad in cache_dict:
            # Si la ciudad está en la caché, usa los datos guardados
            print(f"  -> '{ciudad}': Encontrado en caché.")
            lat, lon = cache_dict[ciudad]
            latitudes.append(lat)
            longitudes.append(lon)
        else:
            # Si no, llama a la API
            print(f"  -> '{ciudad}': No encontrado. Llamando a la API de Nominatim...")
            try:
                location = geocode(f"{ciudad}, Spain")
                if location:
                    lat, lon = location.latitude, location.longitude
                    latitudes.append(lat)
                    longitudes.append(lon)
                    # Añadir la nueva ciudad al diccionario de caché para guardarlo después
                    cache_dict[ciudad] = [lat, lon]
                    nuevas_coordenadas = True
                else:
                    latitudes.append(None)
                    longitudes.append(None)
            except Exception as e:
                print(f"Error geocodificando {ciudad}: {e}")
                latitudes.append(None)
                longitudes.append(None)

    location_counts['latitude'] = latitudes
    location_counts['longitude'] = longitudes

    # Guardar la caché actualizada
    if nuevas_coordenadas:
        print("\nActualizando el archivo de caché con nuevas coordenadas...")
        updated_cache_df = pd.DataFrame.from_dict(cache_dict, orient='index', columns=['latitude', 'longitude']).reset_index()
        updated_cache_df.rename(columns={'index': 'ciudad'}, inplace=True)
        updated_cache_df.to_csv(archivo_cache, index=False)
        print(f"Caché guardada en '{archivo_cache}'.")

    print("\nProceso completado.")

    print("\nCiudades con sus coordenadas y recuento:")
    print(location_counts)

    # Eliminar filas donde no se encontraron coordenadas
    location_counts.dropna(subset=['latitude', 'longitude'], inplace=True)

    # Crear un mapa base centrado en España
    mapa_spain = folium.Map(location=[40.416775, -3.703790], zoom_start=6)

    # Preparar los datos para el mapa de calor: una lista de [lat, lon, peso]
    heat_data = [[row['latitude'], row['longitude'], row['count']] for index, row in location_counts.iterrows()]

    # Añadir la capa del mapa de calor al mapa base
    HeatMap(heat_data, radius=25, blur=15).add_to(mapa_spain)

    # Guardar el mapa en un archivo HTML
    mapa_spain.save("../mapa_concentracion_vehiculos.html")


except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{ruta_del_archivo}'. Asegúrate de que la ruta es correcta.")