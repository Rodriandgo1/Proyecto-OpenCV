"""
Enunciado de mi idea
"""
import cv2
import numpy as np

import winsound
import time
import math
import wave

"""
Captura de la partitura y cambio de colores.
Este módulo carga una imagen estática de una partitura, la convierte a escala de grises, 
la suaviza para eliminar ruido y finalmente aplica un umbral invertido para que 
el fondo (papel) quede negro y las figuras (notas y líneas) destaquen en blanco puro.
"""
def intercambiarColores(ruta):
    # Cargamos la imagen 
    img = cv2.imread(ruta)
    
    # Validación de seguridad: comprobamos que la imagen se ha cargado correctamente
    if img is None:
        print("Error: no se ha podido cargar la imagen.")
        return

    # Convertimos a escala de grises (necesario para poder aplicar el umbral)
    img_byn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suavizamos la imagen para eliminar pequeñas manchas de suciedad o ruido
    # Usamos un kernel de (5, 5) o (10, 10) dependiendo de la resolución de la foto
    #img_suavizada = cv2.blur(img_byn, (10, 10))
    
    # Definimos el valor de corte para separar la tinta del papel
    umbral_valor = 200
    
    # Aplicamos el umbral invertido (THRESH_BINARY_INV)
    # _, guarda el valor del umbral devuelto (que no usamos) y img_umbral guarda la imagen final
    #_, img_umbral = cv2.threshold(img_suavizada, umbral_valor, 255, cv2.THRESH_BINARY_INV)
    _, img_umbral = cv2.threshold(img_byn, umbral_valor, 255, cv2.THRESH_BINARY_INV)
    
    return img, img_umbral

"""
Aislar las 5 líneas del pentagrama.
Usa operaciones morfológicas con un molde rectangular horizontal para borrar
todo lo que no sea una línea continua (elimina cabezas de notas, plicas y texto).
"""
def aislarLineas(img_umbral):
    # Creamos el "molde" (kernel) horizontal. 
    # (80, 1) significa 80 píxeles de ancho por 1 píxel de alto.
    # Cualquier figura blanca más estrecha de 80 píxeles (como las notas) desaparecerá.
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    
    # Aplicamos la erosión: pasamos el molde para "desgastar" la imagen
    img_erosionada = cv2.erode(img_umbral, kernel_horizontal, iterations=1)
    
    # Aplicamos la dilatación: pasamos el molde sobre el resultado para devolver
    # el grosor original a las líneas horizontales que han sobrevivido
    img_dilatada = cv2.dilate(img_erosionada, kernel_horizontal, iterations=1)
    
    return img_dilatada

"""
EXTRAER ALTURAS DEL PENTAGRAMA
Detecta las coordenadas Y de las lineas del pentagrama.
Elimina detecciones duplicadas y calcula la distancia interlinea media (S). 
"""
def obtenerAlturasLineas(img_lineas):
    contornos_lineas, _ = cv2.findContours(img_lineas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    alturas_y_crudas = []
    ancho_minimo = int(img_lineas.shape[1] * 0.30)  # La linea debe medir al menos el 30% de la imagen
    
    for contorno in contornos_lineas:
        x, y, w, h = cv2.boundingRect(contorno)
        if w > ancho_minimo:
            centro_y = y + (h // 2)
            alturas_y_crudas.append(centro_y)
            
    alturas_y_crudas.sort()
    
    # Limpieza de lineas duplicadas 
    alturas_y = []
    for y_actual in alturas_y_crudas:
        if len(alturas_y) == 0:
            alturas_y.append(y_actual)
        else:
            diferencia = y_actual - alturas_y[-1]
            if diferencia > 5:
                alturas_y.append(y_actual)
                
    # Cálculo de la distancia media entre lineas consecutivas (S)
    distancias = []
    for i in range(len(alturas_y) - 1):
        dist = alturas_y[i + 1] - alturas_y[i]
        # Solo consideramos distancias que correspondan al interior de un pentagrama
        # (descartamos el hueco grande que hay entre un pentagrama y otro)
        if 8 < dist < 45:
            distancias.append(dist)
            
    if len(distancias) > 0:
        S = float(np.median(distancias))
    else:
        S = 20.0  # Valor de respaldo por seguridad
        
    return alturas_y, S


"""
Aislar y detectar las notas musicales.
Utiliza operaciones morfológicas con un kernel elíptico para borrar las líneas
y dejar solo las cabezas de las notas. Luego busca los contornos y filtra por área.
"""
def detectarNotas(img_umbral, S, alturas_pentagrama):
    img_trabajo = img_umbral.copy()

    # RELLENO INTELIGENTE DE BLANCAS
    # RETR_CCOMP organiza contornos en dos niveles: exteriores y agujeros interiores
    contornos_todos, jerarquia = cv2.findContours(img_trabajo, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if jerarquia is not None:
        for i in range(len(contornos_todos)):
            # Si el contorno tiene un padre en la jerarquia, significa que es un agujero interior
            indice_padre = jerarquia[0][i][3]
            if indice_padre != -1:
                area_hueco = cv2.contourArea(contornos_todos[i])
                # El agujero de una blanca ocupa entre el 5% y el 70% del cuadrado de la interlinea
                area_min_hueco = 0.05 * (S ** 2)
                area_max_hueco = 0.70 * (S ** 2)
                
                if area_min_hueco < area_hueco < area_max_hueco:
                    # Rellenamos de blanco el interior de la nota
                    cv2.drawContours(img_trabajo, contornos_todos, i, 255, thickness=cv2.FILLED)

    # ELIMINACION DE PLICAS Y BARRAS BASADA EN S
    # El grosor del kernel debe ser un tercio de S para cortar el palo sin romper la cabeza
    tamano_kernel = int(round(0.35 * S))
    if tamano_kernel % 2 == 0:
        tamano_kernel = tamano_kernel + 1
    if tamano_kernel < 3:
        tamano_kernel = 3
        
    kernel_corte = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_kernel, tamano_kernel))
    
    # 1 iteracion con el kernel ajustado a la escala es suficiente
    img_erosion = cv2.erode(img_trabajo, kernel_corte, iterations=1)
    img_notas_aisladas = cv2.dilate(img_erosion, kernel_corte, iterations=1)

    # FILTRADO GEOMETRICO ADAPTATIVO
    contornos_candidatos, _ = cv2.findContours(img_notas_aisladas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notas_validas = []
    
    # Rangos de area proporcionales a S^2
    area_minima_nota = 0.25 * (S ** 2)
    area_maxima_nota = 1.80 * (S ** 2)
    margen_clave_x = int(3.0 * S)  # La zona de la clave y compas ocupa los primeros 3 espacios
    
    for contorno in contornos_candidatos:  
        area = abs(cv2.contourArea(contorno))
        x, y, w, h = cv2.boundingRect(contorno)
        
        proporcion = w / float(h)
        area_caja = w * h
        if area_caja > 0:
            solidez = area / float(area_caja)
        else:
            solidez = 0.0

        # Filtro de forma
        es_tamano_correcto = area_minima_nota < area < area_maxima_nota
        es_redondeada = 0.70 <= proporcion <= 1.65
        es_solida = solidez > 0.55
        esta_fuera_de_la_clave = x > margen_clave_x
        
        # Filtro vertical: la nota debe estar cerca de algun pentagrama conocido
        centro_y = y + (h // 2)
        esta_en_rango_pentagrama = False
        for i in range(0, len(alturas_pentagrama), 5):
            bloque = alturas_pentagrama[i:i+5]
            if len(bloque) == 5:
                tope_superior = bloque[0] - int(1.5 * S)
                tope_inferior = bloque[4] + int(1.5 * S)
                if tope_superior <= centro_y <= tope_inferior:
                    esta_en_rango_pentagrama = True

        if es_tamano_correcto and es_redondeada and es_solida and esta_fuera_de_la_clave and esta_en_rango_pentagrama:
            notas_validas.append(contorno)
            
    return img_notas_aisladas, notas_validas

"""
Ubicar la nota espacialmente.
Obtiene las coordenadas (x, y, w, h) y el centro (cx, cy) de cada nota válida
usando cajas delimitadoras (bounding boxes). Ordena las notas de izquierda a derecha.
"""
def ubicarNotas(img_original, contornos_notas):
    img_resultado = img_original.copy()
    datos_notas = []
    
    for contorno in contornos_notas:
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Calculamos los momentos para extraer el centro de masa físico real
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + (w // 2)
            cy = y + (h // 2)

        # Dibujamos rectángulo y centroide
        cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_resultado, (cx, cy), 3, (0, 0, 255), -1)
        
        # Guardamos 'h' para usarlo de forma dinámica en el radio
        datos_notas.append({'cx': cx, 'cy': cy, 'x': x, 'y': y, 'w': w, 'h': h})
        
    datos_notas.sort(key=lambda nota: nota['cx'])
    return img_resultado, datos_notas

"""
Obtiene las líneas dependiendo de qué pentagrama necesite.
"""
def obtenerLineasCercanas(cy, alturas_pentagrama):
    # Agrupamos las líneas detectadas de 5 en 5 
    pentagramas = []

    for i in range(0, len(alturas_pentagrama), 5):
        # Extrae bloques de 5 elementos en cada iteración
        grupo = alturas_pentagrama[i:i+5]
        pentagramas.append(grupo)
    
    # Buscamos en qué pentagrama encaja la nota basándonos en su altura Y (cy)
    for pentagrama in pentagramas:
        linea_superior = pentagrama[0]
        linea_inferior = pentagrama[4]
        
        # Damos un margen de 50 píxeles por arriba y por abajo de las líneas
        if (linea_superior - 50) <= cy <= (linea_inferior + 50):
            return pentagrama
            
    # Si por algún motivo la nota está muy fuera de rango, devolvemos el primer pentagrama por seguridad
    return pentagramas[0]

"""
IDENTIFICAR NOTA MUSICAL (Por Proximidad Geométrica Directa)
Compara la coordenada 'cy' de la nota con las posiciones teóricas exactas 
de cada línea y espacio, seleccionando el escalón más cercano.
"""
def identificarNota(cy, lineas_pentagrama):
    L0, L1, L2, L3, L4 = lineas_pentagrama
    
    distancia_media = (L4 - L0) / 4.0
    paso = distancia_media / 2.0
    
    # Mapa de posiciones Y teóricas absolutas
    posiciones_y = {
        -2: L4 + distancia_media,   # DO (línea adicional)
        -1: L4 + paso,              # RE
         0: L4,                     # MI (1ª línea)
         1: (L4 + L3) / 2.0,        # FA
         2: L3,                     # SOL (2ª línea)
         3: (L3 + L2) / 2.0,        # LA
         4: L2,                     # SI (3ª línea)
         5: (L2 + L1) / 2.0,        # DO5
         6: L1,                     # RE5 (4ª línea)
         7: (L1 + L0) / 2.0,        # MI5
         8: L0,                     # FA5 (5ª línea)
         9: L0 - paso               # SOL5
    }
    
    # Asignación por mínima distancia euclídea pura
    escalon_mas_cercano = min(posiciones_y.keys(), key=lambda e: abs(posiciones_y[e] - cy))
    
    diccionario_notas = {
        -2: "DO", -1: "RE", 0: "MI", 1: "FA",
        2: "SOL", 3: "LA", 4: "SI", 5: "DO5", 
        6: "RE5", 7: "MI5", 8: "FA5", 9: "SOL5"
    }
    
    return diccionario_notas.get(escalon_mas_cercano, "?")

"""
Colorear notas y transparencia.
Crea una capa superpuesta donde pinta círculos rellenos sobre las notas,
y luego la fusiona con la imagen original usando cv2.addWeighted.
"""
def colorearNotas(img_original, datos_notas, alturas_pentagrama):
    capa_color = img_original.copy()
    
    # Diccionario de colores (BGR) para cada nota
    colores = {
        "DO": (0, 0, 255), "RE": (0, 128, 255), 
        "MI": (0, 140, 200),  # Naranja oscuro / Ocre
        "FA": (0, 150, 0),    # Verde bosque
        "SOL": (200, 150, 0), # Azul acero
        "LA": (255, 0, 0), "SI": (255, 0, 255), 
        "DO5": (128, 0, 255), "RE5": (128, 0, 128), 
        "MI5": (0, 140, 200), "FA5": (0, 150, 0), "SOL5": (200, 150, 0)
    }
    
    for nota in datos_notas:
        cx, cy, h = nota['cx'], nota['cy'], nota['h']
        radio = max(3, h // 2)
        
        # Pedimos dinámicamente las 5 líneas que corresponden a la altura de esta nota
        lineas = obtenerLineasCercanas(cy, alturas_pentagrama)
            
        nombre_nota = identificarNota(cy, lineas)

        print(f"Nota en Y={cy} -> Escalon matemático -> Nombre devuelto: {nombre_nota}")

        color_nota = colores.get(nombre_nota, (255,255,255))
        
        # Pintamos el círculo relleno en la capa de color
        cv2.circle(capa_color, (int(cx), int(cy)), radio, color_nota, -1)
        
        # Pintamos el texto con el nombre de la nota encima de la original para que se lea nítido
        cv2.putText(img_original, nombre_nota, (int(cx) - 15, int(cy) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_nota, 1)
        
    # Fusionamos
    img_fusionada = cv2.addWeighted(capa_color, 0.5, img_original, 0.5, 0)
    return img_fusionada

"""
Ordenación humana y reproducción de audio con BPM.
Genera un pitido basado en frecuencias calculando la duración mediante el Tempo.
"""
def reproducirMelodia(datos_notas, alturas_pentagrama):
    # Agrupamos las líneas en pentagramas (de 5 en 5)
    pentagramas = [alturas_pentagrama[i:i+5] for i in range(0, len(alturas_pentagrama), 5)]
    
    # Creamos una lista vacía para cada pentagrama detectado
    notas_por_pentagrama = {}
    for i in range(len(pentagramas)):
        notas_por_pentagrama[i] = []
    
    # Metemos cada nota en el saco de su propio pentagrama
    for nota in datos_notas:
        cy = nota['cy']
        for i, pentagrama in enumerate(pentagramas):
            if len(pentagrama) >= 5: # Filtro de seguridad
                if (pentagrama[0] - 50) <= cy <= (pentagrama[4] + 50):
                    notas_por_pentagrama[i].append(nota)
                    break
                    
    # Ordenamos de izq a der DENTRO de cada pentagrama y luego los pegamos en orden
    melodia_final = []
    for i in range(len(pentagramas)):
        notas_del_bloque = notas_por_pentagrama[i]
        notas_del_bloque.sort(key=lambda n: n['cx'])
        melodia_final.extend(notas_del_bloque)
    
    BPM = 120
    duracion_pulso_seg = 60.0 / BPM
    sample_rate = 44100 # Calidad de CD
    
    frecuencias = {
        "DO": 261.63, "RE": 293.66, "MI": 329.63, 
        "FA": 349.23, "SOL": 392.00, "LA": 440.00, 
        "SI": 493.88, "DO5": 523.25, "RE5": 587.33, 
        "MI5": 659.25, "FA5": 698.46, "SOL5": 783.99
    }
    
    print("\n--- SINTETIZANDO MELODÍA A", BPM, "BPM ---")
    
    # Aquí iremos guardando los números de la onda de sonido
    audio_completo = []
    
    for nota in melodia_final:
        cy = nota['cy']
        # Pedimos dinámicamente las 5 líneas que corresponden a la altura de esta nota
        lineas = obtenerLineasCercanas(cy, alturas_pentagrama)
        nombre = identificarNota(cy, lineas)
        
        print("Procesando:", nombre)
        
        if nombre in frecuencias:
            freq = frecuencias[nombre]
            
            # Generamos el tiempo matemático para la nota (85% del pulso)
            t = np.linspace(0, duracion_pulso_seg * 0.85, int(sample_rate * duracion_pulso_seg * 0.85), False)
            # Generamos la onda senoidal pura
            onda_nota = np.sin(freq * 2 * np.pi * t)
            
            # Generamos el silencio para la articulación (15% del pulso)
            silencio = np.zeros(int(sample_rate * duracion_pulso_seg * 0.15))
            
            # Añadimos la nota y el silencio a la pista final
            audio_completo.extend(onda_nota)
            audio_completo.extend(silencio)
        else:
            # Si no hay nota, generamos silencio puro
            silencio_completo = np.zeros(int(sample_rate * duracion_pulso_seg))
            audio_completo.extend(silencio_completo)

    print("Creando archivo de audio y reproduciendo...")
    
    # Convertimos los datos matemáticos a formato de audio real de 16 bits
    audio_pcm = np.int16(np.array(audio_completo) * 32767)
    
    # Escribimos el archivo .wav en tu disco duro
    nombre_archivo = "resultado.wav"
    with wave.open(nombre_archivo, "w") as f:
        f.setnchannels(1)       # Mono
        f.setsampwidth(2)       # 2 bytes = 16 bits
        f.setframerate(sample_rate)
        f.writeframes(audio_pcm.tobytes())
        
    # Usamos winsound.PlaySound que SÍ usa el motor moderno de audio de Windows
    winsound.PlaySound(nombre_archivo, winsound.SND_FILENAME)

#----------------------------------------------------------------------------
if __name__ == "__main__":

    #ruta = '.\prueba\partituraCumpleFeliz.png'
    #ruta = '.\prueba\partituraEstrellita2.png'
    ruta = '.\prueba\partituraPinpon.jpg'

    # Ejecutamos Módulo 1
    img_original, img_umbral = intercambiarColores(ruta)
    
    # Solo continuamos si la imagen se cargó bien
    if img_original is not None:
        
        # Ejecutamos Módulo 2
        img_lineas = aislarLineas(img_umbral)
        
        # Ejecutamos la extracción de alturas y obtenemos S
        alturas_pentagrama, S = obtenerAlturasLineas(img_lineas)
        
        print("Líneas del pentagrama detectadas:", len(alturas_pentagrama))
        print("Alturas Y:", alturas_pentagrama)
        print("Distancia interlínea calculada (S):", S)

        # Mostramos la evolución del procesamiento en ventanas separadas
        cv2.imshow('1. Original', img_original)
        cv2.imshow('2. Binarizada (Tinta blanca)', img_umbral)
        cv2.imshow('3. Solo Lineas (Morfologia)', img_lineas)
        
        # Ejecutamos Módulo 4 pasando S y alturas_pentagrama
        img_notas_aisladas, contornos_notas = detectarNotas(img_umbral, S, alturas_pentagrama)
        
        # Imprimimos cuántas notas hemos encontrado
        print("Total de notas detectadas:", len(contornos_notas))
        
        # Visualizamos la imagen donde solo sobrevivieron las manchas elípticas
        cv2.imshow('4. Notas aisladas', img_notas_aisladas)
        
        # Ejecutamos Módulo 5
        img_con_cajas, informacion_notas = ubicarNotas(img_original, contornos_notas)
        
        # Visualizamos la imagen final con las cajas y los centros marcados
        cv2.imshow('5. Notas Ubicadas', img_con_cajas)

        # Ejecutamos Módulo 6 actualizado
        img_coloreada = colorearNotas(img_original, informacion_notas, alturas_pentagrama)

        # Visualizamos la imagen final con la máscara transparente
        cv2.imshow('6. Notas Coloreadas', img_coloreada)

        cv2.waitKey(1)
        # Llamamos al Módulo 7
        reproducirMelodia(informacion_notas, alturas_pentagrama)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
