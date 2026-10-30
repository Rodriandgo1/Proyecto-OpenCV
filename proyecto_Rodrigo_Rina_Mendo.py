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
Detecta las coordenadas Y de las lineas, elimina duplicados
y extrae las dos metricas fundamentales de escala:
S (distancia entre lineas) y T (grosor de cada linea).
"""
def obtenerAlturasLineas(img_lineas):
    contornos_lineas, _ = cv2.findContours(img_lineas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lineas_detectadas = []
    ancho_minimo = int(img_lineas.shape[1] * 0.30)
    
    for contorno in contornos_lineas:
        x, y, w, h = cv2.boundingRect(contorno)
        if w > ancho_minimo:
            centro_y = y + (h // 2)
            lineas_detectadas.append({'y': centro_y, 'x': x, 'h': h})
            
    # Ordenamos de arriba a abajo por coordenada Y
    lineas_detectadas.sort(key=lambda item: item['y'])
    
    # 1. Unificar lineas duplicadas
    lineas_limpias = []
    for linea in lineas_detectadas:
        if len(lineas_limpias) == 0:
            lineas_limpias.append(linea)
        else:
            diferencia = linea['y'] - lineas_limpias[-1]['y']
            if diferencia > 5:
                lineas_limpias.append(linea)
                
    alturas_y = [l['y'] for l in lineas_limpias]
    grosores_h = [l['h'] for l in lineas_limpias]
    
    # 2. Calculo de S (distancia interlinea)
    distancias = []
    for i in range(len(alturas_y) - 1):
        dist = alturas_y[i + 1] - alturas_y[i]
        if 8 < dist < 45:
            distancias.append(dist)
            
    if len(distancias) > 0:
        S = float(np.median(distancias))
    else:
        S = 20.0
        
    # 3. Calculo de T (grosor de linea)
    if len(grosores_h) > 0:
        T = int(round(np.median(grosores_h)))
    else:
        T = 2
    if T < 1:
        T = 1
        
    # 4. Calculo del x_inicio especifico para cada bloque de 5 lineas
    x_inicios_pentagramas = []
    for i in range(0, len(lineas_limpias), 5):
        bloque = lineas_limpias[i:i+5]
        if len(bloque) > 0:
            min_x_bloque = min([item['x'] for item in bloque])
            x_inicios_pentagramas.append(min_x_bloque)
        else:
            x_inicios_pentagramas.append(0)
            
    return alturas_y, S, T, x_inicios_pentagramas

"""
Borra las lineas del pentagrama mediante sustraccion y cicatriza 
los cortes horizontales de las notas usando morfologia vertical adaptada a T.
"""
def sustraerPentagramaYCicatrizar(img_umbral, img_lineas, T):
    # Restamos la mascara de lineas a la imagen umbralizada
    img_sin_lineas = cv2.subtract(img_umbral, img_lineas)
    
    # Creamos un kernel vertical de 1 pixel de ancho
    # La altura debe superar el grosor T para tender un puente sobre el corte
    alto_kernel = (2 * T) + 3
    if alto_kernel < 5:
        alto_kernel = 5
        
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, alto_kernel))
    
    # El Cierre Morfológico (Dilatacion vertical seguida de Erosion vertical)
    # Suelda la brecha horizontal de las cabezas sin recrear las lineas
    img_cicatrizada = cv2.morphologyEx(img_sin_lineas, cv2.MORPH_CLOSE, kernel_vertical)
    
    return img_sin_lineas, img_cicatrizada


"""
Fases 3 y 4 definitivas:
- Relleno topologico de blancas.
- Apertura morfologica para aislar notas.
- Filtro de relacion de aspecto y solidez (elimina vigas y barras finales).
- Margen de cabecera diferenciado (Pentagrama 1 vs Resto).
"""
def detectarNotas(img_cicatrizada, S, alturas_pentagrama, x_inicios_pentagramas):

    img_trabajo = img_cicatrizada.copy()

    # Relleno topologico de blancas
    contornos_todos, jerarquia = cv2.findContours(img_trabajo, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is not None:
        for i in range(len(contornos_todos)):
            indice_padre = jerarquia[0][i][3]
            if indice_padre != -1:
                area_hueco = cv2.contourArea(contornos_todos[i])
                area_min_hueco = 0.05 * (S ** 2)
                area_max_hueco = 0.85 * (S ** 2)
                if area_min_hueco < area_hueco < area_max_hueco:
                    cv2.drawContours(img_trabajo, contornos_todos, i, 255, thickness=cv2.FILLED)

    # Apertura morfologica: para escalas pequeñas (S <= 10.5), limitamos a 3x3
    # para no desintegrar notas reales de baja resolucion
    tamano_molde = int(round(0.42 * S))
    if tamano_molde % 2 == 0:
        tamano_molde = tamano_molde + 1
    if tamano_molde < 3:
        tamano_molde = 3
        
    molde_nota = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_molde, tamano_molde))
    img_notas_puras = cv2.morphologyEx(img_trabajo, cv2.MORPH_OPEN, molde_nota)

    # Filtrado geometrico
    contornos_candidatos, _ = cv2.findContours(img_notas_puras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    notas_validas = []
    
    # 0.35 * S^2 erradica la gota del silencio (30-35 px) pero conserva notas reales (40-90 px)
    area_minima = 0.34 * (S ** 2)
    area_maxima = 2.20 * (S ** 2)
    
    ancho_minimo_nota = int(0.55 * S)
    alto_minimo_nota = int(0.40 * S)
    
    for contorno in contornos_candidatos:
        area = abs(cv2.contourArea(contorno))
        x, y, w, h = cv2.boundingRect(contorno)
        
        proporcion = w / float(h)
        area_caja = w * h
        if area_caja > 0:
            solidez = area / float(area_caja)
        else:
            solidez = 0.0
            
        es_redondeada = 0.55 <= proporcion <= 1.85
        es_solida = solidez > 0.52
        tamano_valido = area_minima < area < area_maxima
        dimensiones_validas = (w >= ancho_minimo_nota) and (h >= alto_minimo_nota)
        
        centro_y = y + (h // 2)
        en_pentagrama = False
        esta_fuera_cabecera = False
        
        for i in range(0, len(alturas_pentagrama), 5):
            bloque = alturas_pentagrama[i:i+5]
            if len(bloque) == 5:
                limite_superior = bloque[0] - int(1.5 * S)
                limite_inferior = bloque[4] + int(1.5 * S)
                
                if limite_superior <= centro_y <= limite_inferior:
                    en_pentagrama = True
                    numero_pentagrama = i // 5
                    
                    if numero_pentagrama < len(x_inicios_pentagramas):
                        x_ini = x_inicios_pentagramas[numero_pentagrama]
                    else:
                        x_ini = 0
                        
                    if numero_pentagrama == 0:
                        margen_x = x_ini + int(5.2 * S)
                    else:
                        margen_x = x_ini + int(3.2 * S)
                        
                    if x > margen_x:
                        esta_fuera_cabecera = True

        if tamano_valido and dimensiones_validas and es_redondeada and es_solida and en_pentagrama and esta_fuera_cabecera:
            notas_validas.append(contorno)
            
    return img_notas_puras, notas_validas

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
        
        # fitEllipse requiere al menos 5 puntos en el contorno
        if len(contorno) >= 5:
            (centro_x, centro_y), (eje_w, eje_h), angulo = cv2.fitEllipse(contorno)
            cx = int(round(centro_x))
            cy = int(round(centro_y))
        else:
            cx = x + (w // 2)
            cy = y + (h // 2)

        cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_resultado, (cx, cy), 3, (0, 0, 255), -1)
        
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
    tolerancia_linea = 0.30 * distancia_media  # Radio de atraccion de cada linea (~3 px)
    
    # 1. Comprobacion prioritaria: ¿Cae la nota sobre alguna de las 5 lineas fisicas?
    # L4 = 1a linea (MI), L3 = 2a (SOL), L2 = 3a (SI), L1 = 4a (RE5), L0 = 5a (FA5)
    if abs(cy - L4) <= tolerancia_linea:
        return "MI"
    if abs(cy - L3) <= tolerancia_linea:
        return "SOL"
    if abs(cy - L2) <= tolerancia_linea:
        return "SI"
    if abs(cy - L1) <= tolerancia_linea:
        return "RE5"
    if abs(cy - L0) <= tolerancia_linea:
        return "FA5"
        
    # 2. Si no cae sobre una linea fisica, calculamos la distancia a todos los escalones
    posiciones_y = {
        -2: L4 + distancia_media,   # DO
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
    
    escalon_mas_cercano = min(posiciones_y.keys(), key=lambda e: abs(posiciones_y[e] - cy))
    
    diccionario_notas = {
        -2: "DO", -1: "RE", 0: "MI", 1: "FA",
        2: "SOL", 3: "LA", 4: "SI", 5: "DO5", 
        6: "RE5", 7: "MI5", 8: "FA5", 9: "SOL5"
    }
    
    nombre = diccionario_notas.get(escalon_mas_cercano, "?")
    return nombre

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

    ruta = '.\prueba\partituraCumpleFeliz.png'
    #ruta = '.\prueba\partituraEstrellita2.png'
    #ruta = '.\prueba\partituraPinpon.jpg'

    # Ejecutamos Módulo 1
    img_original, img_umbral = intercambiarColores(ruta)
    
    # Solo continuamos si la imagen se cargó bien
    if img_original is not None:
        
        # Aislar líneas
        img_lineas = aislarLineas(img_umbral)
        
        # FASE 1: Extraer alturas, escala S, grosor T e inicio X
        alturas_pentagrama, S, T, x_inicios_pentagramas = obtenerAlturasLineas(img_lineas)
        
        print("--- METRICAS DE ESCALA ---")
        print("Lineas detectadas:", len(alturas_pentagrama))
        print("Distancia interlinea (S):", S, "px")
        print("Grosor de linea (T):", T, "px")
        print("Inicio horizontal:", x_inicios_pentagramas, "px")

        # FASE 2: Sustraer pentagrama y cicatrizar cortes en notas
        img_rotas, img_cicatrizada = sustraerPentagramaYCicatrizar(img_umbral, img_lineas, T)

        # Ventanas de diagnostico de la Fase 1 y 2
        cv2.imshow('1. Original', img_original)
        cv2.imshow('2. Solo Lineas', img_lineas)
        cv2.imshow('3.1 Lineas Sustraidas (Notas Cortadas)', img_rotas)
        cv2.imshow('3.2 Notas Cicatrizadas (Sin Pentagrama)', img_cicatrizada)
        
        # De momento pasamos "img_cicatrizada" al detector de notas existente para ver el efecto
        img_notas_aisladas, contornos_notas = detectarNotas(img_cicatrizada, S, alturas_pentagrama, x_inicios_pentagramas)
        
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
        cv2.imshow('6. Notas coloreadas', img_coloreada)

        cv2.waitKey(1)

        # Llamamos al Módulo 7
        reproducirMelodia(informacion_notas, alturas_pentagrama)

        cv2.waitKey(0)
        cv2.destroyAllWindows()