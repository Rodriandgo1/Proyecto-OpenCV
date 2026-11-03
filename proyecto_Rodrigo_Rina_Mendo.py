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
    #umbral_valor = 200
    
    # Aplicamos el umbral invertido (THRESH_BINARY_INV)
    # _, guarda el valor del umbral devuelto (que no usamos) y img_umbral guarda la imagen final
    _, img_umbral = cv2.threshold(img_byn, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  
    #_, img_umbral = cv2.threshold(img_byn, 180, 255, cv2.THRESH_BINARY_INV)   
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
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    
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
    
    # Unificar lineas duplicadas
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
    
    # Cálculo de S (distancia interlínea)
    distancias = []
    for i in range(len(alturas_y) - 1):
        dist = alturas_y[i + 1] - alturas_y[i]
        if 8 < dist < 45:
            distancias.append(dist)
            
    if len(distancias) > 0:
        S = float(np.median(distancias))
    else:
        S = 20.0
        
    # Cálculo de T (grosor de línea)
    if len(grosores_h) > 0:
        T = int(round(np.median(grosores_h)))
    else:
        T = 2
    if T < 1:
        T = 1
        
    # Cálculo del x_inicio especifico para cada bloque de 5 lineas
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
def sustraerPentagramaYCicatrizar(img_umbral, img_lineas, T, S):    # Restamos la mascara de lineas a la imagen umbralizada
    img_sin_lineas = cv2.subtract(img_umbral, img_lineas)
    
    # El kernel mide un 40% de S (mínimo 5 px).
    # Suelda sobradamente los cortes de 3-4px en Cumpleaños (S=20) y en Estrellita (S=9)
    alto_kernel = int(round(0.40 * S))
    if alto_kernel < 5:
        alto_kernel = 5
        
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, alto_kernel))
    img_cicatrizada = cv2.morphologyEx(img_sin_lineas, cv2.MORPH_CLOSE, kernel_vertical)
    return img_sin_lineas, img_cicatrizada

"""
Localiza los huecos internos de las blancas,
guarda sus centros y los rellena para tratarlas como manchas sólidas.
"""
def rellenarBlancasYRegistrarHuecos(img_cicatrizada, S):

    img_rellena = img_cicatrizada.copy()
    centros_huecos = []
    
    contornos, jerarquia = cv2.findContours(img_rellena, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is not None:
        for i in range(len(contornos)):
            indice_padre = jerarquia[0][i][3]
            # Si tiene padre en la jerarquía, es un agujero interior
            if indice_padre != -1:
                area_hueco = cv2.contourArea(contornos[i])
                area_min = 2 
                area_max = 0.85 * (S ** 2)
                
                if area_min < area_hueco < area_max:
                    hx, hy, hw, hh = cv2.boundingRect(contornos[i])
                    centros_huecos.append((hx + (hw // 2), hy + (hh // 2)))
                    cv2.drawContours(img_rellena, contornos, i, 255, thickness=cv2.FILLED)
                    
    return img_rellena, centros_huecos

"""
Aplica apertura morfológica adaptada a S
para eliminar plicas y barras gruesas, dejando solo las cabezas.
"""
def aislarCabezasNotas(img_rellena, S):

    tamano_molde = int(round(0.42 * S))
    if tamano_molde % 2 == 0:
        tamano_molde = tamano_molde + 1
    if tamano_molde < 3:
        tamano_molde = 3
        
    molde = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano_molde, tamano_molde))
    img_cabezas = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, molde)
    return img_cabezas

"""
Valida si un contorno cumple las restricciones físicas y espaciales
para considerarse una cabeza de nota real.
"""
def esNotaValida(contorno, S, alturas_pentagrama, x_inicios_pentagramas):

    area = abs(cv2.contourArea(contorno))
    x, y, w, h = cv2.boundingRect(contorno)
    
    # Filtros de área y dimensiones
    if area <= (0.34 * (S ** 2)) or area >= (2.20 * (S ** 2)):
        return False
    if w < int(0.55 * S) or h < int(0.40 * S):
        return False
        
    # Filtros de proporción y solidez
    proporcion = w / float(h)
    if proporcion < 0.55 or proporcion > 1.85:
        return False
        
    area_caja = w * h
    if area_caja > 0:
        solidez = area / float(area_caja)
    else:
        solidez = 0.0
    if solidez <= 0.52:
        return False
        
    #Comprobación espacial: pertenencia a un pentagrama y fuera de la clave
    centro_y = y + (h // 2)
    for i in range(0, len(alturas_pentagrama), 5):
        bloque = alturas_pentagrama[i:i+5]
        if len(bloque) == 5:
            limite_sup = bloque[0] - int(1.5 * S)
            limite_inf = bloque[4] + int(1.5 * S)
            
            if limite_sup <= centro_y <= limite_inf:
                num_pentagrama = i // 5
                if num_pentagrama < len(x_inicios_pentagramas):
                    x_ini = x_inicios_pentagramas[num_pentagrama]
                else:
                    x_ini = 0
                    
                if num_pentagrama == 0:
                    margen_x = x_ini + int(5.2 * S)
                else:
                    margen_x = x_ini + int(3.2 * S)
                    
                if x > margen_x:
                    return True
                    
    return False

"""
Determina la duración y figura musical según la presencia de hueco o la conexión con vigas de corcheas.
"""
def clasificarRitmoNota(cx, cy, img_byn, contornos_con_palos, todas_las_notas, S):

    # ¿ES BLANCA? Inspección de brillo en el centro de la nota en la imagen original
    radio = max(2, int(round(0.20 * S)))
    alto_img, ancho_img = img_byn.shape
    y_min = max(0, cy - radio)
    y_max = min(alto_img, cy + radio + 1)
    x_min = max(0, cx - radio)
    x_max = min(ancho_img, cx + radio + 1)
    
    parche = img_byn[y_min:y_max, x_min:x_max]
    if parche.size > 0:
        # Una negra tiene tinta en todo el parche (media < 100).
        # Una blanca tiene papel blanco dentro (media > 130).
        if np.mean(parche) > 125:
            return 2.0, "BLANCA"

    # ¿ES CORCHEA? Comprobamos si comparte bloque ancho con otra nota
    for c_palo in contornos_con_palos:
        px, py, pw, ph = cv2.boundingRect(c_palo)
        if px <= cx <= (px + pw) and py <= cy <= (py + ph):
            notas_en_este_bloque = 0
            for otra in todas_las_notas:
                if px <= otra['cx'] <= (px + pw) and py <= otra['cy'] <= (py + ph):
                    notas_en_este_bloque = notas_en_este_bloque + 1
                    
            if notas_en_este_bloque >= 2 and pw >= int(1.5 * S):
                return 0.5, "CORCHEA"
                
    # Por defecto es una negra
    return 1.0, "NEGRA"

"""
Coordina el relleno, aislamiento morfológico, filtrado y clasificación rítmica de notas.
"""
def detectarNotas(img_cicatrizada, img_byn, S, alturas_pentagrama, x_inicios_pentagramas):
    # Relleno para que las blancas sean manchas sólidas para el contorno
    img_rellena, _ = rellenarBlancasYRegistrarHuecos(img_cicatrizada, S)

    # Aislamiento de manchas
    img_notas_puras = aislarCabezasNotas(img_rellena, S)

    # Detección de siluetas completas con plicas
    contornos_con_palos, _ = cv2.findContours(img_cicatrizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Evaluación de candidatos
    candidatos, _ = cv2.findContours(img_notas_puras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos_validos = []
    for c in candidatos:
        if esNotaValida(c, S, alturas_pentagrama, x_inicios_pentagramas):
            x, y, w, h = cv2.boundingRect(c)
            candidatos_validos.append({'contorno': c, 'cx': x + (w // 2), 'cy': y + (h // 2)})
            
    notas_validas = []
    for cand in candidatos_validos:
        duracion, tipo = clasificarRitmoNota(cand['cx'], cand['cy'], img_byn, contornos_con_palos, candidatos_validos, S)
        notas_validas.append({'contorno': cand['contorno'], 'duracion': duracion, 'tipo': tipo})
            
    return img_notas_puras, notas_validas

"""
Ubicar la nota espacialmente.
Obtiene las coordenadas (x, y, w, h) y el centro (cx, cy) de cada nota válida
usando cajas delimitadoras (bounding boxes). Ordena las notas de izquierda a derecha.
"""
def ubicarNotas(img_original, notas_detectadas):

    img_resultado = img_original.copy()
    datos_notas = []
    
    for item in notas_detectadas:
        contorno = item['contorno']
        duracion = item['duracion']
        tipo = item['tipo']
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Ajuste de elipse para fijar el centroide perfecto
        if len(contorno) >= 5:
            (centro_x, centro_y), (eje_w, eje_h), angulo = cv2.fitEllipse(contorno)
            cx = int(round(centro_x))
            cy = int(round(centro_y))
        else:
            cx = x + (w // 2)
            cy = y + (h // 2)

        # Dibujar rectángulos y centros
        cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_resultado, (cx, cy), 3, (0, 0, 255), -1)
        
        datos_notas.append({
            'cx': cx, 
            'cy': cy, 
            'x': x, 
            'y': y, 
            'w': w, 
            'h': h,
            'duracion': duracion,
            'tipo': tipo
        })
        
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
    tolerancia_linea = 0.20 * distancia_media  # Radio de atraccion de cada linea (~3 px)
    
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

        print("Nota en Y=", cy, "-> Escalon matemático -> Nombre devuelto:", nombre_nota)

        color_nota = colores.get(nombre_nota, (255,255,255))
        
        # Pintamos el círculo relleno en la capa de color
        cv2.circle(capa_color, (int(cx), int(cy)), radio, color_nota, -1)
        
        # Pintamos el texto con el nombre de la nota encima de la original para que se lea nítido
        cv2.putText(img_original, nombre_nota, (int(cx) - 15, int(cy) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_nota, 1)
        
    # Fusionamos
    img_fusionada = cv2.addWeighted(capa_color, 0.5, img_original, 0.5, 0)
    return img_fusionada

"""
Agrupa las notas por pentagrama, las ordena en secuencia temporal
y sintetiza el audio WAV respetando la duración exacta de cada figura.
"""
def reproducirMelodia(datos_notas, alturas_pentagrama):

    pentagramas = [alturas_pentagrama[i:i+5] for i in range(0, len(alturas_pentagrama), 5)]
    
    notas_por_pentagrama = {}
    for i in range(len(pentagramas)):
        notas_por_pentagrama[i] = []
    
    for nota in datos_notas:
        cy = nota['cy']
        for i, pentagrama in enumerate(pentagramas):
            if len(pentagrama) >= 5:
                if (pentagrama[0] - 50) <= cy <= (pentagrama[4] + 50):
                    notas_por_pentagrama[i].append(nota)
                    break
                    
    melodia_final = []
    for i in range(len(pentagramas)):
        notas_del_bloque = notas_por_pentagrama[i]
        notas_del_bloque.sort(key=lambda n: n['cx'])
        melodia_final.extend(notas_del_bloque)
    
    BPM = 120
    duracion_pulso_seg = 60.0 / BPM  # 1 tiempo = 0.5 segundos a 120 BPM
    sample_rate = 44100
    
    frecuencias = {
        "DO": 261.63, "RE": 293.66, "MI": 329.63, 
        "FA": 349.23, "SOL": 392.00, "LA": 440.00, 
        "SI": 493.88, "DO5": 523.25, "RE5": 587.33, 
        "MI5": 659.25, "FA5": 698.46, "SOL5": 783.99
    }
    
    print("\n--- SINTETIZANDO MELODÍA CON RITMO REAL A", BPM, "BPM ---")
    audio_completo = []
    
    for nota in melodia_final:
        cy = nota['cy']
        duracion_figura = nota['duracion']
        tipo_figura = nota['tipo']
        
        lineas = obtenerLineasCercanas(cy, alturas_pentagrama)
        nombre = identificarNota(cy, lineas)
        
        tiempo_total_nota = duracion_pulso_seg * duracion_figura
        print(f"Nota: {nombre:<5} | Figura: {tipo_figura:<7} | Duración: {duracion_figura} tiempos ({tiempo_total_nota:.2f}s)")
        
        if nombre in frecuencias:
            freq = frecuencias[nombre]
            tiempo_sonido = tiempo_total_nota * 0.85
            tiempo_silencio = tiempo_total_nota * 0.15
            
            t = np.linspace(0, tiempo_sonido, int(sample_rate * tiempo_sonido), False)
            onda_nota = np.sin(freq * 2 * np.pi * t)
            silencio = np.zeros(int(sample_rate * tiempo_silencio))
            
            audio_completo.extend(onda_nota)
            audio_completo.extend(silencio)
        else:
            silencio_completo = np.zeros(int(sample_rate * tiempo_total_nota))
            audio_completo.extend(silencio_completo)

    print("\nGenerando archivo y reproduciendo audio...")
    audio_pcm = np.int16(np.array(audio_completo) * 32767)
    
    nombre_archivo = "resultado.wav"
    with wave.open(nombre_archivo, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_pcm.tobytes())
        
    winsound.PlaySound(nombre_archivo, winsound.SND_FILENAME)

#----------------------------------------------------------------------------
if __name__ == "__main__":

    #ruta = '.\partituras\cumple.png'
    #ruta = '.\partituras\partituraEstrellita2.png'
    #ruta = '.\partituras\partituraPinpon.jpg'
    #ruta = '.\partituras\partituraMarchaImperial2.png'
    ruta = '.\partituras\Interstellar.png'

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
        img_rotas, img_cicatrizada = sustraerPentagramaYCicatrizar(img_umbral, img_lineas, T, S)

        # Ventanas de diagnostico de la Fase 1 y 2
        cv2.imshow('1. Original', img_original)
        cv2.imshow('2. Solo Lineas', img_lineas)
        cv2.imshow('3.1 Lineas Sustraidas (Notas Cortadas)', img_rotas)
        cv2.imshow('3.2 Notas Cicatrizadas (Sin Pentagrama)', img_cicatrizada)
        
        # De momento pasamos "img_cicatrizada" al detector de notas existente para ver el efecto
        img_byn = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        img_notas_aisladas, contornos_notas = detectarNotas(img_cicatrizada, img_byn, S, alturas_pentagrama, x_inicios_pentagramas)        
        
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