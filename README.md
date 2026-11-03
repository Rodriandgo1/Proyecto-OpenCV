# Reconocimiento óptico de partituras (OMR) y síntesis musical con OpenCV

Sistema de visión por computador para la digitalización, análisis espacial, clasificación rítmica y reproducción acústica de partituras musicales a partir de imágenes estáticas.

> [!IMPORTANT]
> **Documentación completa:** el análisis técnico exhaustivo, la arquitectura por módulos, diagramas de flujo y la resolución de problemas técnicos se encuentran detallados en el archivo adjunto **`doc_proyecto.pdf`**.

---

## Descripción del proyecto

El objetivo de este proyecto es transformar una imagen de una partitura en una experiencia audiovisual interactiva. A través de un pipeline completo de procesamiento digital de imágenes, el sistema:

1. **Aísla y analiza las 5 líneas del pentagrama** para calibrar la escala canónica del documento ($S$ o distancia interlínea).
2. **Elimina el pentagrama y cicatriza las notas** para evitar deformaciones en su contorno.
3. **Diferencia figuras musicales por su morfología**: detecta cabezas de notas, identifica blancas examinando el núcleo de brillo original y agrupa corcheas mediante análisis de vigas conjuntas.
4. **Calcula la altura tonal geométrica** mediante ajuste de elipses subpíxel y radios de atracción gravitatoria sobre las líneas físicas.
5. **Renderiza una vista enriquecida** con efecto de resaltador fluorescente translúcido y nombres de notas.
6. **Sintetiza la melodía en formato WAV** con frecuencias matemáticas puras en hercios respetando el tempo (BPM) y la duración de cada figura.

---

## Funciones de OpenCV utilizadas

A lo largo de los diferentes módulos del proyecto se emplean las siguientes funciones de OpenCV (`cv2`):

| Función | Propósito en el proyecto |
| :--- | :--- |
| **`cv2.imread()`** | Carga la imagen de la partitura desde disco en formato BGR. |
| **`cv2.cvtColor()`** | Conversión de espacio de color BGR a escala de grises de canal único (`cv2.COLOR_BGR2GRAY`). |
| **`cv2.threshold()`** | Binarización e inversión con umbral adaptativo global de Otsu (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`). |
| **`cv2.getStructuringElement()`** | Fabricación de kernels o moldes morfológicos rectangulares (`cv2.MORPH_RECT`) y elípticos (`cv2.MORPH_ELLIPSE`) escalados dinámicamente con la interlínea $S$. |
| **`cv2.erode()`** | Desgaste morfológico para suprimir trazos finos, palos (plicas) y notas no horizontales. |
| **`cv2.dilate()`** | Expansión morfológica para devolver el grosor a líneas continuas. |
| **`cv2.subtract()`** | Resta algebraica píxel a píxel para sustraer la máscara del pentagrama de la partitura. |
| **`cv2.morphologyEx()`** | Aplicación de operaciones morfológicas compuestas: cierre (`cv2.MORPH_CLOSE`) para cicatrizar cortes verticales y apertura (`cv2.MORPH_OPEN`) para aislar cabezas elípticas. |
| **`cv2.findContours()`** | Detección topológica de contornos exteriores (`cv2.RETR_EXTERNAL`) y jerarquías de huecos interiores (`cv2.RETR_CCOMP`) con compresión simple (`cv2.CHAIN_APPROX_SIMPLE`). |
| **`cv2.contourArea()`** | Medición de superficie en píxeles para filtrar ruido, motas de polvo y símbolos ajenos según el área normalizada de una nota. |
| **`cv2.boundingRect()`** | Extracción de cajas envolventes `(x, y, w, h)` para cálculo de solidez, aspecto y pertenencia espacial a compases. |
| **`cv2.drawContours()`** | Relleno sólido (`cv2.FILLED`) de los huecos internos de las notas blancas. |
| **`cv2.fitEllipse()`** | Ajuste algebraico de elipses para fijar el centroide `(cx, cy)` con precisión subpíxel, inmune a restos de plicas. |
| **`cv2.rectangle()`** | Dibujado de cajas verdes de depuración alrededor de las notas validadas. |
| **`cv2.circle()`** | Dibujado de centros de masa y pastillas de color asignadas a cada tono musical. |
| **`cv2.putText()`** | Impresión nítida del nombre de la nota ("DO", "SOL", "MI") sobre la partitura. |
| **`cv2.addWeighted()`** | Mezcla alfa ponderada al 50% entre la imagen original y la capa de color para lograr transparencia translúcida. |
| **`cv2.imshow()`** / **`cv2.waitKey()`** | Despliegue de ventanas de diagnóstico en tiempo de ejecución. |

---

## Requisitos e instalación

### Prerrequisitos
* Python 3.8 o superior.
* Sistema Operativo: Windows (recomendado por el uso del módulo nativo `winsound`).

### Librerías requeridas
Instala las dependencias necesarias mediante `pip`:

```bash
pip install opencv-python numpy
```
*(Las librerías `winsound`, `time`, `math` y `wave` forman parte de la biblioteca estándar de Python).*

---

## Cómo ejecutar el proyecto

1. **Clonar o descargar el repositorio:**  
   Ubica el archivo principal `proyecto_Rodrigo_Rina_Mendo.py` y asegúrate de contar con una carpeta `partituras/` que contenga las imágenes a analizar.

2. **Seleccionar la partitura:**  
   Abre `proyecto_Rodrigo_Rina_Mendo.py` y en el bloque `if __name__ == "__main__":` descomenta la ruta de la partitura que deseas probar:
   ```python
   # ruta = './partituras/partituraEstrellita2.png'
   # ruta = './partituras/partituraPinpon.jpg'
   ruta = './partituras/cumple.png'
   ```

3. **Ejecutar el script:**  
   Lanza la ejecución desde la terminal:
   ```bash
   python proyecto_Rodrigo_Rina_Mendo.py
   ```

4. **Interacción:**  
   * Se abrirán ventanas con las etapas del pipeline (Líneas aisladas, notas cicatrizadas, notas ubicadas y partitura coloreada).
   * La consola imprimirá las métricas de escala ($S$ y $T$), la detección de notas y la partitura interpretada.
   * El sistema generará el archivo `resultado.wav` y reproducirá la melodía a través de los altavoces.
   * Pulsa cualquier tecla sobre las ventanas de OpenCV para cerrarlas.

---

## Más información

Para consultar el desglose pormenorizado de cada uno de los 14 módulos, sus diagramas de arquitectura, las problemáticas afrontadas (como la resolución de los problemas en notas blancas o la estabilización de centroides) y las líneas de trabajo futuro, consulta el documento:
    **`doc_proyecto.pdf`**

