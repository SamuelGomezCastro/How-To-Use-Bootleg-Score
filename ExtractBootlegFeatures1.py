#!/usr/bin/env python3


# This code imports various functions needed to extract the bootleg features
# from an image of sheet music.
import sys

import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
import cv2
from skimage import filters, measure
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from scipy.signal import convolve2d
from scipy.spatial import KDTree
import seaborn as sns
import pickle
import librosa as lb
import time
import cProfile
import os
import os.path
import pyximport; pyximport.install()
import multiprocessing

#Esta función toma una imagen pimg y realiza un procesamiento para eliminar la iluminación de fondo. 
#Primero, redimensiona la imagen a un tamaño más pequeño (thumbnailW y thumbnailH) para acelerar el proceso. 
#Luego, calcula una versión difuminada de esta imagen redimensionada para estimar las sombras. Finalmente, 
#invierte y resta esta estimación de sombras a la imagen original.

def removeBkgdLighting(pimg, filtsz=5, thumbnailW = 100, thumbnailH = 100):
    tinyimg = pimg.copy()
    tinyimg.thumbnail([thumbnailW, thumbnailH]) # resize to speed up
    shadows = tinyimg.filter(ImageFilter.GaussianBlur(filtsz)).resize(pimg.size)
    result = ImageChops.invert(ImageChops.subtract(shadows, pimg))
    return result

#Esta función genera un filtro de peine penalizado. Crea una secuencia de "picos" 
#positivos y negativos a lo largo de la longitud del filtro (linesep * 5).

def getPenalizedCombFilter(linesep):
    filt = np.zeros(int(np.round(linesep * 5)))

    # positive spikes
    for i in range(5):
        offset = int(np.round(.5*linesep + i*linesep))
        filt[offset-1:offset+2] = 1.0

    # negative spikes
    for i in range(6):
        center = int(np.round(i*linesep))
        startIdx = max(center - 1, 0)
        endIdx = min(center + 2, len(filt))
        filt[startIdx:endIdx] = -1.0

    return filt

#Esta función estima la separación entre líneas en una imagen de partitura (pentagrama). Divide la imagen en columnas y 
#calcula las medianas de las filas para cada columna. Luego, aplica filtros de peine penalizado con 
#diferentes separaciones y encuentra la separación que produce la respuesta más fuerte.

def estimateLineSep(pim, ncols, lrange, urange, delta):

    # break image into columns, calculate row medians for inner columns (exclude outermost columns)
    
    img = 255 - np.array(pim) #Invierte los valores de los píxeles en la imagen pim 
                              #para trabajar con la imagen negativa.
    imgHeight, imgWidth = img.shape #Obtiene las dimensiones de la imagen (altura y ancho).
    rowMedians = np.zeros((imgHeight, ncols)) #Inicializa una matriz de ceros para almacenar las medianas de las filas.
    colWidth = imgWidth // (ncols + 2) #Calcula el ancho de cada columna después de excluir las dos columnas externas.
    for i in range(ncols): #Itera sobre el número de columnas y calcula las medianas de 
                           #las filas para las columnas internas.
        rowMedians[:,i] = np.median(img[:,(i+1)*colWidth:(i+2)*colWidth], axis=1)

    # apply comb filters
    
    lineseps = np.arange(lrange, urange, delta) #Genera un rango de valores para las separaciones de líneas (lineseps)
                                                #dentro del intervalo [lrange, urange) con un paso delta.
    responses = np.zeros((len(lineseps), imgHeight, ncols)) #Inicializa una matriz de ceros para almacenar 
                                                            #las respuestas de los filtros de peine.
    for i, linesep in enumerate(lineseps): #Itera sobre las separaciones de líneas y aplica filtros de peine penalizados 
                                           #a las medianas de las filas.
        filt = getPenalizedCombFilter(linesep).reshape((-1,1)) #Obtiene el filtro de peine penalizado para una separación
                                                               # de líneas específica  y lo remodela para que sea un vector columna.
        responses[i,:,:] = convolve2d(rowMedians, filt, mode = 'same') #Aplica la convolución 2D entre las medianas de las filas y 
                                                                       #el filtro de peine, almacenando las respuestas en la matriz responses.

    # find comb filter with strongest response
    
    scores = np.sum(np.max(responses, axis=1), axis=1) #Calcula las puntuaciones sumando los máximos de las respuestas para cada separación de líneas.
    bestIdx = np.argmax(scores) #Encuentra el índice de la separación de líneas con la respuesta más fuerte.
    estLineSep = lineseps[bestIdx] #Obtiene la estimación de la separación de líneas basada en el índice encontrado.

    return estLineSep, scores #Devuelve la estimación de la separación de líneas y las puntuaciones asociadas.

#Esta función calcula las dimensiones redimensionadas de una imagen (pim) en base a la separación 
#estimada de las líneas (estimatedLineSep) y la separación deseada (desiredLineSep).

def calcResizedDimensions(pim, estimatedLineSep, desiredLineSep):
    curH, curW = pim.height, pim.width
    scale_factor = 1.0 * desiredLineSep / estimatedLineSep
    targetH = int(curH * scale_factor)
    targetW = int(curW * scale_factor)
    return targetH, targetW

#Esta función normaliza una imagen, invirtiendo los valores de píxeles y dividiendo por 255 para obtener 
#valores en el rango [0, 1].

def getNormImage(img):
    X = 1 - np.array(img) / 255.0
    return X

#Esta función muestra una imagen en escala de grises. Permite especificar el tamaño de la figura (sz), 
#el valor máximo (maxval), y si la imagen debe invertirse (inverted).

def showGrayscaleImage(X, sz = (10,10), maxval = 1, inverted = True):
    # by default assumes X is a normalized image between 0 (white) and 1 (black)
    plt.figure(figsize = sz)
    if inverted:
        plt.imshow(maxval-X, cmap='gray')
    else:
        plt.imshow(X, cmap='gray')
        
#Esta función utiliza un filtro morfológico rectangular para realizar una combinación de erosión y 
#dilatación en la matriz bidimensional de entrada (arr). Este tipo de operaciones morfológicas son 
#comunes en el procesamiento de imágenes y se utilizan para realizar tareas como eliminación de ruido, 
#detección de bordes, entre otros. En este caso específico, el filtro rectangular puede ser útil para 
#resaltar o suprimir ciertas características en la imagen, dependiendo de las dimensiones del kernel rectangular.

def morphFilterRectangle(arr, kernel_height, kernel_width):
    kernel = np.ones((kernel_height, kernel_width),np.uint8)
    result = cv2.erode(arr, kernel, iterations = 1)
    result = cv2.dilate(result, kernel, iterations = 1)
    return result

#Esta función, isolateStaffLines, se encarga de aislar las líneas del pentagrama en una imagen. 
#En resumen, la función utiliza operaciones morfológicas para aislar las líneas del pentagrama 
#en una imagen. Primero, se identifican las líneas horizontales y luego se eliminan las barras de 
#notas para obtener solo las líneas del pentagrama. 

def isolateStaffLines(arr, kernel_len, notebarfilt_len, notebar_removal):
    lines = morphFilterRectangle(arr, 1, kernel_len) # isolate horizontal lines
    notebarsOnly = morphFilterRectangle(lines, notebarfilt_len, 1) # isolate thick notebars
    #Resta las barras de notas (engrosadas) multiplicadas por un factor de eliminación de barras de notas,
    #de las líneas horizontales. Esto tiene el efecto de eliminar las barras de notas de las líneas.
    result = np.clip(lines - notebar_removal*notebarsOnly, 0, None) # subtract out notebars
    return result

#Esta función genera un filtro de peine donde hay "púas" en posiciones específicas basadas en la separación 
#de las líneas en una partitura. La longitud del filtro se determina multiplicando la separación de líneas 
#por 4, redondeando hacia arriba, y sumando 1. Luego, se asignan pesos a los índices del filtro para crear 
#las "púas" del peine. El filtro de peine resultante se utiliza posteriormente en el procesamiento de la 
#imagen de la partitura.

def getCombFilter(lineSep):
    # generate comb filter of specified length
    # e.g. if length is 44, then spikes at indices 0, 11, 22, 33, 44
    # e.g. if length is 43, then spikes at 0 [1.0], 10 [.25], 11 [.75], 21 [.5], 22 [.5], 32 [.75], 33 [.25], 43 [1.0]
    stavelen = int(np.ceil(4 * lineSep)) + 1
    combfilt = np.zeros(stavelen)
    for i in range(5):
        idx = i * lineSep
        idx_below = int(idx)
        idx_above = idx_below + 1
        remainder = idx - idx_below
        combfilt[idx_below] = 1 - remainder
        if idx_above < stavelen:
            combfilt[idx_above] = remainder
    return combfilt, stavelen

# La función computeStaveFeatureMap, que calcula un mapa de características para una imagen de partitura. 
# En resumen:
#1) Divide la imagen en ncols columnas y calcula las sumas de las filas para cada columna. Esto se almacena en rowSums.
#2) Genera un rango de separaciones de líneas (lineseps) y calcula el tamaño máximo del filtro de peine basado 
# en la separación máxima.
#3) Inicializa un mapa de características (featmap) y un arreglo de longitudes de pauta (stavelens).
#4) Itera sobre las separaciones de líneas, obtiene el filtro de peine para cada separación, y aplica convoluciones 
# a las sumas de filas usando estos filtros. Devuelve el mapa de características resultante, las longitudes de 
# pauta y el ancho de la columna. Este mapa de características se utiliza para analizar patrones en la imagen 
# de la partitura, específicamente para detectar pautas musicales.

def computeStaveFeatureMap(img, ncols, lrange, urange, delta):

    # break image into columns, calculate row medians
    imgHeight, imgWidth = img.shape
    rowSums = np.zeros((imgHeight, ncols))
    colWidth = int(np.ceil(imgWidth/ncols))
    for i in range(ncols):
        startCol = i * colWidth
        endCol = min((i+1)*colWidth, imgWidth)
        rowSums[:,i] = np.sum(img[:,startCol:endCol], axis=1)

    # apply comb filters
    lineseps = np.arange(lrange, urange, delta)
    maxFiltSize = int(np.ceil(4 * lineseps[-1])) + 1
    featmap = np.zeros((len(lineseps), imgHeight - maxFiltSize + 1, ncols))
    stavelens = np.zeros(len(lineseps), dtype=np.int)
    for i, linesep in enumerate(lineseps):
        filt, stavelen = getCombFilter(linesep)
        padded = np.zeros((maxFiltSize, 1))
        padded[0:len(filt),:] = filt.reshape((-1,1))
        featmap[i,:,:] = convolve2d(rowSums, np.flipud(np.fliplr(padded)), mode = 'valid')
        stavelens[i] = stavelen

    return featmap, stavelens, colWidth

# Este código define la función morphFilterCircle, que aplica una operación de morfología matemática a una
# imagen utilizando un elemento estructurante con forma de círculo. Retorna la imagen después de aplicar 
# la morfología con el elemento estructurante de reducción y, opcionalmente, la expansión. 
# La función se utiliza para filtrar o resaltar ciertas características de la imagen mediante 
# operaciones morfológicas con elementos estructurantes circulares, en este caso de notas musicales. 

def morphFilterCircle(pimg, sz_reduce = 5, sz_expand = 0):
    kernel_reduce = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz_reduce, sz_reduce))
    result = cv2.dilate(np.array(pimg), kernel_reduce, iterations = 1)
    if sz_expand > 0:
        kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz_expand, sz_expand))
        result = cv2.erode(result, kernel_expand, iterations = 1)
    return result

# Este código implementa la detección de "blobs" (manchas o regiones con características particulares) 
# en una imagen utilizando el módulo SimpleBlobDetector de OpenCV. 
# En resumen, la función detectNoteheadBlobs toma una imagen y busca blobs (por ejemplo, las cabezas de 
# las notas musicales) dentro de ciertos límites de área especificados por minarea y maxarea. Luego, 
# devuelve los keypoints encontrados y la imagen original con los keypoints dibujados para visualización.

def detectNoteheadBlobs(img, minarea, maxarea):

    # define blob detector
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 100;
    # params.maxThreshold = 200;

    # Filter by Area
    # params.filterByArea = True
    params.minArea = minarea
    params.maxArea = maxarea

    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(np.array(img), keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return keypoints, im_with_keypoints

# esta función facilita la visualización de imágenes a color. Puede llamarse con una imagen a color 
# (representada como una matriz NumPy) y opcionalmente especificar el tamaño de la figura en la que 
# se mostrará la imagen.

def showColorImage(X, sz = (10,10)):
    plt.figure(figsize = sz)
    plt.imshow(X)

# Este código define una función llamada getNoteTemplate que genera una plantilla (template) para notas 
# musicales a partir de una imagen y un conjunto de puntos clave (keypoints). 

def getNoteTemplate(arr, keypoints, sz=21):
    template = np.zeros((sz, sz))  # Inicializa una matriz de ceros para la plantilla.
    L = (sz - 1) // 2  # Calcula la mitad del tamaño de la plantilla.
    numCrops = 0  # Inicializa el contador de recortes válidos.
    
    # Itera sobre los keypoints (puntos clave) proporcionados.
    for k in keypoints:
        xloc = int(np.round(k.pt[0]))  # Obtiene la ubicación x del punto clave redondeada.
        yloc = int(np.round(k.pt[1]))  # Obtiene la ubicación y del punto clave redondeada.
        
        # Verifica que el recorte no se salga de los límites de la imagen.
        if xloc - L >= 0 and xloc + L + 1 <= arr.shape[1] and yloc - L >= 0 and yloc + L + 1 <= arr.shape[0]:
            crop = arr[yloc - L:yloc + L + 1, xloc - L:xloc + L + 1]  # Recorta la región alrededor del punto clave.
            template += crop  # Suma el recorte a la plantilla.
            numCrops += 1  # Incrementa el contador de recortes válidos.
    
    if numCrops > 0:
        template = template / numCrops  # Normaliza la plantilla dividiendo por el número de recortes válidos.
    
    return template, numCrops  # Devuelve la plantilla normalizada y el número total de recortes válidos.

# Este código define una función llamada adaptiveNoteheadDetect que realiza la detección adaptativa 
# de cabezas de notas en una imagen. La función utiliza la binarización de Otsu para convertir la 
# imagen de entrada en una imagen binarizada. Luego, etiqueta las regiones conectadas en la imagen 
# binarizada y examina cada región para determinar si representa una cabeza de nota individual o 
# un bloque de acordes. Las cabezas de notas detectadas se almacenan en una lista llamada notes, y 
# la función devuelve esta lista junto con la imagen binarizada.

def adaptiveNoteheadDetect(arr, template, noteTolRatio, chordBlockSpecs):
    # Binariza la imagen utilizando el método de Otsu.
    binarized, _ = binarize_otsu(arr)
    
    # Etiqueta las regiones conectadas en la imagen binarizada.
    labels = measure.label(binarized)
    
    notes = []  # Inicializa una lista para almacenar las coordenadas de las cabezas de las notas detectadas.
    
    if template.max() == 0:  # Si no se detectan cabezas de notas en la plantilla, devuelve una lista vacía.
        return notes, binarized
    
    templateSpecs = getNoteTemplateSpecs(template)  # Obtiene las especificaciones de la plantilla de la cabeza de la nota.
    
    # Itera sobre las regiones etiquetadas en la imagen binarizada.
    for region in regionprops(labels):
        # Verifica si la región es una cabeza de nota válida.
        if isValidNotehead(region, noteTolRatio, templateSpecs):
            notes.append(region.bbox)  # Agrega las coordenadas de la cabeza de la nota a la lista.
        # Verifica si la región es un bloque de acordes válido.
        elif isValidChordBlock(region, chordBlockSpecs, templateSpecs):
            # Extrae las cabezas de notas individuales de un bloque de acordes.
            chordNotes = extractNotesFromChordBlock(region, templateSpecs)
            notes.extend(chordNotes)  # Agrega las coordenadas de las cabezas de notas del bloque de acordes a la lista.
    
    return notes, binarized  # Devuelve la lista de coordenadas de las cabezas de notas detectadas y la imagen binarizada.

# En resumen, el código utiliza el método de Otsu para determinar un umbral óptimo para binarizar la 
# imagen. Después de calcular el umbral, la imagen se binariza, y los píxeles se establecen en 
# True si su intensidad es mayor que el umbral y en False en caso contrario. La función devuelve la 
# imagen binarizada y el umbral calculado.

def binarize_otsu(img):
    arr = np.array(img)
    thresh = filters.threshold_otsu(arr)
    binarized = arr > thresh
    return binarized, thresh

# Este código calcula las especificaciones de un template de nota musical, como la altura máxima (maxH), 
# el ancho máximo (maxW), y el área máxima (maxArea). 

def getNoteTemplateSpecs(template):
    # Binariza el template utilizando el umbral de Otsu.
    _, thresh = binarize_otsu(template)
    binarized = template > thresh
    
    # Etiqueta las regiones conectadas en la imagen binarizada.
    labels = measure.label(binarized)
    
    # Inicializa variables para las dimensiones máximas y el área máxima.
    maxH, maxW, maxArea = (0, 0, 0)
    
    # Itera sobre las regiones etiquetadas.
    for region in regionprops(labels):
        # Calcula las dimensiones y el área de la región actual.
        curH = region.bbox[2] - region.bbox[0]
        curW = region.bbox[3] - region.bbox[1]
        curArea = region.area
        
        # Actualiza las dimensiones y el área máxima si la región actual tiene un área mayor.
        if curArea > maxArea:
            maxArea = curArea
            maxH = curH
            maxW = curW
    
    # Devuelve las dimensiones y el área máxima del template de nota.
    return (maxH, maxW, maxArea)

# Este bloque de código verifica si una región dada (supuestamente representando la cabeza 
# de una nota musical) es válida según ciertos criterios proporcionados.
# la función compara las dimensiones y el área de la región de la cabeza de la nota musical 
# con las especificaciones del template de nota y verifica si cumplen con ciertos límites 
# superiores e inferiores en términos de proporciones y áreas. Si todas las condiciones se 
# cumplen, la región se considera válida como la cabeza de una nota musical.

def isValidNotehead(region, tol_ratio, templateSpecs):
    templateH, templateW, templateArea = templateSpecs
    max_ratio = 1 + tol_ratio
    min_ratio = 1 / (1 + tol_ratio)
    curH = region.bbox[2] - region.bbox[0]
    curW = region.bbox[3] - region.bbox[1]
    curArea = region.area
    curRatio = 1.0 * curH / curW
    templateRatio = 1.0 * templateH / templateW
    validH = curH < templateH * max_ratio and curH > templateH * min_ratio
    validW = curW < templateW * max_ratio and curW > templateW * min_ratio
    validArea = curArea < templateArea * max_ratio * max_ratio and curArea > templateArea * min_ratio * min_ratio
    validRatio = curRatio < templateRatio * max_ratio and curRatio > templateRatio * min_ratio
    result = validH and validW and validRatio and validArea
    return result

# Se utiliza para determinar si una región en una imagen es válida como un bloque de acordes. 
# Un bloque de acordes generalmente consiste en varias cabezas de notas agrupadas. La función 
# devuelve True si la región cumple con todos estos criterios y, por lo tanto, se considera 
# un bloque de acordes válido; de lo contrario, devuelve False.

def isValidChordBlock(region, params, templateSpecs):
    templateH, templateW, templateArea = templateSpecs
    minH, maxH, minW, maxW, minArea, maxArea, minNotes, maxNotes = params
    curH = region.bbox[2] - region.bbox[0]
    curW = region.bbox[3] - region.bbox[1]
    curArea = region.area
    curNotes = int(np.round(curArea / templateArea))
    validH = curH >= minH * templateH and curH <= maxH * templateH
    validW = curW >= minW * templateW and curW <= maxW * templateW
    validArea = curArea >= minArea * templateArea and curArea <= maxArea * templateArea
    validNotes = curNotes >= minNotes and curNotes <= maxNotes
    result = validH and validW and validArea and validNotes
    return result

# Se utiliza para estimar las posiciones de las cabezas de notas dentro de un bloque de acordes. 
# La función utiliza el algoritmo KMeans para agrupar las coordenadas de la región y determinar 
# los centros de los clusters como las ubicaciones estimadas de las cabezas de notas.
# En resumen, la función utiliza el algoritmo KMeans para estimar los centros de las cabezas de 
# notas dentro de un bloque de acordes y devuelve los cuadros delimitadores correspondientes a 
# esas posiciones estimadas.

def extractNotesFromChordBlock(region, templateSpecs):
    # use kmeans to estimate note centers
    templateH, templateW, templateArea = templateSpecs
    numNotes = int(np.round(region.area / templateArea))
    regionCoords = np.array(region.coords)
    kmeans = KMeans(n_clusters=numNotes, n_init = 1, random_state = 0).fit(regionCoords)
    bboxes = []
    for (r,c) in kmeans.cluster_centers_:
        rmin = int(np.round(r - templateH/2))
        rmax = int(np.round(r + templateH/2))
        cmin = int(np.round(c - templateW/2))
        cmax = int(np.round(c + templateW/2))
        bboxes.append((rmin, cmin, rmax, cmax))
    return bboxes

# En resumen, la función se utiliza para visualizar las regiones en una imagen al 
# dibujar rectángulos alrededor de cada región. Esto puede ser útil para verificar 
# la precisión de los resultados del procesamiento de imágenes o para inspeccionar 
# visualmente las regiones identificadas en la imagen.

def visualizeLabels(img, bboxes):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    for (minr, minc, maxr, maxc) in bboxes:
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

# La función toma una lista de cuadros delimitadores que representan cabezas de notas, calcula 
# las coordenadas centrales de cada cabeza de nota y proporciona estimaciones promedio del ancho 
# y la longitud de las cabezas de notas en esa lista. Estas estimaciones pueden ser útiles para 
# ajustar o normalizar las dimensiones de las cabezas de notas en un contexto específico.

def getNoteheadInfo(bboxes):
    nhlocs = [(.5*(bbox[0] + bbox[2]), .5*(bbox[1] + bbox[3])) for bbox in bboxes]
    nhlens = [(bbox[2] - bbox[0]) for bbox in bboxes]
    nhwidths = [(bbox[3] - bbox[1]) for bbox in bboxes]
    nhlen_est = int(np.ceil(np.mean(nhlens)))
    nhwidth_est = int(np.ceil(np.mean(nhwidths)))
    return nhlocs, nhlen_est, nhwidth_est

# La función utiliza el mapa de características para estimar la ubicación de las líneas del pentagrama 
# alrededor de las cabezas de las notas en una imagen de partitura. La longitud estimada del filtro 
# (sfiltlen) puede ser útil en el procesamiento posterior de la imagen.

def getEstStaffLineLocs(featmap, nhlocs, stavelens, colWidth, deltaRowMax, globalOffset = 0):
    preds = []
    if np.isscalar(globalOffset):
        globalOffset = [globalOffset] * len(nhlocs)
    for i, nhloc in enumerate(nhlocs):
        r = int(np.round(nhloc[0]))
        c = int(np.round(nhloc[1]))
        rupper = min(r + deltaRowMax + 1 + globalOffset[i], featmap.shape[1])
        rlower = max(r - deltaRowMax + globalOffset[i], 0)
        featmapIdx = c // colWidth
        regCurrent = np.squeeze(featmap[:, rlower:rupper, featmapIdx])
        mapidx, roffset = np.unravel_index(regCurrent.argmax(), regCurrent.shape)
        rstart = rlower + roffset
        rend = rstart + stavelens[mapidx] - 1
        preds.append((rstart, rend, c, r, mapidx))

    sfiltlen = int(np.round(np.median([stavelens[tup[4]] for tup in preds])))
    return preds, sfiltlen

# La función visualiza las líneas del pentagrama estimadas junto con las cabezas de las notas en la imagen de 
# partitura. Realiza las siguientes acciones:
# 1) Muestra la imagen de partitura en escala de grises utilizando la función showGrayscaleImage.
# 2) Extrae las coordenadas relevantes de las predicciones en preds (ubicaciones de las líneas superior e inferior 
# del pentagrama, columna y fila de la cabeza de nota).
# 3) Utiliza la función scatter de Matplotlib para visualizar puntos en la imagen:
# 4) Puntos rojos para las líneas superiores del pentagrama. Puntos azules para las líneas inferiores del pentagrama. 
# Puntos amarillos para las cabezas de las notas.

# Cada punto en el gráfico representa una estimación de la ubicación de una línea del pentagrama o 
# una cabeza de nota en la imagen de partitura. Esto proporciona una representación visual de las 
# predicciones generadas por el modelo o el algoritmo en el contexto de la imagen original.

def visualizeEstStaffLines(preds, arr):
    showGrayscaleImage(arr, (15,15))
    rows1 = np.array([pred[0] for pred in preds]) # top staff line
    rows2 = np.array([pred[1] for pred in preds]) # bottom staff line
    cols = np.array([pred[2] for pred in preds]) # nh col
    rows3 = np.array([pred[3] for pred in preds]) # nh row
    plt.scatter(cols, rows1, c = 'r', s = 3)
    plt.scatter(cols, rows2, c = 'b', s = 3)
    plt.scatter(cols, rows3, c = 'y', s = 3)

# En resumen, esta función intenta estimar las ubicaciones verticales de los centros de los 
# pentagramas en una imagen basándose en las ubicaciones estimadas de las líneas del pentagrama. 
# Utiliza el algoritmo de agrupamiento KMeans para encontrar los clusters más significativos en 
# estas ubicaciones.

def estimateStaffMidpoints(preds, clustersMin, clustersMax, threshold):
    r = np.array([.5*(tup[0] + tup[1]) for tup in preds]) # midpts of estimated stave locations
    models = []
    for numClusters in range(clustersMin, clustersMax + 1):
        kmeans = KMeans(n_clusters=numClusters, n_init=1, random_state = 0).fit(r.reshape(-1,1))
        sorted_list = np.array(sorted(np.squeeze(kmeans.cluster_centers_)))
        mindiff = np.min(sorted_list[1:] - sorted_list[0:-1])
        if numClusters > clustersMin and mindiff < threshold:
            break
        models.append(kmeans)
    staffMidpts = np.sort(np.squeeze(models[-1].cluster_centers_))
    return staffMidpts

# Este código parece ser una función de depuración (debugStaffMidpointClustering) que analiza el comportamiento 
# del algoritmo de agrupamiento KMeans para diferentes cantidades de clusters. La función toma como entrada 
# una lista de tuplas (preds) que representan las ubicaciones estimadas de las líneas del pentagrama y 
# realiza lo siguiente:

def debugStaffMidpointClustering(preds):
    r = np.array([.5*(tup[0] + tup[1]) for tup in preds]) # midpts of estimated stave locations
    inertias = []
    mindiffs = []
    clusterRange = np.arange(2,12)
    for numClusters in clusterRange:
        kmeans = KMeans(n_clusters=numClusters, n_init=1, random_state = 0).fit(r.reshape(-1,1))
        inertias.append(kmeans.inertia_)
        sorted_list = np.array(sorted(np.squeeze(kmeans.cluster_centers_)))
        diffs = sorted_list[1:] - sorted_list[0:-1]
        mindiffs.append(np.min(diffs))
    plt.subplot(211)
    plt.plot(clusterRange, np.log(inertias))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.subplot(212)
    plt.plot(clusterRange, mindiffs)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Min Centroid Separation')
    plt.axhline(60, color='r')

# esta función visualiza la distribución de los puntos medios de las ubicaciones estimadas de las 
# líneas del pentagrama junto con líneas verticales rojas en los centros especificados. Esto puede 
# ser útil para evaluar visualmente cómo se están agrupando los puntos medios y si los centros 
# están ubicados en posiciones relevantes en el eje x.

def visualizeStaffMidpointClustering(preds, centers):
    r = np.array([.5*(tup[0] + tup[1]) for tup in preds]) # midpts of estimated stave locations
    plt.plot(r, np.random.uniform(size = len(r)), '.')
    for center in centers:
        plt.axvline(x=center, color='r')

# Este código asigna ubicaciones de cabezas de notas (nhlocs) a las líneas de pentagrama estimadas 
# (los centros de las líneas de pentagrama proporcionados en staveCenters). En resumen, este código 
# asigna cada cabeza de nota a la línea del pentagrama más cercana y calcula la diferencia entre la 
# ubicación de la cabeza de nota y el centro de la línea del pentagrama asignada.

def assignNoteheadsToStaves(nhlocs, staveCenters):
    nhrows = np.matlib.repmat([tup[0] for tup in nhlocs], len(staveCenters), 1)
    centers = np.matlib.repmat(staveCenters.reshape((-1,1)), 1, len(nhlocs))
    staveIdxs = np.argmin(np.abs(nhrows - centers), axis=0)
    offsets = staveCenters[staveIdxs] - nhrows[0,:] # row offset between note and staff midpoint
    return staveIdxs, offsets

def visualizeClusters(arr, nhlocs, clusters):
    showGrayscaleImage(arr)
    rows = np.array([tup[0] for tup in nhlocs])
    cols = np.array([tup[1] for tup in nhlocs])
    plt.scatter(cols, rows, c=clusters)
    for i in range(len(clusters)):
        plt.text(cols[i], rows[i] - 15, str(clusters[i]), fontsize = 12, color='red')

# Estas funciones se utilizan para estimar y visualizar las etiquetas de las notas en la partitura musical 
# basándose en las predicciones previas de las líneas de pentagrama.

def estimateNoteLabels(preds):
    nhvals = [] # estimated note labels
    for i, (rstart, rend, c, r, filtidx) in enumerate(preds):
        # if a stave has height L, there are 8 stave locations in (L-1) pixel rows
        staveMidpt = .5 * (rstart + rend)
        noteStaveLoc = -1.0 * (r - staveMidpt) * 8 / (rend - rstart)
        nhval = int(np.round(noteStaveLoc))
        nhvals.append(nhval)
    return nhvals

def visualizeNoteLabels(arr, vals, locs):
    showGrayscaleImage(arr)
    rows = np.array([loc[0] for loc in locs])
    cols = np.array([loc[1] for loc in locs])
    plt.scatter(cols, rows, color='blue')
    for i in range(len(rows)):
        plt.text(cols[i], rows[i] - 15, str(vals[i]), fontsize = 12, color='red')

# Esta función, isolateBarlines, se utiliza para aislar las líneas de barra en una imagen. 
# En resumen, esta función utiliza operaciones morfológicas y filtros para aislar las líneas 
# de barra en una imagen, eliminando líneas verticales no deseadas y minimizando falsos positivos.

def isolateBarlines(im, morphFilterVertLineLength, morphFilterVertLineWidth, maxBarlineWidth):
    hkernel = np.ones((1, morphFilterVertLineWidth), np.uint8) # dilate first to catch warped barlines
    vlines = cv2.dilate(im, hkernel, iterations = 1)
    vlines = morphFilterRectangle(vlines, morphFilterVertLineLength, 1) # then filter for tall vertical lines
    nonbarlines = morphFilterRectangle(vlines, 1, maxBarlineWidth)
    vlines = np.clip(vlines - nonbarlines, 0, 1)
    return vlines

# Este código determina la agrupación de los pentagramas (staves) en dos posibles grupos (grupo A y grupo B) 
# basándose en la evidencia de la presencia de líneas de barra en la imagen. 

def determineStaveGrouping(staveMidpts, vlines):

    N = len(staveMidpts)
    rowSums = np.sum(vlines, axis=1)

    # grouping A: 0-1, 2-3, 4-5, ...
    elems_A = []
    map_A = {}
    for i, staveIdx in enumerate(np.arange(0, N, 2)):
        if staveIdx+1 < N:
            startRow = int(staveMidpts[staveIdx])
            endRow = int(staveMidpts[staveIdx+1]) + 1
            elems_A.extend(rowSums[startRow:endRow])
            map_A[staveIdx] = staveIdx
            map_A[staveIdx+1] = staveIdx + 1
        else:
            map_A[staveIdx] = -1 # unpaired stave

    # grouping B: 1-2, 3-4, 5-6, ...
    elems_B = []
    map_B = {}
    map_B[0] = -1
    for i, staveIdx in enumerate(np.arange(1, N, 2)):
        if staveIdx+1 < N:
            startRow = int(staveMidpts[staveIdx])
            endRow = int(staveMidpts[staveIdx+1]) + 1
            elems_B.extend(rowSums[startRow:endRow])
            map_B[staveIdx] = staveIdx - 1
            map_B[staveIdx + 1] = staveIdx
        else:
            map_B[staveIdx] = -1

    if N > 2:
        evidence_A = np.median(elems_A)
        evidence_B = np.median(elems_B)
        if evidence_A > evidence_B:
            mapping = map_A
        else:
            mapping = map_B
    else:
        evidence_A = np.median(elems_A)
        evidence_B = 0
        mapping = map_A

    return mapping, (evidence_A, evidence_B, elems_A, elems_B)

# En resumen, el bloque de código permite visualizar las líneas de barra y las ubicaciones estimadas de los 
# centros de los pentagramas en la imagen para facilitar la depuración y comprensión del proceso de 
# agrupación de pautas.

def debugStaveGrouping(vlines, staveCenters):
    plt.plot(np.sum(vlines, axis=1))
    for m in staveCenters:
        plt.axvline(m, color = 'r')

# Este código se encarga de agrupar las cabezas de las notas (noteheads) en "clusters" o grupos 
# según su pertenencia a pentagramas específicos. En resumen, el código ayuda a organizar las 
# cabezas de notas en clusters específicos según su pertenencia a pentagramas, y proporciona 
# información sobre cómo están agrupados esos clusters.

def clusterNoteheads(staveIdxs, mapping):
    clusterIdxs = [mapping[staveIdx] for staveIdx in staveIdxs]
    maxClusterIdx = np.max(np.array(clusterIdxs))
    clusterPairs = []
    for i in range(0, maxClusterIdx, 2):
        clusterPairs.append((i,i+1))
    return clusterIdxs, clusterPairs

# Este código está diseñado para generar una línea de música en formato de partitura (bootleg score) 
# a partir de datos de cabezas de notas (noteheads) agrupadas en clusters correspondientes a dos 
# pentagramas específicos. En resumen, este código es parte de un sistema más amplio para convertir 
# datos de cabezas de notas en una representación de partitura musical.

def generateSingleBootlegLine(nhdata, clusterR, clusterL, minColDiff, repeatNotes = 1, filler = 1):
    notes = [tup for tup in nhdata if tup[3] == clusterR or tup[3] == clusterL]
    notes = sorted(notes, key = lambda tup: (tup[1], tup[0])) # sort by column, then row
    collapsed = collapseSimultaneousEvents(notes, minColDiff) # list of (rows, cols, vals, clusters)
    bscore, eventIndices, staffLinesBoth, _, _ = constructBootlegScore(collapsed, clusterR, clusterL, repeatNotes, filler)
    return bscore, collapsed, eventIndices, staffLinesBoth

# Este código está diseñado para colapsar eventos simultáneos en una secuencia de cabezas de notas (noteheads) 
# en música. La idea es agrupar las cabezas de notas que están muy cerca en la misma columna, asumiendo que 
# corresponden a notas que suenan simultáneamente. 

def collapseSimultaneousEvents(notes, minColDiff):
    assigned = np.zeros(len(notes), dtype=bool)
    events = [] # list of simultaneous note events
    for i, (row, col, val, cluster) in enumerate(notes):
        if assigned[i]: # has already been assigned
            continue
        rows = [row] # new event
        cols = [col]
        vals = [val]
        clusters = [cluster]
        assigned[i] = True
        for j in range(i+1, len(notes)):
            nrow, ncol, nval, ncluster = notes[j]
            if ncol - col < minColDiff: # assign to same event if close
                rows.append(nrow)
                cols.append(ncol)
                vals.append(nval)
                clusters.append(ncluster)
                assigned[j] = True
            else:
                break
        events.append((rows, cols, vals, clusters))

    assert(np.all(assigned))
    return events

#La función constructBootlegScore recibe varios parámetros relacionados con eventos de notas musicales y devuelve 
# un conjunto de datos organizado para representar una partitura musical.

def constructBootlegScore(noteEvents, clusterIndexRH, clusterIndexLH, repeatNotes = 1, filler = 1):
    # note that this has to match generateBootlegScore() in the previous notebook!
    rh_dim = 34 # E3 to C8 (inclusive)  # Dimensión de la mano derecha, establecida en 34.
    lh_dim = 28 # A1 to G4 (inclusive)  # Dimensión de la mano izquierda, establecida en 28.
    rh = [] # list of arrays of size rh_dim # Lista vacía que se utilizará para almacenar matrices de tamaño rh_dim.
    lh = [] # list of arrays of size lh_dim # Lista vacía que se utilizará para almacenar matrices de tamaño rh_dim.
    eventIndices = [] # index of corresponding simultaneous note event  # Lista vacía para almacenar el índice de eventos de notas simultáneos correspondientes.
    for i, (rows, cols, vals, clusters) in enumerate(noteEvents):

        # insert empty filler columns between note events
        if i > 0:
            for j in range(filler):
                rh.append(np.zeros((rh_dim,1)))
                lh.append(np.zeros((lh_dim,1)))
                eventIndices.append(i-1) # assign filler to previous event

        # insert note events columns
        rhvec, lhvec = getNoteheadPlacement(vals, clusters, rh_dim, lh_dim, clusterIndexRH, clusterIndexLH)
        for j in range(repeatNotes):
            rh.append(rhvec)
            lh.append(lhvec)
            eventIndices.append(i)
    rh = np.squeeze(np.array(rh)).reshape((-1, rh_dim)).T # reshape handles case when len(rh) == 1
    lh = np.squeeze(np.array(lh)).reshape((-1, lh_dim)).T
    both = np.vstack((lh, rh))
    staffLinesRH = [7,9,11,13,15]
    staffLinesLH = [13,15,17,19,21]
    staffLinesBoth = [13,15,17,19,21,35,37,39,41,43]
    return both, eventIndices, staffLinesBoth, (rh, staffLinesRH), (lh, staffLinesLH)

# la función getNoteheadPlacement, toma valores y clusters como entrada y devuelve vectores rhvec y lhvec que 
# indican la ubicación de las cabezas de las notas en las manos derecha e izquierda, respectivamente. 
# La posición de la cabeza de la nota se determina según el valor y el cluster especificado.

def getNoteheadPlacement(vals, clusters, rdim, ldim, clusterRH, clusterLH):
    rhvec = np.zeros((rdim, 1))
    lhvec = np.zeros((ldim, 1))
    assert(clusterLH == clusterRH + 1)
    for (val, cluster) in zip(vals, clusters):
        if cluster == clusterRH:
            idx = val + 11
            if idx >= 0 and idx < rdim:
                rhvec[idx, 0] = 1
        elif cluster == clusterLH:
            idx = val + 17
            if idx >= 0 and idx < ldim:
                lhvec[idx, 0] = 1
        else:
            print("Invalid cluster: {} (LH {}, RH {})".format(cluster, clusterLH, clusterRH))
            sys.exit(1)
    return rhvec, lhvec

# Se muestra una visualización de la partitura musical donde las áreas ocupadas o marcadas están en blanco y 
# las áreas vacías están en negro o más oscuras. Además, algunas líneas horizontales grises y líneas horizontales 
# rojas (especificadas por el parámetro lines) se agregan para resaltar ciertas líneas de pentagrama.
# En resumen, esta función es útil para visualizar la representación de una partitura musical en forma de matriz, con 
# la opción de resaltar líneas de pentagrama específicas en rojo.

def visualizeBootlegScore(bs, lines):
    plt.figure(figsize=(10,10))
    plt.imshow(1 - bs, cmap='gray', origin = 'lower')
    for l in range(1, bs.shape[0], 2):
        plt.axhline(l, c='grey')
    for l in lines:
        plt.axhline(l, c='r')
        
# Este código genera una representación visual de una partitura musical (panorama) a partir de datos de notas (nhdata) y 
# pares de clusters (pairings).

def generateImageBootlegScore(nhdata, pairings, repeatNotes = 1, filler = 1, minColDiff = 10):
    allScores = [] # Lista que almacenará las líneas de partitura generadas.
    allEvents = [] # Lista que almacenará los eventos generados.
    globIndices = [] # Lista que almacenará los índices globales de los eventos.
    eventCount = 0 # Contador de eventos acumulados.
    if len(pairings) == 0:
        return None, None, None, None
    for i, (clusterR, clusterL) in enumerate(pairings): #Itera sobre los pares de clusters en pairings.
        # Para cada par de clusters, llama a la función generateSingleBootlegLine para generar una línea de partitura, 
        # eventos y líneas de pentagrama.
        score, events, eventIndices, staffLinesBoth = generateSingleBootlegLine(nhdata, clusterR, clusterL, minColDiff, repeatNotes, filler)
        #Agrega la línea de partitura y los eventos generados a las listas correspondientes (allScores, allEvents, globIndices).
        allScores.append(score)
        allEvents.extend(events)
        globIndices.extend([idx + eventCount for idx in eventIndices])
        if filler > 0 and i < len(pairings) - 1:
            allScores.append(np.zeros((score.shape[0], filler))) # append filler columns between bootleg scores
            globIndices.extend([globIndices[-1]] * filler) # map filler columns to last event index
        eventCount += len(events)
    # Concatena todas las líneas de partitura generadas en un panorama.
    panorama = np.hstack(allScores)
    return panorama, allEvents, globIndices, staffLinesBoth

# En resumen, esta función divide la partitura musical extendida en fragmentos más pequeños y llama a la función 
# visualizeBootlegScore para visualizar cada fragmento por separado, posiblemente con el objetivo de facilitar la 
# visualización de partituras muy largas.

def visualizeLongBootlegScore(bs, lines, chunksz = 150):
    chunks = bs.shape[1] // chunksz + 1
    for i in range(chunks):
        startcol = i * chunksz
        endcol = min((i + 1)*chunksz, bs.shape[1])
        visualizeBootlegScore(bs[:,startcol:endcol], lines)

# Este código procesa una imagen de partitura musical y realiza una serie de operaciones para identificar y 
# generar una representación simplificada de la partitura, llamada "Bootleg Score". 

def processImageFile(imagefile, outfile):

    ### system parameters ###

    # Pre-processing
    thumbnailW = 100  # bkgd lighting
    thumbnailH = 100
    thumbnailFilterSize = 5
    estLineSep_NumCols = 3
    estLineSep_LowerRange = 12 # adjusted from 25
    estLineSep_UpperRange = 30 # adjusted from 45
    estLineSep_Delta = 1
    targetLineSep = 10.0

    # Staff Line Features
    morphFilterHorizLineSize = 41
    notebarFiltLen = 3
    notebarRemoval = 0.9
    calcStaveFeatureMap_NumCols = 10
    calcStaveFeatureMap_LowerRange = 8.5
    calcStaveFeatureMap_UpperRange = 11.75
    calcStaveFeatureMap_Delta = 0.25

    # Notehead Detection
    morphFilterCircleSizeReduce = 5
    morphFilterCircleSizeExpand = 5
    #morphFilterCircleSize = 5
    notedetect_minarea = 50
    notedetect_maxarea = 200
    noteTemplateSize = 21
    notedetect_tol_ratio = .4
    chordBlock_minH = 1.25
    chordBlock_maxH = 4.25
    chordBlock_minW = .8
    chordBlock_maxW = 2.25
    chordBlock_minArea = 1.8
    chordBlock_maxArea = 4.5
    chordBlock_minNotes = 2
    chordBlock_maxNotes = 4

    # Staffline Detection
    maxDeltaRowInitial = 50
    minNumStaves = 6 # adjusted from 2
    maxNumStaves = 16 # adjusted from 12
    minStaveSeparation = 6 * targetLineSep
    maxDeltaRowRefined = 15

    # Group Staves
    morphFilterVertLineLength = 101
    morphFilterVertLineWidth = 7
    maxBarlineWidth = 15
    #maxBarlineLenFactor = .25

    # Generate Bootleg Score
    bootlegRepeatNotes = 1
    bootlegFiller = 0

    ##########################

    print("Processing {}".format(imagefile))
    profileStart = time.time()

    # pre-processing
    try:
        pim1 = Image.open(imagefile).convert('L') # pim indicates PIL image object, im indicates raw pixel values
    except:
        if os.path.exists(imagefile):
            with open(outfile,'a') as f:
                print(imagefile+"-- cannot open file",file=f)
            return np.array([]).reshape(62,0)
        else:
            with open(outfile,'a') as f:
                print(imagefile+"-- imagefile not found",file=f)
            return np.array([]).reshape(62,0)
        return
    pim2 = removeBkgdLighting(pim1, thumbnailFilterSize, thumbnailW, thumbnailH)
    linesep, scores = estimateLineSep(pim2, estLineSep_NumCols, estLineSep_LowerRange, estLineSep_UpperRange, estLineSep_Delta)
    targetH, targetW = calcResizedDimensions(pim2, linesep, targetLineSep)
    pim2 = pim2.resize((targetW, targetH))
    scale_factor = pim1.height / targetH

    # staff line features
    X2 = getNormImage(pim2)
    hlines = isolateStaffLines(X2, morphFilterHorizLineSize, notebarFiltLen, notebarRemoval)
    featmap, stavelens, columnWidth = computeStaveFeatureMap(hlines, calcStaveFeatureMap_NumCols, calcStaveFeatureMap_LowerRange, calcStaveFeatureMap_UpperRange, calcStaveFeatureMap_Delta)

    # notehead detection
    im3 = morphFilterCircle(pim2, morphFilterCircleSizeReduce, morphFilterCircleSizeExpand)
    keypoints, im_with_keypoints = detectNoteheadBlobs(im3, notedetect_minarea, notedetect_maxarea)
    if len(keypoints) == 0:
        with open(outfile,'a') as f:
            print(imagefile+"-- there are no keypoints", file=f)
        return np.array([]).reshape(62,0)
    X3 = getNormImage(im3) # im indicates grayscale [0, 255], X indicates [0, 1] inverted grayscale
    ntemplate, numCrops = getNoteTemplate(X3, keypoints, noteTemplateSize)
    chordBlockSpecs = (chordBlock_minH, chordBlock_maxH, chordBlock_minW, chordBlock_maxW, chordBlock_minArea, chordBlock_maxArea, chordBlock_minNotes, chordBlock_maxNotes)
    notes, img_binarized_notes = adaptiveNoteheadDetect(X3, ntemplate, notedetect_tol_ratio, chordBlockSpecs)
    if len(notes) < maxNumStaves: # if few or no notes detected, stop early (avoids later errors during kmeans clustering)
        with open(outfile,'a') as f:
            print(imagefile+"-- too few noteheads", file=f)
        return np.array([]).reshape(62, 0)
    nhlocs, nhlen_est, nhwidth_est = getNoteheadInfo(notes)

    # infer note values
    estStaffLineLocs, sfiltlen = getEstStaffLineLocs(featmap, nhlocs, stavelens, columnWidth, maxDeltaRowInitial, int(-2*targetLineSep))
    staveMidpts = estimateStaffMidpoints(estStaffLineLocs, minNumStaves, maxNumStaves, minStaveSeparation)
    staveIdxs, nhRowOffsets = assignNoteheadsToStaves(nhlocs, staveMidpts)
    estStaffLineLocs, sfiltlen = getEstStaffLineLocs(featmap, nhlocs, stavelens, columnWidth, maxDeltaRowRefined, (nhRowOffsets - 2*targetLineSep).astype(np.int))
    nhvals = estimateNoteLabels(estStaffLineLocs)

    # cluster noteheads & staves
    vlines = isolateBarlines(X2, morphFilterVertLineLength, morphFilterVertLineWidth, maxBarlineWidth)
    staveMapping, evidence = determineStaveGrouping(staveMidpts, vlines)
    nhclusters, clusterPairs = clusterNoteheads(staveIdxs, staveMapping)

    # generate bootleg scores
    nhdata = [(int(np.round(nhlocs[i][0])), int(np.round(nhlocs[i][1])), nhvals[i], nhclusters[i]) for i in range(len(nhlocs))]
    bscore, events, eventIndices, staffLinesBoth = generateImageBootlegScore(nhdata, clusterPairs, bootlegRepeatNotes, bootlegFiller, minColDiff = nhwidth_est)

    return bscore
