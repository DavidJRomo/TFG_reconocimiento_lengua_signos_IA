import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KPC_dibujar
from model import KeyPointClassifier

# Añadidos
from utils import diccionario
from PIL import ImageFont, ImageDraw, Image


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Asignacion de argumentos de ejecucion
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Preparamos la camara
    cap = cv.VideoCapture(cap_device)    

    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 3)

    # Variables con el ancho y alo del video capturado
    video_ancho = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    video_alto = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    # Cargamos el modelo
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    keypoint_classifierd = KPC_dibujar()

    # Cargamos las etiquetas de los diferentes gestos
    with open('model/keypoint_classifier/keypoint_etiquetas_LSE.csv', 'r',
            encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
    ]

    with open('model/keypoint_classifier/keypoint_etiquetas_dibujar.csv', 'r',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels_d = csv.reader(f)
        keypoint_classifier_labels_d = [
            row[0] for row in keypoint_classifier_labels_d
        ]



    # Calculamos los fotogramas por segundo (FPS)
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Datos y variables
    mode = 0 # Modo inicial
    programa = 1 # Programa inicial

    # Variables gestos
    buffer_letras = '' # Buffer donde se guardaran las letras de los gestos que vamos realizando
    texto_corregido = '' # Texto final que se mostrará en pantalla con las palabras corregidas
    letra_final = '' # Letra que se deduce de las acumuladas en buffer_letras
    texto_temp = '' # Variable auxiliar donde se guardan las letras antes de ser corregidas
    espacio = 0 # Contador para añadir un espacio al texto
    final = 0 # Contador para detectar cuando deseamos borrar todo el texto escrito

    # Variables dibujar
    dibujo = [] # Lista donde se guardan las coordenadas de la linea que se esta dibujando
    dibujoC = [] # Lista donde se guaran todas las coordenadas de las lineas pintadas
    aux = [] # Lista auxiliar
    previo = "" # Variable que guarda el gesto previo realizado
    grosor_linea = 2 # Grosor inicial de las lineas
    color_linea = (255,255,255) # Color inicial de las lineas
    g_sel = "2" # Grosor seleccionado
    c_sel = "w" # Color seleccionado
    borrado = 0 # Variable para contar el numero de iteraciones necesarias hasta confirmar borrado
    contador_dibujar = 0 # Contador para el cambio a dibujar
    contador_parar = 0 # Contador para el cambio a parar
    contador_borrado = 0 # Contador para el cambio a borrar
    contador_deshacer = 0 # Contador para el cambio a deshacer
    gesto = "" # Gesto que queremos forzar
    repeticiones = 10 # Numero de repeticiones de un gesto para considerarlo activo

    # Variables enseñar
    valor = -1 # Variable para guardar el valor de la tecla pulsada
    numero = 0 # Valor correspondiente a la tecla pulsada
    contador = 0 # Contador del numero de veces que se realizará la grabación
    # Variable para determinar que gestos vamos a guardar
    # Si es true es para guardar gestos del programa dibujar, si es false del programa de signos
    dibujar = True
    if dibujar:
        etiquetas = 'model/keypoint_classifier/keypoint_etiquetas_dibujar.csv'
        datos = 'model/keypoint_classifier/keypoint_dibujar.csv'
    else:
        etiquetas = 'model/keypoint_classifier/keypoint_etiquetas_LSE.csv'
        datos = 'model/keypoint_classifier/keypoint_LSE.csv'

    ########################################################

    while True:
        fps = cvFpsCalc.get()

        # Salimos de la aplicación con ESC
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode, programa = select_mode(key, mode, programa)

        # Capturamos la imagen de la camara
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        
        # Implementación de la deteccion de las manos
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if programa == 1:   # Modo Signos ----------------------------------------------------------------------------------------------------------------
             # Comprobamos si hemos detectado alguna mano en la camara
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    # Calculamos la posicion de los diferentes apendices
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Normalizamos las coordenadas de los apendices
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # Identificamos que gesto se esta realizando
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    
                    # Dibujamos un circulo verde arriba a la derecha de la camara para indicar que se esta reconociendo la mano
                    cv.circle(debug_image,(int(video_ancho - 15), 15), 5, (0,255,0), thickness=-1)
                    
                    # Comprobamos el texto
                    # Guardamos en un buffer las letras que detecta el modelo
                    buffer_letras += keypoint_classifier_labels[hand_sign_id -1]
                    # Hasta que no hay 40 letras en el buffer no comprobamos que letra puede ser
                    if len(buffer_letras) >= 40:
                        # Si en el buffer la letra mas comun se ha repetido mas de 20 veces, guardamos esa letra
                        if Counter(buffer_letras).most_common(1)[0][1] > 20:
                            letra_final = Counter(buffer_letras).most_common(1)[0][0]
                            # Si ya hay al menos 2 elementos en el array...
                            if len(texto_temp) > 2:
                                # ... comprobamos si son iguales y la misma que la siguiente letra
                                # En caso afirmativo no la guardamos
                                if not (texto_temp[-1] == texto_temp[-2] == letra_final):
                                    texto_temp += str(letra_final)
                            else:
                                # Las letras que vamos guardando las acumulamos en un array temporal
                                texto_temp += str(letra_final)
                        # Tras realizar las operaciones limpiamos el buffer y las variables de añadir un espacio y final de texto
                        buffer_letras = ''
                        espacio = 0
                        final = 0

            else:
                buffer_letras = ''
                # Dibujamos un circulo rojo arriba a la derecha de la camara para indicar que NO se esta reconociendo la mano
                cv.circle(debug_image,((int(video_ancho - 15)), 15), 5, (0,0,255), thickness=-1)

                # Comprobamos si hemos estado suficiente tiempo sin mostrar una mano como para añadir un espacio
                if espacio == 30 and texto_temp:
                    # Comprobamos la ultima letra añadida
                    # Si es \ quiere decir que el texto escrito no se debe modificar
                    if texto_temp[-1] == "\\":
                        # Eliminamos el caracter \ y ponemos el texto en minuscula
                        texto_temp = texto_temp[:-1]
                        texto_temp = texto_temp.lower()
                    # En caso de no ser \, comprobamos si la palabra escrita esta en el diccionario y la corregimos en funcion a la mas probable
                    else:
                        texto_temp = diccionario.comprobar_palabra(texto_temp.lower())
                    # Añadimos al texto final la palabra corregida junto al espacio
                    texto_corregido += texto_temp + " "
                    texto_corregido = texto_corregido.replace(texto_corregido[0], texto_corregido[0].upper())
                    # Vaciamos el texto temporal y el contador de espacio y final de texto
                    texto_temp = ""
                    espacio = 0
                    final = 0
                espacio += 1
                # En caso de no haber una mano en camara durante un tiempo se considera que se desea borrar el texto escrito en pantalla
                if final == 60:
                    texto_corregido = ''
                    texto_temp = ''
                    final = 0
                final += 1

            # Dibujamos en la parte superior izquierda los fps
            #debug_image = draw_fps(debug_image, fps)
            
            # Escribir el texto en pantalla
            # Comprobamos si no esta vacio el array del texto temporal o el texto corregido y llamamos a la funcion que escribe el texto por pantalla
            if texto_corregido or texto_temp:
                debug_image = subtitulo(debug_image, texto_corregido, texto_temp, video_ancho, video_alto)
                            
        elif programa == 2: # Modo Dibujar ----------------------------------------------------------------------------------------------------------------
            # Comprobamos si hemos detectado alguna mano en la camara
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):

                    # Calculamos la posicion de los diferentes apendices
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Normalizamos las coordenadas de los apendices
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # Identificamos que gesto se esta realizando
                    hand_sign_id = keypoint_classifierd(pre_processed_landmark_list)

                    # Comprobamos que gesto estamos realizando
                    # Si el gesto es "Dibujar" reiniciamos los contadores y fijamos "dibujar"
                    if keypoint_classifier_labels_d[hand_sign_id -1] == "Dibujar":
                        if previo == "Borrar":
                            contador_dibujar += 1
                        else:
                            contador_dibujar = repeticiones
                            contador_parar = 0
                            contador_borrado = 0
                            contador_deshacer = 0
                            gesto = "Dibujar"
                    # Si el gesto es "Parar" pero previamente habia otro, iniciamos un contador para que el cambio sea tras varios bucles con dicho gesto
                    # Con esto evitamos que ante un fallo de la deteccion no se corte el dibujo instantanemante
                    elif keypoint_classifier_labels_d[hand_sign_id -1] == "Parar" and previo in ("Dibujar", "Borrar", "Deshacer"):
                        contador_parar += 1
                        contador_borrado = 0
                        contador_deshacer = 0
                        if contador_parar == repeticiones:
                            gesto = "Parar"
                        else:
                            if previo == "Borrar":
                                contador_dibujar += 1
                            else:
                                gesto = "Dibujar"
                    # Si el gesto es "Borrar" pero previamente habia otro, iniciamos un contador para que el cambio sea tras varios bucles con dicho gesto
                    # Con esto evitamos que ante un fallo de la deteccion no se corte el dibujo instantanemante
                    elif keypoint_classifier_labels_d[hand_sign_id -1] == "Borrar" and previo in ("Dibujar", "Parar", "Deshacer"):
                        contador_borrado += 1
                        contador_parar = 0
                        contador_deshacer = 0
                        if contador_borrado == repeticiones:
                            gesto = "Borrar"
                        else:
                            gesto = "Dibujar"
                    # Si el gesto es "Deshacer" pero previamente habia otro, iniciamos un contador para que el cambio sea tras varios bucles con dicho gesto
                    # Con esto evitamos que ante un fallo de la deteccion no se corte el dibujo instantanemante
                    elif keypoint_classifier_labels_d[hand_sign_id -1] == "Deshacer" and previo in ("Dibujar", "Parar", "Borrar"):
                        contador_deshacer += 1
                        contador_parar = 0
                        contador_borrado = 0
                        if contador_deshacer == int(fps / 2):
                            gesto = "Deshacer"
                        else:
                            gesto = "Dibujar"

                    # Si estamos en el gesto de "Dibujar" comenzamos a realizar lineas
                    # Los puntos que forman la linea son las coordenadas por las que va pasando el dedo indice
                    if gesto == "Dibujar" or (keypoint_classifier_labels_d[hand_sign_id -1] == "Dibujar" and contador_dibujar == int(fps / 2)):
                        # Vamos concatenando las coordenadas en 2 variables
                        dibujo.append(landmark_list[8] + list(color_linea) + [grosor_linea])
                        aux.append(dibujo[-1])
                        # Recorremos la variable dibujo pintando lineas
                        for i in range(len(dibujo) -1):
                            cv.line(debug_image,
                                    (dibujo[i][0], dibujo[i][1]), (dibujo[i+1][0], dibujo[i+1][1]),
                                    color=(dibujo[i][2],dibujo[i][3],dibujo[i][4]),
                                    thickness=dibujo[i][5]
                                    )
                        previo = "Dibujar"
                    # Si usamos el gesto "Parar" paramos de dibujar
                    elif gesto == "Parar" and keypoint_classifier_labels_d[hand_sign_id -1] == "Parar":
                        if dibujo:
                            dibujoC.append(list(aux))
                            aux.clear()
                            dibujo.clear()
                        grosor_linea, color_linea, c_sel, g_sel = selector(landmark_list, video_ancho, grosor_linea, color_linea, c_sel, g_sel)
                        contador_parar = 0
                        previo = "Parar"
                    # Si usamos el gesto "Borrar", borramos el dibujo completo de la pantalla
                    elif gesto == "Borrar" and keypoint_classifier_labels_d[hand_sign_id -1] == "Borrar":
                        if borrado == repeticiones:
                            dibujoC.clear()
                            aux.clear()
                            dibujo.clear()
                            borrado = 0
                        borrado += 1
                        contador_dibujar = 0
                        contador_borrado = 0
                        previo = "Borrar"
                    # Si usamos el gesto "Deshacer", vamos borrando poco a poco los ultimos trazos realizados
                    elif gesto == "Deshacer" and keypoint_classifier_labels_d[hand_sign_id -1] == "Deshacer":
                        if dibujo:
                            dibujoC.append(list(aux))
                            aux.clear()
                            dibujo.clear()
                        if dibujoC:
                            if len(dibujoC[len(dibujoC) -1]) == 0:
                                dibujoC = [l for l in dibujoC if l !=[]]
                            else:
                                dibujoC[len(dibujoC) -1].pop(-1)
                        contador_deshacer = 0
                        previo = "Deshacer"

            # Dibujamos en la parte superior izquierda los fps
            #debug_image = draw_fps(debug_image,fps)

            # Dibujamos el dibujo completo mientras no estamos en modo dibujo para que se vea continuamente en pantalla
            for j in range(len(dibujoC)):
                for k in range(len(dibujoC[j]) -1):
                    cv.line(debug_image,
                            (dibujoC[j][k][0], dibujoC[j][k][1]), (dibujoC[j][k+1][0], dibujoC[j][k+1][1]),
                            color=(dibujoC[j][k+1][2],dibujoC[j][k+1][3],dibujoC[j][k+1][4]),
                            thickness=dibujoC[j][k+1][5]
                            )

            # Dibujamos un cuadrado sobre el color y grosor seleccionados
            estilo_sel(debug_image, video_ancho, g_sel, c_sel)

            # Dibujamos las opciones de estilo disponibles
            estilos(debug_image, video_ancho)
            
        elif programa == 3: # Modo Enseñar ----------------------------------------------------------------------------------------------------------------
            # Comprobamos si hemos detectado alguna mano en la camara
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):

                    # Calculamos la posicion de los diferentes apendices
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Normalizamos las coordenadas de los apendices
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # Identificamos que gesto se esta realizando
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                    # Mensaje informativo en la parte inferior de la pantalla
                    mensaje = "Presionar \"S/D\" y a continuacion la letra/numero del gesto que se va a guardar"
                    mensaje_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) - int(cv.getTextSize(mensaje, cv.FONT_HERSHEY_SIMPLEX, 0.9, 4)[0][0])) / 2
                    cv.putText(debug_image, mensaje, (int(mensaje_size), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) - 20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv.LINE_AA)
                    cv.putText(debug_image, mensaje, (int(mensaje_size), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) - 20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2, cv.LINE_AA)

                    # Si entramos en el modo 1 (S) y no esta activo el modo dibujar
                    if dibujar:
                        debug_image, mode, valor, numero, contador = grabacion_dibujo(debug_image, mode,valor, numero, contador, pre_processed_landmark_list, etiquetas, datos)
                    else:
                        debug_image, mode, valor, numero, contador = grabacion_signos(debug_image, mode, valor, numero, contador, pre_processed_landmark_list, etiquetas, datos)
                    
                    # Dibujamos información
                    debug_image = draw_landmarks(debug_image, landmark_list)

            # Dibujamos en la parte superior izquierda los fps
            debug_image = draw_fps(debug_image,fps)
            
            # Si el valor introducido es un valor correcto lo mostramos por pantalla
            if valor != -1:
                cv.putText(debug_image, "Valor:" + chr(valor), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
                cv.putText(debug_image, "Valor:" + chr(valor), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 2, cv.LINE_AA)

        # Información de funcionamiento
        debug_image = informacion(debug_image, video_ancho, programa)
        # Creamos una ventana que muestra la imagen que capta la camara junto a las modificaciones que le hemos realizado durante el programa
        cv.imshow('Signos/Dibujar/Ensenar', debug_image)    
       
    cap.release()
    cv.destroyAllWindows()


# Funciones por defecto del programa ----------------------------------------------------------------------------------------
def select_mode(key, mode, programa):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 115:  # s
        mode = 1
    if key == 100:  # d
        mode = 2
    if key == 49:  # 1
        programa = 1
    if key == 50:  # 2
        programa = 2
    if key == 48:  # 0
        programa = 3
    return number, mode, programa

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

# Funciones añadidas -----------------------------------------------------------------------------------------------------------
# Funciones globales
def informacion(image, ancho, programa):
    informacion = "Modos: 1-Gestos | 2-Dibujar"
    modo_1 = "1-Gestos"
    modo_2 = "2-Dibujar"
    modo_3 = "0-Enseñar"
    # Transformamos la imagen a RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Transformamos la imagen de formato mat a formato image
    imagen_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(imagen_pil)
    # Fuente de la letra
    font = ImageFont.truetype("arial.ttf", 20)
    # Posicion central texto
    mitad = (int(ancho) - int(draw.textlength(informacion,font=font)))
    # Imprimimos el texto subrayando la opcion actual
    if programa == 1:
        draw.text((mitad-50, 10), "Modos: " + modo_1 + " | " + modo_2, font=font,) 
        draw.line([(mitad - 50 + font.getlength("Modos: "), font.getbbox(informacion)[3] +10),
                   (mitad - 50 + font.getlength("Modos: " + modo_1), font.getbbox(informacion)[3] +10)],
                   fill=None, width= 1)  
    elif programa == 2: 
        draw.text((mitad-50, 10), "Modos: " + modo_1 + " | " + modo_2, font=font,) 
        draw.line([(mitad - 50 + font.getlength("Modos: " + modo_1 + " | "), font.getbbox(informacion)[3] +10),
                   (mitad - 50 + font.getlength("Modos: " + modo_1 + " | " + modo_2), font.getbbox(informacion)[3] +10)],
                   fill=None, width= 1)     
    elif programa == 3:
        draw.text((mitad-50, 10), "Modo: " + modo_3, font=font,)
        draw.line([(mitad - 50 + font.getlength("Modo: "), font.getbbox(informacion)[3] +10),
            (mitad - 50 + font.getlength("Modo: " + modo_3), font.getbbox(informacion)[3] +10)],
            fill=None, width= 1)  

    # Devolvemos la imagen a formato mat
    image = cv.cvtColor(np.array(imagen_pil), cv.COLOR_RGB2BGR)

    return image

# Funciones Signos
def subtitulo (image, corregido, temp, ancho, alto):
    # Para poder imprimir por pantala simbolos especiales (como la letra Ñ) debemos usar la libreria PIL y no openCV
    # En primer lugar convertimos la imagen a formato RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Transformamos la imagen de formato mat a formato image
    imagen_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(imagen_pil)
    # Elegimos la fuente del texto (si da error con "arial", se cambiará a "Arial")
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:        
        font = ImageFont.truetype("Arial.ttf", 30)    
    # Calculamos diferentes tamaños para facilitar las impresiones en pantalla
    # Calculamos la posicion donde ira el texto para estar centrado en la camara
    texto_pos = (int(ancho) - int(draw.textlength(corregido + temp,font=font))) / 2
    # Tamaño del signo _ para calcular el tamaño del rectangulo de fondo
    size_ = font.getlength("_")
    # Ancho del texto completo
    texto_ancho = draw.textlength(corregido + temp,font=font)
    # Tamaño del texto segun la fuente elegida
    font_x0, font_y0, font_x1, font_y1 = font.getbbox(corregido + temp)
    # Dibujamos un rectangulo negro tras el texto para facilitar su lectura
    draw.rectangle([(texto_pos - font_x0 - 5, alto - font_y1 - 15),
        (texto_pos + font_x1 + size_ + 5, alto - font_y0 - 5)],
        fill=(0,0,0), outline=(0,0,0)
        )
    # Imprimimos el texto por pantalla + una barra baja indicando donde ira la siguiente letra
    draw.text((texto_pos, alto - font_y1 - 15), corregido + temp, font=font)
    draw.text((texto_pos + texto_ancho, alto - font_y1 - 21), "_", font=font)
    # Devolvemos la imagen a formato mat bgr
    image = cv.cvtColor(np.array(imagen_pil), cv.COLOR_RGB2BGR)

    return image

# Funciones Dibujar
def draw_fps(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    return image

def selector(landmark, width, grosor, color, c, g):
    width_mid = int(width / 2)

    # Primero comprobamos si el dedo indice esta a la altura de los estilos
    if 0 < landmark[8][1] < 50:  
        # Despues comprobamos sobre que estilo esta en funcion de su coordenada X
        # En cada seleccion se fija el color/grosor tanto en valor como con su identificador
        if width_mid >= landmark[8][0] >= width_mid - 50: # Rojo
            color = (0,0,255)
            c = "r"
        elif width_mid - 50 > landmark[8][0] >= width_mid - 100: # Verde
            color = (0,255,0)
            c = "g"
        elif width_mid - 100 > landmark[8][0] >= width_mid - 150: # Azul
            color = (255,0,0)
            c = "b"
        elif width_mid - 150 > landmark[8][0] >= width_mid - 200: # Blanco
            color = (255,255,255)
            c = "w"
        elif width_mid - 200 > landmark[8][0] >= width_mid - 250: # Negro
            color = (0,0,0)
            c = "bl"
        elif width_mid < landmark[8][0] <= width_mid + 50: # Grosor 3
            grosor = 3
            g = "3"
        elif width_mid + 50 < landmark[8][0] <= width_mid + 100: # Grosor 2
            grosor = 2
            g = "2"
        elif width_mid + 100 < landmark[8][0] <= width_mid + 150: # Grosor 1
            grosor = 1 
            g = "1"

    return grosor, color, c, g

def estilos(image, width):
    # Dibujamos los circulos de colores
    cv.circle(image,(int(width/2) -30, 30), 20, (0,0,255), thickness=-1) # Circulo rojo
    cv.circle(image,(int(width/2) -80, 30), 20, (0,255,0), thickness=-1) # Circulo verde
    cv.circle(image,(int(width/2) -130, 30), 20, (255,0,0), thickness=-1) # Circulo azul
    cv.circle(image,(int(width/2) -180, 30), 20, (255,255,255), thickness=-1) # Circulo blanco
    cv.circle(image,(int(width/2) -230, 30), 20, (0,0,0), thickness=-1) # Circulo negro

    # Dibujamos los diferentes grosores grosores
    cv.circle(image,(int(width/2) +30, 30), 3, (255,255,255), thickness=-1) # Grosor 3
    cv.circle(image,(int(width/2) +30, 30), 20, (255,255,255), thickness=2) # Circulo esterior
    cv.circle(image,(int(width/2) +80, 30), 2, (255,255,255), thickness=-1) # Grosor 2
    cv.circle(image,(int(width/2) +80, 30), 20, (255,255,255), thickness=2) # Circulo esterior
    cv.circle(image,(int(width/2) +130, 30), 1, (255,255,255), thickness=-1) # Grosor 1
    cv.circle(image,(int(width/2) +130, 30), 20, (255,255,255), thickness=2) # Circulo esterior

def estilo_sel(image, width, grosor, color):
    width_mid = int(width / 2)

    # En funcion de los identificadores dibujamos un cuadrado sobre el color/grosor seleccionado
    if color == "r":
        cv.rectangle(image, (width_mid - 5, 10), (width_mid - 55, 50), color=(255,255,255), thickness=2) # Rojo
    elif color == "g":
        cv.rectangle(image, (width_mid - 55, 10), (width_mid - 105, 50), color=(255,255,255), thickness=2) # Verde
    elif color == "b":
        cv.rectangle(image, (width_mid - 105, 10), (width_mid - 155, 50), color=(255,255,255), thickness=2) # Azul
    elif color == "w":
        cv.rectangle(image, (width_mid - 155, 10), (width_mid - 205, 50), color=(255,255,255), thickness=2) # Blanco
    elif color == "bl":
        cv.rectangle(image, (width_mid - 205, 10), (width_mid - 255, 50), color=(255,255,255), thickness=2) # Negro
    
    if grosor == "3":
        cv.rectangle(image, (width_mid + 5, 10), (width_mid + 55, 50), color=(255,255,255), thickness=2) # Grosor 3
    elif grosor == "2":
        cv.rectangle(image, (width_mid + 55, 10), (width_mid + 105, 50), color=(255,255,255), thickness=2) # Grosor 2
    elif grosor == "1":
        cv.rectangle(image, (width_mid + 105, 10), (width_mid + 155, 50), color=(255,255,255), thickness=2) # Grosor 1

# Funciones Enseñar
def logging_csv_hand(number, landmark_list, fichero):
    with open(fichero, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])

    return

def busca_letra(letra, fichero):
    i = 1
    number = 0
    
    with open(fichero, 'r', newline="", encoding='utf-8-sig') as l:
            archivo = csv.reader(l)
            for row in archivo:
                if chr(letra).lower() == row[0].lower():
                    number = i
                i += 1

def busca_numero(numero, fichero):
    i = 1
    fila = 0
    with open(fichero, 'r', newline="", encoding='utf-8-sig') as l:
            archivo = csv.reader(l)
            for row in archivo:
                if int(chr(numero)) == i:
                    fila = i
                i += 1
    return fila

def barra_progreso(image, contador):
    barra = contador/10
    # Punto inicial de la barra
    punto_inicial = (10,120)
    # Punto final de la barra
    punto_final = (int(10 + barra),120)
    # Posicion del %
    punto_final_texto = (int(20 + barra), 120)
    # Linea de progreso 
    cv.line(image, punto_inicial, punto_final, (0, 0, 0), 10, cv.LINE_AA)
    cv.line(image, punto_inicial, punto_final, (255, 255, 255), 9, cv.LINE_AA)
    # Texto + %
    cv.putText(image, str(barra) + " %", punto_final_texto, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, str(barra) + " %", punto_final_texto, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    return image

def grabacion_signos(image, mode, valor, numero, contador, landmarks, etiquetas, datos):
    # Si entramos en el modo 1 (S) y no esta activo el modo dibujar
    if mode == 1:
        # Si no se ha pulsado tecla o es una tecla incorrecta
        if valor == -1:
            # Nos quedamos a la espera de que se pulse una tecla
            valor = cv.waitKey(0)
            # Si pulsamos ESC dejamos de esperar por una tecla
            if valor == 27:
                mode = 0
                valor = -1
            else:
                numero = busca_letra(valor, etiquetas)
                if numero == 0:
                    cv.putText(image, "Valor no valido", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 4, cv.LINE_AA)
                    valor = -1
        # Si el valor es correcto, guardamos 1 linea en el archivo csv que las coordenadas actuales de la mano
        else:
            logging_csv_hand(numero, mode, landmarks, datos)

            # Numero lineas que se guardarán
            if contador == 1000:
                mode = 0        # Reiniciamos el modo seleccionado
                contador = 0    # Reiniciamos el contador
                numero = 0      # Reiniciamos el valor correspondiente al introducido por teclado
                valor = -1      # Reiniciamos el valor de la tecla pulsada
                #print("Fin de la grabacion")

            # Mientras no se llegue al final de la grabación imprimimos una barra de progreso
            else:
                image = barra_progreso(image, contador)
                contador += 1

    return image, mode, valor, numero, contador

def grabacion_dibujo(image, mode, valor,numero, contador, landmarks, etiquetas, datos):
    # Si entramos en el modo 2 (D) y esta activo el modo dibujar
    if mode == 2:
        if valor == -1:
            valor = cv.waitKey(0)
            # Si pulsamos ESC dejamos de esperar por una tecla
            if valor == 27:
                mode = 0
                valor = -1
            else:
                numero = busca_numero(valor, etiquetas)
                if numero == 0:
                    cv.putText(image, "Valor no valido", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 4, cv.LINE_AA)
                    valor = -1
        else:
            logging_csv_hand(numero, landmarks, datos)

            # Numero lineas que se guardarán
            if contador == 1000:
                mode = 0        # Reiniciamos el modo seleccionado
                contador = 0    # Reiniciamos el contador
                numero = 0      # Reiniciamos el valor correspondiente al introducido por teclado
                valor = -1      # Reiniciamos el valor de la tecla pulsada
                #print("Fin de la grabacion")

            else:
                image = barra_progreso(image, contador)
                contador += 1

    return image, mode, valor, numero, contador


if __name__ == '__main__':
    main()
