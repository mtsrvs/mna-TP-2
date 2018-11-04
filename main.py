# coding=utf-8
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2
import fft as myfft

def yaml_loader(file_path):
    with open(file_path, "r") as file_descriptor:
        data = yaml.load(file_descriptor)
    return data


def frame_selection(position, W, H, w):
    M = [
            ["NW","N","NE"],
            ["W","C","E"],
            ["SW","S","SE"]
        ]
    for i in range(len(M)):
        for j in range(len(M[i])):
            if(position == M[i][j]):
                return int(float(1) / float(3) * (i) * W), int(float(1) / float(3) * (i) * W + w), int(float(1) / float(3) * (j) * H), int(float(1) / float(3) * (j) * H + w)

def process_file(filename, heartbeats, duration, size, position):
    cap = cv2.VideoCapture('videos/'+filename)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #no estoy utilizando el length porque el largo de cada array es en base a la duracion del video.
    #Por lo tanto, la longitud es fps*duration
    frame_duration = int(fps * duration)
    r = np.zeros((1, frame_duration))
    g = np.zeros((1, frame_duration))
    b = np.zeros((1, frame_duration))

    # por lo pronto asumo posicion noroeste
    frame_size = int(min(width, height * size / 100))
    wmin, wmax, hmin, hmax = frame_selection(position, width, height, frame_size)
    print("-----------------------------------")
    print(position, width, height, frame_size)
    print(hmin, hmax, wmin, wmax)
    k = 0
    while (cap.isOpened() and r.size > k):
        ret, frame = cap.read()
        if ret == True:
            #cambie el order de los parametros porque frame es de [720][1280]
            r[0, k] = np.mean(frame[hmin:hmax, wmin:wmax, 0])
            g[0, k] = np.mean(frame[hmin:hmax, wmin:wmax, 1])
            b[0, k] = np.mean(frame[hmin:hmax, wmin:wmax, 2])
        else:
            break
        k = k + 1
    print (k)
    cap.release()
    cv2.destroyAllWindows()

#    n = 1024

    #n tiene que ser multiplo de 2, por lo tanto, en base a la duracion,
    #busco el multiplo mas cercano.
    n = multiploDe2(frame_duration)
    f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n

    r = r[0, 0:n] - np.mean(r[0, 0:n])
    g = g[0, 0:n] - np.mean(g[0, 0:n])
    b = b[0, 0:n] - np.mean(b[0, 0:n])

    # codigo nuestro
    R = np.abs(myfft.fftshift(myfft.fft(r))) ** 2
    G = np.abs(myfft.fftshift(myfft.fft(g))) ** 2
    B = np.abs(myfft.fftshift(myfft.fft(b))) ** 2

    plt.plot(60 * f, R)
    plt.xlim(0, 200)

    plt.plot(60 * f, G)
    plt.xlim(0, 200)
    plt.xlabel("frecuencia [1/minuto]")

    plt.plot(60 * f, B)
    plt.xlim(0, 200)

    print("VIDEO: " + filename)
    print(filename, heartbeats, duration, size, position)
    print("R - Frecuencia cardíaca: ", abs(f[np.argmax(R)]) * 60, " pulsaciones por minuto");
    print("G - Frecuencia cardíaca: ", abs(f[np.argmax(G)]) * 60, " pulsaciones por minuto");
    print("B - Frecuencia cardíaca: ", abs(f[np.argmax(B)]) * 60, " pulsaciones por minuto");

def multiploDe2(length):
    prev = 2
    curr = 2

    while(curr <= length):
        prev = curr
        curr = curr * 2
 #   print("length:", length , "  mul2: " , prev)
    return prev


if __name__ == "__main__":
    file_path = "config.yaml"
    data = yaml_loader(file_path)

    filename = data.get("video").get("name")

    heartbeats = data.get("heartbeats")

    duration = data.get("duration")

    size_change = data.get("video").get("sizeChange")
    for size in range(size_change.get("min"), size_change.get("max"), size_change.get("step")):
        for position in data.get("video").get("positionChange"):
            process_file(filename, heartbeats, duration, size, position)
