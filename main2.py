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
                return int(1/3*i*W), int(1/3*i*W+w), int(1/3*j*H), int(1/3*j*H+w)

def process_file(filename, heartbeats, duration_porc, size, position):
    #print(filename, heartbeats, duration, size, position)
    cap = cv2.VideoCapture('videos/'+filename)

    # if not cap.isOpened():
    #    print("No lo pude abrir")
    #    return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    until = int(length*duration_porc)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    r = np.zeros((1, length))
    g = np.zeros((1, length))
    b = np.zeros((1, length))

    k = 0
    while (cap.isOpened() and (length > k and until > k)):

        ret, frame = cap.read()
        width = frame.shape[0]
        height = frame.shape[1]
        frame_size = int(min(width, height*size/100))
        wmin,wmax,hmin,hmax = frame_selection(position, width, height, frame_size)
        #por lo pronto asumo posicion noroeste

        if ret == True:
            r[0, k] = np.mean(frame[wmin:wmax, hmin:hmax, 0])
            g[0, k] = np.mean(frame[wmin:wmax, hmin:hmax, 1])
            b[0, k] = np.mean(frame[wmin:wmax, hmin:hmax, 2])
        #        print(k)
        else:
            break
        k = k + 1

    cap.release()
    cv2.destroyAllWindows()

    n = 1024
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

    resultados = {
        "position": position,
        "R": abs(abs(f[np.argmax(R)]) * 60- heartbeats),
        "G": abs(abs(f[np.argmax(G)]) * 60- heartbeats),
        "B": abs(abs(f[np.argmax(B)]) * 60- heartbeats),
        "real": heartbeats
    }
    print(resultados)
    #print("Frecuencia cardíaca: ", abs(f[np.argmax(G)]) * 60, " pulsaciones por minuto")


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
