# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import fft as myfft

cap = cv2.VideoCapture('videos/2017-09-14 21.53.59.mp4')

#if not cap.isOpened(): 
#    print("No lo pude abrir")
#    return

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1,length))
g = np.zeros((1,length))
b = np.zeros((1,length))

k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        r[0,k] = np.mean(frame[330:360,610:640,0])
        g[0,k] = np.mean(frame[330:360,610:640,1])
        b[0,k] = np.mean(frame[330:360,610:640,2])
#        print(k)
    else:
        break
    k = k + 1


cap.release()
cv2.destroyAllWindows()

n = 1024
f = np.linspace(-n/2,n/2-1,n)*fps/n

r = r[0,0:n]-np.mean(r[0,0:n])
g = g[0,0:n]-np.mean(g[0,0:n])
b = b[0,0:n]-np.mean(b[0,0:n])

#len([1,2,3,4,5,6])//2 #me devuelve el floor de la division.
#len([1,2,3,4,5,6])%2 #me retorna el modulo de la operacion.
#con estas dos operaciones podria crear el fftshift.

#print(np.fft.fftshift([1,2,3,4,5]))
#print(myfft.fftshift([1,2,3,4,5]))

#codigo del profesor
#R = np.abs(np.fft.fftshift(np.fft.fft(r)))**2
#G = np.abs(np.fft.fftshift(np.fft.fft(g)))**2
#B = np.abs(np.fft.fftshift(np.fft.fft(b)))**2
#codigo nuestro
R = np.abs(myfft.fftshift(myfft.fft(r)))**2
G = np.abs(myfft.fftshift(myfft.fft(g)))**2
B = np.abs(myfft.fftshift(myfft.fft(b)))**2


plt.plot(60*f,R)
plt.xlim(0,200)


plt.plot(60*f,G)
plt.xlim(0,200)
plt.xlabel("frecuencia [1/minuto]")

plt.plot(60*f,B)
plt.xlim(0,200)

plt.show()

print("Frecuencia cardíaca: ", abs(f[np.argmax(G)])*60, " pulsaciones por minuto")