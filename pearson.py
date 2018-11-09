from scipy.stats import pearsonr as r2
import matplotlib.pyplot as plt
import numpy as np
import hashlib

def coeficiente_pearson(x, y):
    coef_pearson = r2(x, y)
    return coef_pearson[0]




manual = [6,11, 23, 34, 65]
fft_con_led_azul = [1.17, 10.55, 22.27, 33.4, 66.8]
fft_con_led_rojo  = [1.17, 10.55, 1.17, 3.52, 3.52]
fft_con_led_verde = [75.0, 150, 300.01, 450.01, 900.02]

fft_sin_led_azul = [5.86, 11.72, 24.61, 36.92, 73.83]
fft_sin_led_rojo  = [5.86,11.72, 24.61, 36.92, 1.76]
fft_sin_led_verde = [18.75, 1.17, 1.17, 1.76, 1.76]

print("%.5f" % round(coeficiente_pearson(manual, fft_con_led_azul),5))
print("%.5f" % round(coeficiente_pearson(manual, fft_con_led_rojo),5))
print("%.5f" % round(coeficiente_pearson(manual, fft_con_led_verde),5))
print("%.5f" % round(coeficiente_pearson(manual, fft_sin_led_azul),5))
print("%.5f" % round(coeficiente_pearson(manual, fft_sin_led_rojo),5))
print("%.5f" % round(coeficiente_pearson(manual, fft_sin_led_verde),5))
#print (coeficiente_pearson(manual, fft_con_led_azul))
#print (coeficiente_pearson(manual, fft_con_led_rojo))
#print (coeficiente_pearson(manual, fft_con_led_verde))
#print (coeficiente_pearson(manual, fft_sin_led_azul))
#print (coeficiente_pearson(manual, fft_sin_led_rojo))
#print (coeficiente_pearson(manual, fft_sin_led_verde))


