"""By Abhisek Jana
code taken from https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
blog http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
"""

import numpy as np        # Trabajar con arrays y matrices
import cv2                # Procesamiento de imagenes
import argparse           # Manejar argumentos del command line
import matplotlib.pyplot as plt # Mostrar y trabajar con imagenes
import math               
from convolution import convolution 
 
# Leer argumentos del archivo de texto
def read_config_file(config_path) -> list:
    with open(config_path, 'r') as file:    

        # Leer las lineas independientes como una lista de argumentos
        lineas = file.read().splitlines()
        args = []
        for line in lineas:
            args += line.split() # Separar cada lista por espacios, agregarlo a la lista
        return args
 
# Calcula la distribucion normal (gausiana) con los argumentos dados
def dnorm(x, mu, sd):
    # Ecuacion de la distribucion normal
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
 
# Funcion para generar una matriz representando el blur gausiano en 2D
def gaussian_kernel(size, sigma=1, verbose=False):
    # Array desde -size / 2 hasta size / 2; con elementos equidistantes
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    
    # Calcula la distribucion normal a cada elemento del kernel
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    
    # Extender el kernel original a 2 dimensiones
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    # Normalizar el kernel para que sus valores esten entre 0 y 1
    kernel_2D *= 1.0 / kernel_2D.max()
 
    # Mostrar el kernel en una grafica con matplotlib
    if verbose:
        plt.figure('Figure 1')
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()
 
    return kernel_2D
 
 
# Aplicar el gaussian blur a una imagen
def gaussian_blur(image, kernel_size, verbose=False):
    # Generar el kernel con las dimensiones dadas
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)

    return convolution(image, kernel, average=True, verbose=verbose)
 

# Main (previene la ejecucion del codigo cuando se importa desde otro archivo) 
if __name__ == '__main__':
    # Leer archivo de argumentos
    config_args = read_config_file('config.txt')

    # Objeto para leer los argumentos
    ap = argparse.ArgumentParser()
    
    # Argumentos requeridos de la imagen
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-b", "--blur",type=int, help="Gaussian blur", default = 9)
    args = vars(ap.parse_args(config_args))
 
    # Cargar la imagen con el nombre dado 
    image = cv2.imread(args["image"])
    
 
    gaussian_blur(image, args["blur"], verbose=True)
