"""
TE3002B Implementación de robótica inteligente
Equipo 1
    Diego Reyna Reyes A01657387
    Samantha Barrón Martínez A01652135
    Jorge Antonio Hoyo García A01658142
Reporte Ejercicio FIS
Ciudad de México, 02/06/2022
"""
import numpy as np
import skfuzzy as sk
from matplotlib import pyplot as plt

def evaluar(fx,x,v):
    """
    Evaluate the function fx for a value v
    
    Parameters
    ----------
    fx :    numpy.array(float[])
        The array that contains the output values of the functions
    x :     numpy.array(float[])
        The array that contains all the inputs of the function that define the outputs given in fx
    v :     float
        The value of x we want to obtain it's value in fx
    """
    idx = np.where(x==v)
    #In case it's empty
    if idx[0].shape == (0,):
        closest = 0 
        for i in x:
            if abs(v-i) < abs(v-closest):
                closest = i
        idx = np.where(x==closest)
    #Send the value of fx
    return fx[idx[0][0]]
    
def intersectar(a,ax,va,b,bx,vb):
    """
    Intersects 2 membership functions and outputs the minumum \mu of each of the two evaluations

    Parameters
    ----------
    a :     numpy.array(float[])
        The array that contains the output values of the first membership function
    ax :    numpy.array(float[])
        The array that contains all the inputs of the function that define the outputs given in a
    va :     float
        The value of ax we want to obtain it's value in a
    b :     numpy.array(float[])
        The array that contains the output values of the second membership function
    bx :    numpy.array(float[])
        The array that contains all the inputs of the function that define the outputs given in b
    vb :    numpy.array(float[])
        The value of bx we want to obtain it's value in b
    """
    d = [evaluar(a,ax,va),evaluar(b,bx,vb)]
    return min(d)
def calcular(inputs,possible_values,entrada,vp_qual_ra,vp_qual):
    """
    Calculates the discrete output of our fuzzy control system, using discrete inputs

    Parameters
    ----------
    inputs : numpy.array(float[])
        An array that contains n rows, a row per input parameter. Each of these rows contains all 
        the possible values of the membership function of the linguistic variables used to describe that input
    possible_values : numpy.array(float[])
        An array containing every possible value for the input, these values define the range of the membership functions given on "inputs"
    entrada : float[]
        A list containing the discrete values of the input, this list size has to coincide with the number of rows on the "inputs" array
    vp_qual_ra : numpy.array(float[])
        An array that contains all the possible values of the membership function of the linguistic 
        variables used to describe the output
    vp_qual : numpy.array(float[])
        An array containing every possible value for the output
    """
    minimos = []
    #For each input parameter
    for param_idx in range(len(inputs)-1):
        param = inputs[param_idx]
        my_x = possible_values[param_idx]
        #For each possible membership function of the parameter
        for ran_idx in range(len(param)):
            t = []
            ran = param[ran_idx]
            #For each parameter not intersected with this one
            for other_idx in range(param_idx+1,len(inputs)):
                other_x = possible_values[other_idx]
                #For each membership function of the other paramter
                for other_ran in inputs[other_idx]:
                    #Intersect
                    t.append(intersectar(ran,my_x,entrada[param_idx],other_ran,other_x,entrada[other_idx]))
            minimos.append(t)

    """
    The table defining the output we expect for each intersection
    dc -> Distancia corta
    dm -> Distancia media
    dl -> Distancia larga

    vb -> Velocidad baja
    vm -> Velocidad media
    va -> Velocidad alta

    pb -> Velocidad del peatón media
    psb -> Velocidad del peatón semi-baja
    pm -> Velocidad del peatón media
    psb -> Velocidad del peatón semi-alta
    pa -> Velocidad del peatón alta

    Table:
            dc  dm  dl
            ----------
    vb  |   psa psb pb
    vm  |   pa  pm  pb
    va  |   pa  psa pm
    """
    #Define the table with the index of each output case
    output_case = [[3,1,0],[4,2,0],[4,3,2]]
    outputs = np.zeros((1,len(vp_qual_ra)))
    #Obtain tha maximum per each output case
    for i in range(len(output_case)):
        for j in range(len(output_case[i])):
            if minimos[i][j] > outputs[0][output_case[i][j]]:
                outputs[0][output_case[i][j]] = minimos[i][j]

    #Limit the otuput of each membership function to the maximum obtained
    salidas = []
    idx = 0
    for ran in vp_qual_ra:
        t = []
        for v in ran:
            if v > outputs[0][idx]:
                t.append(outputs[0][idx])
            else:
                t.append(v)
        salidas.append(t)
        idx+=1
        
    
    #Obtain a single array with the contour of the resulting topped membership function
    resultado = []
    #Para cada x
    for i in range(len(vp_qual)):
        maxi = 0
        #Evaluar el valor de cada salida en esa x
        for j in range(len(salidas)):
            if maxi < salidas[j][i]:
                maxi = salidas[j][i]
        resultado.append(maxi)
    #Obtain the discrete output value (z*)
    num = 0
    idx = 0
    for i in resultado:
        num += i * vp_qual[idx]
        idx +=1
    z_estrella = num/np.sum(resultado)
    return resultado,z_estrella
def superficie(inputs,possible_values,vp_qual_ra,vp_qual):
    """
    Creates the control surface fo every possible value

    Parameters
    ----------
    inputs : numpy.array(float[])
        An array that contains n rows, a row per input parameter. Each of these rows contains all 
        the possible values of the membership function of the linguistic variables used to describe that input
    possible_values : numpy.array(float[])
        An array containing every possible value for the input, these values define the range of the membership functions given on "inputs"
    vp_qual_ra : numpy.array(float[])
        An array that contains all the possible values of the membership function of the linguistic 
        variables used to describe the output
    vp_qual : numpy.array(float[])
        An array containing every possible value for the output
    """
    Z = []
    #Calculate for every possible value of the inputs (Only two inputs are considered here)
    for i in possible_values[1]:
        t = []
        for j in possible_values[0]:
            _,z_estrella = calcular(inputs,possible_values,[j,i],vp_qual_ra,vp_qual)
            t.append(z_estrella)
        Z.append(t)
    #Create the 3D graph
    X,Y = np.meshgrid(possible_values[0],possible_values[1])
    Z = np.array(Z)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    ax.set_xlabel('Velocidad del carro [km/h]')
    ax.set_ylabel('Distancia al carro [m]')
    ax.set_zlabel('Velocidad del peatón [km/h]')
if __name__ == "__main__":
    #Interesting input cases
    #Caso 1 8.6 130.55
    #Caso 2 95.6 2.5
    #Caso 3 49.3 49.5
    #Caso varias salidas 80.1,34.1
    #Ask for inputs
    entrada = []
    nombre_variables = ["velocidad del carro","distancia al carro"]
    nombre_salida = "velocidad del peaton"
    for i in nombre_variables:
        s = "Escribe el valor de "+i+": "
        n = input(s)
        entrada.append(float(n))
    #Create every single membership
    va_qual = np.arange(5.1, 97.6,1)#velocidades del auto
    da_qual = np.arange(1.1, 150.1,1)#distancias del auto
    vp_qual = np.arange(0.1, 18.6,0.1)#velocidades de Usain Bolt

    #Velocidades del auto
    va_qual_lo = sk.trapmf(va_qual, [5.1, 5.1, 25 , 36])
    #va_qual_me = sk.gbellmf(va_qual, 7,2,52)
    va_qual_me = sk.trapmf(va_qual, [26, 38, 59 , 84])
    va_qual_hi = sk.trapmf(va_qual, [60, 85, 97.6 , 97.6])
    va_qual_ra = [va_qual_lo,va_qual_me,va_qual_hi]


    #Distancias del auto
    da_qual_lo = sk.trimf(da_qual, [1.1,1.1,43])
    da_qual_me = sk.trapmf(da_qual, [20, 45, 95 , 110])
    da_qual_hi = sk.trimf(da_qual, [102,150.1,150.1])
    da_qual_ra = [da_qual_lo,da_qual_me,da_qual_hi]

    inputs = [va_qual_ra,da_qual_ra]
    possible_values = [va_qual,da_qual]
    #Velocidades para que Usain Bolt cruce la calle
    vp_qual_lo = sk.trapmf(vp_qual, [0.1, 0.1, 1.5 , 2.4])
    vp_qual_sl = sk.trimf(vp_qual, [1.6,3.2,5.4])
    vp_qual_me = sk.gaussmf(vp_qual, 7.2, 0.8)
    vp_qual_sh = sk.trimf(vp_qual, [8.4,12,13.8])
    vp_qual_hi = sk.trapmf(vp_qual, [12.6, 14.4, 18.6 , 18.6])
    vp_qual_ra = [vp_qual_lo,vp_qual_sl,vp_qual_me,vp_qual_sh,vp_qual_hi]


    plt.figure(1)
    plt.subplot(141)
    plt.plot(va_qual,va_qual_lo, label = 'baja')
    plt.plot(va_qual,va_qual_me, label = 'media')
    plt.plot(va_qual,va_qual_hi, label = 'alta')
    plt.title("Velocidad del carro")
    plt.xlabel("km/h")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()

    plt.subplot(142)
    plt.plot(da_qual,da_qual_lo, label = 'corta')
    plt.plot(da_qual,da_qual_me, label = 'moderada')
    plt.plot(da_qual,da_qual_hi, label = 'larga')
    plt.title("Distancia al carro")
    plt.xlabel("m")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()

    plt.subplot(143)
    plt.plot(vp_qual,vp_qual_lo, label = 'baja')
    plt.plot(vp_qual,vp_qual_sl, label = 'semi-baja')
    plt.plot(vp_qual,vp_qual_me, label = 'media')
    plt.plot(vp_qual,vp_qual_sh, label = 'semi-alta')
    plt.plot(vp_qual,vp_qual_hi, label = 'alta')
    plt.title("Velocidad del peatón")
    plt.xlabel("km/h")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()

    resultado,z_estrella = calcular(inputs,possible_values,entrada,vp_qual_ra,vp_qual)
    plt.subplot(144)
    plt.plot(vp_qual,resultado, label = 'salida')
    plt.title("Velocidad del peatón truncada")
    plt.xlabel("km/h")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()
    s = "La " + nombre_salida + " debe ser de " + str(round(z_estrella,1)) + " km/h"
    print(s)
    superficie(inputs,possible_values,vp_qual_ra,vp_qual)
    plt.show()