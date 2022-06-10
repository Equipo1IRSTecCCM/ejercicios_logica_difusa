"""
TE3002B Implementacion de robotica inteligente
Equipo 1
    Diego Reyna Reyes A01657387
    Samantha Barron Martinez A01652135
    Jorge Antonio Hoyo Garcia A01658142
Controlador Difuso
Ciudad de Mexico, 10/06/2022
"""
import numpy as np
import skfuzzy as sk
import csv
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
def calcular(inputs,possible_values,entrada,ls_qual_ra,ls_qual,output_case):
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
    ls_qual_ra : numpy.array(float[])
        An array that contains all the possible values of the membership function of the linguistic 
        variables used to describe the output
    ls_qual : numpy.array(float[])
        An array containing every possible value for the output
    output_case : float[]
        A list containing the relationship between the instersections and the output
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
    
    outputs = np.zeros((1,len(output_case)))
    #Obtain tha maximum per each output case
    for i in range(len(output_case)):
        for j in range(len(output_case[i])):
            if minimos[i][j] > outputs[0][output_case[i][j]]:
                outputs[0][output_case[i][j]] = minimos[i][j]
    #Limit the otuput of each membership function to the maximum obtained
    salidas = []
    idx = 0
    for ran in ls_qual_ra:
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
    for i in range(len(ls_qual)):
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
        num += i * ls_qual[idx]
        idx +=1
    z_estrella = num/np.sum(resultado)
    return resultado,z_estrella
def superficie(inputs,possible_values,ls_qual_ra,ls_qual,output_case,names_variables):
    """
    Creates the control surface fo every possible value

    Parameters
    ----------
    inputs : numpy.array(float[])
        An array that contains n rows, a row per input parameter. Each of these rows contains all 
        the possible values of the membership function of the linguistic variables used to describe that input
    possible_values : numpy.array(float[])
        An array containing every possible value for the input, these values define the range of the membership functions given on "inputs"
    ls_qual_ra : numpy.array(float[])
        An array that contains all the possible values of the membership function of the linguistic 
        variables used to describe the output
    ls_qual : numpy.array(float[])
        An array containing every possible value for the output
    names_variables : str[]
        A list containg the names for the x,y and z axis
    """
    Z = []
    #Calculate for every possible value of the inputs (Only two inputs are considered here)
    for i in possible_values[1]:
        t = []
        for j in possible_values[0]:
            _,z_estrella = calcular(inputs,possible_values,[j,i],ls_qual_ra,ls_qual,output_case)
            t.append(z_estrella)
        Z.append(t)
        #print(i)
    #Create the 3D graph
    X,Y = np.meshgrid(possible_values[0],possible_values[1])
    Z = np.array(Z)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    ax.set_xlabel(names_variables[0])
    ax.set_ylabel(names_variables[1])
    ax.set_zlabel(names_variables[2])

if __name__ == "__main__":
    #Create every single membership
    ld_qual = np.arange(0, 16.1,0.5)#velocidades del auto
    ld_names = ["baja","semi-baja","media","semi-alta","alta"]
    ad_qual = np.arange(-180, 180,1)#distancias del auto
    ad_names = ["negativa alta","negativa media","baja","positiva media","positiva alta"]
    ls_qual = np.arange(0, 4,0.1)#velocidades de Usain Bolt
    ls_names = ["baja","semi-baja","media","semi-alta","alta"]
    as_qual = np.arange(-2.0, 2.0,0.1)#velocidades de Usain Bolt
    as_names = ["negativa alta","negativa media","baja","positiva media","positiva alta"]

    #Linear distance to the objective
    ld_qual_ra = []
    ld_qual_ra.append(sk.gaussmf(ld_qual, 0, 0.7))
    ld_qual_ra.append(sk.gaussmf(ld_qual, 4, 1))
    ld_qual_ra.append(sk.gaussmf(ld_qual, 8, 0.7))
    ld_qual_ra.append(sk.gaussmf(ld_qual, 12, 1))
    ld_qual_ra.append(sk.gaussmf(ld_qual, 16, 0.7))

    #Angular distance to the objective
    ad_qual_ra = []
    ad_qual_ra.append(sk.gaussmf(ad_qual, -180, 25))
    ad_qual_ra.append(sk.gaussmf(ad_qual, -90, 25))
    ad_qual_ra.append(sk.trimf(ad_qual, [-10,0,10]))
    ad_qual_ra.append(sk.gaussmf(ad_qual, 90, 25))
    ad_qual_ra.append(sk.gaussmf(ad_qual, 180, 25))

    #Save the inputs
    inputs = [ld_qual_ra,ad_qual_ra]
    possible_values = [ld_qual,ad_qual]

    #Linear speed
    ls_qual_ra = []
    ls_qual_ra.append(sk.trapmf(ls_qual, [0, 0, 0.2 , 0.8]))
    ls_qual_ra.append(sk.trimf(ls_qual, [0.4,1,1.9]))
    ls_qual_ra.append(sk.trimf(ls_qual, [1.1,2,2.9]))
    ls_qual_ra.append(sk.trimf(ls_qual, [2.1,3,3.6]))
    ls_qual_ra.append(sk.trapmf(ls_qual, [3.2, 3.8, 4 , 4]))

    #Angular speed
    as_qual_ra = []
    as_qual_ra.append(sk.trimf(as_qual, [-2,-2,-1]))
    as_qual_ra.append(sk.trimf(as_qual, [-2,-1,0]))
    as_qual_ra.append(sk.trimf(as_qual, [-1,0,1]))
    as_qual_ra.append(sk.trimf(as_qual, [0,1,2]))
    as_qual_ra.append(sk.trimf(as_qual, [1,2,2]))
    #Create our plot
    plt.figure(1)
    plt.subplot(231)
    for idx in range(len(ld_qual_ra)):
        plt.plot(ld_qual,ld_qual_ra[idx], label = ld_names[idx])

    plt.title("Distancia lineal")
    plt.xlabel("~")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()

    plt.subplot(234)
    for idx in range(len(ad_qual_ra)):
        plt.plot(ad_qual,ad_qual_ra[idx], label = ad_names[idx])
    plt.title("Distancia angular")
    plt.xlabel("Grados")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()

    plt.subplot(232)
    for idx in range(len(ls_qual_ra)):
        plt.plot(ls_qual,ls_qual_ra[idx], label = ls_names[idx])
    plt.title("Velocidad lineal")
    plt.xlabel("~")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()

    plt.subplot(235)
    for idx in range(len(as_qual_ra)):
        plt.plot(as_qual,as_qual_ra[idx], label = as_names[idx])
    plt.title("Velocidad angular")
    plt.xlabel("grados/s")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    plt.legend()
    #Velocidad lineal
    entrada = []
    nombre_variables = ["distancia lineal","distancia angular"]
    nombre_salida = "velocidad del peaton"
    for i in nombre_variables:
        s = "Escribe el valor de "+i+": "
        n = input(s)
        entrada.append(float(n))
    names_variables = ["Distancia lineal","Distancia angular","l"]
    output_case = [[0,0,0,0,0],[0,1,1,1,0],[1,2,2,2,1],[2,3,3,3,2],[2,3,4,3,2]]
    resultado,z_estrella = calcular(inputs,possible_values,entrada,ls_qual_ra,ls_qual,output_case)
    s = "La " + nombre_variables[0] + " debe ser de " + str(round(z_estrella,1))
    print(s)
    plt.subplot(233)
    plt.plot(ls_qual,resultado, label = 'salida')
    plt.title("Vel lineal")
    plt.xlabel("~")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    names_variables = ["Distancia lineal","Distancia angular","a"]
    output_case = [[0,1,2,3,4],[0,1,2,3,4],[1,1,2,3,3],[1,1,2,3,3],[1,1,2,3,3]]
    resultado,z_estrella = calcular(inputs,possible_values,entrada,as_qual_ra,as_qual,output_case)
    s = "La " + nombre_variables[1] + " debe ser de " + str(round(z_estrella,1))
    print(s)
    plt.subplot(236)
    plt.plot(as_qual,resultado, label = 'salida')
    plt.title("Velocidad angular")
    plt.xlabel("grad/s")
    plt.ylabel("μ")
    plt.ylim([0,1.1])
    names_variables = ["Distancia lineal","Distancia angular","l"]
    output_case = [[0,0,0,0,0],[0,1,1,1,0],[1,2,2,2,1],[2,3,3,3,2],[2,3,4,3,2]]
    superficie(inputs,possible_values,ls_qual_ra,ls_qual,output_case,names_variables)
    names_variables = ["Distancia lineal","Distancia angular","a"]
    output_case = [[0,1,2,3,4],[0,1,2,3,4],[1,1,2,3,3],[1,1,2,3,3],[1,1,2,3,3]]
    superficie(inputs,possible_values,as_qual_ra,as_qual,output_case,names_variables)
    plt.show()