##################################################################################
#código para exportar (debería usarse allá donde calculan la superficie) NOTA: la superficie no debe generarse en cada iteración
#se obtiene una superficie (a manera de entrenamiento) y después solo se lee para tomar las acciones de control

size = len(x),len(y)
mat_z = np.zeros(size)
for j in range(len(y)):
    for i in range(len(x)):
        z = fis(x[i],y[j])
        mat_z[i,j] = z
        
with open("surface.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(mat_z)

#código para importar los datos de la(s) superficie(s) de control desde un .csv
with open('~/surface.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    list_z = list(csv_reader)
    
z_surface = [list(map(float, sublist)) for sublist in list_z]

##################################################################################