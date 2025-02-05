import numpy as np
import matplotlib.pyplot as plt
import tangent_boundaries as tb
from math import pi
import scipy
from scipy.sparse import csr_matrix, csc_matrix, linalg as sla
import sys



scale_factor = 50

area = scale_factor*scale_factor 
nbands = 10
parameters = dict()
parameters['theta'] = -pi/2
parameters['mass'] = 1/scale_factor

sides = [x for x in range(scale_factor,5*scale_factor,1) if abs(x*round(scale_factor**2/x)/(scale_factor**2)-1)<0.005]
#print(len(sides))

#spectrum = np.zeros((len(sides),nbands))
job = int(sys.argv[1])
Lx = sides[job]
    
parameters['Nx'] = Lx+2
parameters['Ny'] = int(area/Lx+2)

ener, states, degenerate_indices = tb.solve_eigenproblem_rectangle(parameters, number_of_bands = nbands, plot_shape=False)
spectrum = np.sort(ener)

path = '/home/donisvelaa/data1/tangent-boundaries/'
name = 'rectangles'
np.save(path+name+'_scalefactor'+str(scale_factor)+'_Lx'+str(Lx), spectrum, allow_pickle=True)
