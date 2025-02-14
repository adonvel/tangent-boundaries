import numpy as np
import matplotlib.pyplot as plt
import tangent_boundaries as tb
from math import pi
import scipy
from scipy.sparse import csr_matrix, csc_matrix, linalg as sla
import sys

P = 4.489
scale_factor = 68
Lx = int((P/4+np.sqrt((P/4)**2-1))*scale_factor)
Ly = int(scale_factor**2/Lx)
nbands = 2
parameters = dict()


parameters['theta'] = -pi/2
parameters['Nx'] = Lx+2
parameters['Ny'] = Ly+2


masses = [x/scale_factor for x in np.linspace(0,10,100)]
job = int(sys.argv[1])
parameters['mass'] = masses[job]


ener, states, degenerate_indices = tb.solve_eigenproblem_rectangle(parameters, number_of_bands = nbands, plot_shape=False)
spectrum = np.sort(ener)

path1 = '/home/donisvelaa/data1/tangent-boundaries/'
path2 = '/home/donisvelaa/github/tangent-boundaries/data_here/'
name = 'rectangles_mass'
np.save(path1+name+'_scalefactor'+str(scale_factor)+'_perimeter'+str(P)+'_mass'+str(job), spectrum, allow_pickle=True)
np.save(path2+name+'_scalefactor'+str(scale_factor)+'_perimeter'+str(P)+'_mass'+str(job), spectrum, allow_pickle=True)
