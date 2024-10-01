import numpy as np
import matplotlib.pyplot as plt
import tangent_boundaries as tb
from math import pi

path = '/home/donisvelaa/data1/tangent-boundaries/'

thetas = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

i = -1 ### selected value of theta
Nx = 45
Ny = 35
nbands = int((Nx*Ny)/100*16)
#nbands = 300
print('Bands to calculate: ',nbands)

parameters = dict(
    Nx = Nx,
    Ny = Ny,
    B1 = 0, # no magnetic field
    N1 = 0, #
    d1 = 0, # These are irrelevant for B1 = 0
    N2 = 0, #
    potential = lambda x,y:0.0*np.random.rand(Ny,Nx),
    mass = lambda x,y:0*x,
    disorder = 0,
    theta = -(pi/2)*(thetas[i]/100),
)

# ################## STADIUM
print('Solving stadium')
print('theta/(pi/2) = ', thetas[i])
print('Nx = ', parameters['Nx'])
print('Ny = ', parameters['Ny'])
spectrum_stadium, states_stadium, degenerate_indices_stadium = tb.solve_eigenproblem_stadium(parameters, number_of_bands = nbands, plot_shape = False)

path = '/home/donisvelaa/data1/tangent-boundaries/'

print('Saving stadium spectrum')
name = 'stadium_spectrum'
np.save(path+name+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_theta'+str(thetas[i])+'_nbands'+str(nbands), spectrum_stadium, allow_pickle=True)
print('Saving stadium eigenstates')
name = 'stadium_states'
np.save(path+name+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_theta'+str(thetas[i])+'_nbands'+str(nbands), states_stadium, allow_pickle=True)