import numpy as np
import matplotlib.pyplot as plt
import tangent_boundaries as tb
from math import pi

path = '/home/donisvelaa/data1/tangent-boundaries/'

thetas = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

i = -1 ### selected value of theta
Nx = 43 #Assume translational invariance in x direction
Ny = 37
#nbands = int((Nx*Ny)/100*16)
nbands = 200
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

# ################## ELLIPSE
# print('Solving ellipse')
# print('theta/(pi/2) = ', thetas[i])
# print('Nx = ', parameters['Nx'])
# print('Ny = ', parameters['Ny'])
# spectrum_ellipse, states_ellipse, degenerate_indices_ellipse = tb.solve_eigenproblem_ellipse(parameters, number_of_bands = nbands, plot_shape = False)


# path = '/home/donisvelaa/data1/tangent_boundaries/final/'
# print('Saving ellipse spectrum')
# name = 'ellipse_spectrum'
# np.save(path+name+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_theta'+str(thetas[i])+'_nbands'+str(nbands), spectrum_ellipse, allow_pickle=True)
# print('Saving ellipse eigenstates')
# name = 'ellipse_states'
# np.save(path+name+'_Nx'+str(Nx)+'_Ny'+str(Ny)+'_theta'+str(thetas[i])+'_nbands'+str(nbands), states_ellipse, allow_pickle=True)



################## SQUARE
parameters['Nx'] =  int(np.round(np.sqrt(pi*Nx*Ny/4))) # We want the square to have the same area as the ellipse
parameters['Ny'] =  int(np.round(np.sqrt(pi*Nx*Ny/4)))

print('Solving square')
print('theta/(pi/2) = ', thetas[i])
print('Nx = ', parameters['Nx'])
print('Ny = ', parameters['Ny'])

spectrum_square, states_square, degenerate_indices_square = tb.solve_eigenproblem_square(parameters, number_of_bands = nbands, plot_shape = False)

print('Saving square spectrum')
name = 'square_spectrum'
np.save(path+name+'_Nx'+str(parameters['Nx'])+'_Ny'+str(parameters['Ny'])+'_theta'+str(thetas[i])+'_nbands'+str(nbands), spectrum_square, allow_pickle=True)
print('Saving square spectrum')
name = 'square_states'
np.save(path+name+'_Nx'+str(parameters['Nx'])+'_Ny'+str(parameters['Ny'])+'_theta'+str(thetas[i])+'_nbands'+str(nbands), states_square, allow_pickle=True)
