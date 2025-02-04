import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.fft import fft2, ifft2
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, linalg as sla
from math import pi

sigma_0 = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

################# Auxiliary functions####################

# Units are chosen so that lattice constant is ds = 1, velocity is c = 1,
# reduced Plack constant is hbar = 1 and e = 1.

def make_potential(parameters, plot = False):
    '''Produces an array potential_array[y,x] = potential(x,y).'''
    
    Nx = parameters['Nx'] #Odd
    Ny = parameters['Ny'] #Odd
    potential_function = parameters['potential'] #Must be applicable to arrays

    x, y = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny), sparse=False, indexing='xy')
    potential_array = potential_function(x,y)
    
    if plot:
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.imshow(potential_array, origin = 'lower')
        ax.set_xlabel(r'$x$ (a)', fontsize = 14)
        ax.set_ylabel(r'$y$ (a)', fontsize = 14)
        ax.set_title(r'Potential', fontsize = 20)
        
    return potential_array

def make_mass(parameters, plot = False):
    '''Produces an array mass_array[y,x] = mass(x,y).'''
    
    Nx = parameters['Nx'] #Odd
    Ny = parameters['Ny'] #Odd
    mass_function = parameters['mass'] #Must be applicable to arrays

    x, y = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny), sparse=False, indexing='xy')
    mass_array = mass_function(x,y)
    
    if plot:
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.imshow(mass_array, origin = 'lower')
        ax.set_xlabel(r'$x$ (a)', fontsize = 14)
        ax.set_ylabel(r'$y$ (a)', fontsize = 14)
        ax.set_title(r'Mass', fontsize = 20)
        
    return mass_array

def make_fluxes(parameters, plot = False):
        '''Produces array of magnetic fluxes for 2 regions with opposite average field.'''

        Nx = parameters['Nx']
        Ny = parameters['Ny']
        N1 = parameters['N1']
        d1 = parameters['d1']
        N2 = parameters['N2']
        B1 = parameters['B1']  #Magnetic field in the first region
        disorder = parameters['disorder']
        
        if B1 == 0:
            B2 = 0
        elif N2 == 0:
            print('Warning: N2 must be non-zero for the total flux through the system to vanish.')
        else:
            B2 = B1*N1/N2
            
        d2 = Ny-N1-N2-d1
        

        fluxes = np.zeros((Ny, Nx))
        for x in range(Nx):
            for y in range(Ny):
                if y < N1:
                    fluxes[y,x] = B1 + disorder*(np.random.rand()-0.5)
                elif y < N1 + d1:
                    fluxes[y,x] = 0
                elif y < N1 + d1 + N2:
                    fluxes[y,x] = -B2 + disorder*(np.random.rand()-0.5)
                else:
                    fluxes[y,x] = 0

        fluxes = fluxes-np.sum(fluxes)/(Nx*Ny)
    
        if plot:
            fig = plt.figure(figsize = (7,7))
            ax = fig.add_subplot(111)
            ax.imshow(fluxes, origin = 'lower', cmap = 'bwr',vmax = max(np.amax(fluxes),-np.amin(fluxes)),vmin=min(-np.amax(fluxes),np.amin(fluxes)))
            ax.set_xlabel(r'$x$ (a)', fontsize = 14)
            ax.set_ylabel(r'$y$ (a)', fontsize = 14)
            ax.set_title(r'Magnetic field', fontsize = 20)
            
        return fluxes
    

def vector_potential(parameters, fluxes):
    'Obtain Peierls phases from fluxes through each lattice cell.'
    
    Nx = parameters['Nx']
    Ny = parameters['Ny']
    
    
    # We are assuming a gauge in which A is zero along all edges except at the top one,
    # where it is equal -total_flux/Nx. This should produce the ordinary magnetic translations exp(1j*B*Ny*x).

    def index(direction, i,j):
        if i < Nx and j < Ny:              # first all the hoppings in the unit cell
            idx = (j*Nx+i)*2 + direction 
        elif j == Ny:                       # then all hoppings on the top edge
            idx = 2*Nx*Ny+i
        elif i == Nx:                       # then all hoppings on the right edge
            idx = 2*Nx*Ny+Nx+j
        return idx

    row = []
    col = []
    data = []
    rhs = []

    row_i = 0

    # Rotational equations
    for i in range(Nx):
        for j in range(Ny):
            if not (i==Nx//2 and j==Ny//2): #skip one (one of them is linearly dependent)
                row += [row_i,row_i, row_i, row_i]
                col += [index(0,i,j),index(1,i+1,j),index(0,i,j+1),index(1,i,j)]
                data += [1,1,-1,-1]
                rhs += [fluxes[j,i]]
                row_i += 1

    # Divergence equations (not at the edges): Coulomb gauge 
    for i in range(1,Nx):
        for j in range(1,Ny):
            row += [row_i, row_i, row_i, row_i]
            col += [index(0,i,j), index(1,i,j), index(0,i-1,j), index(1,i,j-1)]
            data += [1,1,-1,-1]
            rhs += [0]
            row_i += 1

    #Fix the value of A at the edges (allowed by gauge freedom)
    for i in range(Nx): #bottom edge = 0
        row += [row_i]
        col+= [index(0,i,0)]
        data += [1]
        rhs += [0]
        row_i += 1

    for j in range(Ny): #left edge = 0
        row += [row_i]
        col+= [index(1,0,j)]
        data += [1]
        rhs += [0]
        row_i += 1

    for i in range(Nx): #top edge = -total_flux/Nx =0 in this case
        row += [row_i]
        col+= [index(0,i,Ny)]
        data += [1]
        rhs += [0]
        row_i += 1

    for j in range(Ny): #right edge = 0
        row += [row_i]
        col += [index(1,Nx,j)]
        data += [1]
        rhs += [0]
        row_i += 1
        

    equations = csr_matrix((data, (row, col)), shape=(2*Nx*Ny+Nx+Ny, 2*Nx*Ny+Nx+Ny))
    vector_potential = sla.spsolve(equations, rhs)
    vector_potential = vector_potential[:2*Nx*Ny].reshape(Ny,Nx,2)
    a_e = vector_potential[:,:,0]
    a_n = vector_potential[:,:,1]
    
    a_e = a_e - np.average(a_e)
    a_n = a_n - np.average(a_n)
    
    return a_e, a_n


def operators_ribbon(parameters, plot_potential = False, plot_mass = False, plot_mag_field = False):
    '''Returns operators Phi, H and P for a square 
    geometry boundary condition given by a magnetization that rotates parallel to the edge.'''
    
    Nx = parameters['Nx']     # Number of unit cells in x direction (should be odd)
    Ny = parameters['Ny']     # Number of unit cells in y direction (should be odd)
    kx = parameters['kx']
    ky = parameters['ky']
    
    #Generate Peierls phases
    #np.random.seed(0)
    if parameters['B1'] == 0:
        a_e = np.zeros((Ny,Nx))
        a_n = np.zeros((Ny,Nx))
    else:
        fluxes = make_fluxes(parameters, plot = plot_mag_field)
        a_e, a_n = vector_potential(parameters,fluxes)
        
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    
    
    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []    

    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
        
        #Peierls phases
        p_e = np.exp(1j*(a_e[y,x]))
        p_n = np.exp(1j*(a_n[y,x]))
        
        #Standard translations
        trs_e = np.exp(-(1j*kx*Nx)*((x+1)//Nx))
        trs_n = np.exp(-(1j*ky*Ny)*((y+1)//Ny))
        
        #Total phases
        phase_e = p_e*trs_e
        phase_n = p_n*trs_n
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e]
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n*(1-(y//(Ny-1)))] ################## Open boundaries in y direction
        
    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2

    potential_array = make_potential(parameters, plot = plot_potential).flatten()    
    pot = scipy.sparse.spdiags(potential_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    V = scipy.sparse.kron(csc_matrix(sigma_0), pot, format = "csc")    
    
    mass_array = make_mass(parameters, plot = plot_mass).flatten()    
    mass = scipy.sparse.spdiags(mass_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")

    H = H_0 + Phi.H@V@Phi + Phi.H@M@Phi
    
    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a theta,phi orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = site[0] + site[1]*Nx
        spindown = site[0] + site[1]*Nx + Nx*Ny
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # Now rotate the spins on the edge
    theta = parameters['theta']
    for x in range(Nx):
        rotation = spin_rotation([x,0], theta,0)
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
        
    for x in range(Nx):
        rotation = spin_rotation([x,Ny-1], theta, np.pi)
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
    
    ####
    
    indices_to_delete = []#sites and spins on the edge. bot_bound and top_bound must be 0 or 1 to pick spin
    
    for x in range(Nx):
        indices_to_delete.append(Nx*Ny + int(x)) #bottom edge
        indices_to_delete.append(int((Nx*Ny + Nx*(Ny-1) + x))) #top edge
  
        
    # Transforming the sparse matrix into dense to delete spins is probably not the best way to do this
    H_aux = H.toarray()
    Phi_aux = Phi.toarray()
   
    
    H_aux = np.delete(H_aux, indices_to_delete, axis=0)
    H_aux = np.delete(H_aux, indices_to_delete, axis=1)
    
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=0)
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=1)
        
    H = csc_matrix(H_aux)
    Phi = csc_matrix(Phi_aux)
    P = Phi.H@Phi
    
    return Phi, H, P, indices_to_delete

def make_bands_x(parameters,number_of_bands = int(20), number_of_points = int(101),kmin = -pi, kmax = pi):
    '''Calculate and plot bands in x direction.'''
    
    #Generate Peierls phases
    np.random.seed(0)
    fluxes = make_fluxes(parameters)
    a_e, a_n = vector_potential(parameters,fluxes)
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n

    momenta = np.linspace(kmin,kmax, num = number_of_points)
    bands = np.zeros((number_of_points,number_of_bands))
    #Solve generalised eigenproblem fro all k
    for j, kx in enumerate(momenta):
        parameters['kx'] = kx
        Phi, H, P, deleted_indices = operators_ribbon(parameters)
        bands[j] = sla.eigsh(H, M=P, k = number_of_bands, tol = 1e-7, sigma = 0.000001, which = 'LM',return_eigenvectors = False)

    return momenta,bands


############## SQUARE

def generate_square(L,npoints=1000000):
    'Generates the set of points in the grid closest to an square with radii r1 and r2 and the angle of the normal vector.'
    length = np.linspace(0, 4*L, npoints, endpoint = False)
    
    x = np.zeros(npoints)
    y = np.zeros(npoints)
    angles = np.zeros(npoints)
    

    x = np.where(np.abs(length-L/2)<=L/2,L/2*np.ones(npoints),x)
    y = np.where(np.abs(length-L/2)<=L/2,np.linspace(-L/2,4*L-L/2,npoints, endpoint = False),y)
    angles = np.where(np.abs(length-L/2)<=L/2,np.zeros(npoints),angles)

    x = np.where(np.abs(length-3*L/2)<=L/2,np.linspace(3*L/2,3*L/2-4*L,npoints, endpoint = False),x)
    y = np.where(np.abs(length-3*L/2)<=L/2,L/2*np.ones(npoints),y)
    angles = np.where(np.abs(length-3*L/2)<=L/2,pi/2*np.ones(npoints),angles)

    x = np.where(np.abs(length-5*L/2)<=L/2,-L/2*np.ones(npoints),x)
    y = np.where(np.abs(length-5*L/2)<=L/2,np.linspace(5*L/2,5*L/2-4*L,npoints, endpoint = False),y)
    angles = np.where(np.abs(length-5*L/2)<=L/2,pi*np.ones(npoints),angles)

    x = np.where(np.abs(length-7*L/2)<=L/2,np.linspace(-7*L/2,-7*L/2+4*L,npoints, endpoint = False),x)
    y = np.where(np.abs(length-7*L/2)<=L/2,-L/2*np.ones(npoints),y)
    angles = np.where(np.abs(length-7*L/2)<=L/2,3*pi/2*np.ones(npoints),angles)

    
    z = np.exp(1j*angles)


    boundary_points, indices = np.unique(np.round(np.stack((x, y))), axis = 1, return_inverse=True)

    # Calculate the sum of elements in z that correspond to the same value in mask
    sums_real = np.bincount(indices, weights=z.real)
    sums_imag = np.bincount(indices, weights=z.imag)
    
    # Calculate the count of elements in b that correspond to the same value in mask
    counts = np.bincount(indices)
    
    # Calculate the average of elements in b that correspond to the same value in mask
    normal_angles = np.angle((sums_real + 1j*sums_imag)/counts)
    
    
    return boundary_points, normal_angles

def operators_square(parameters, plot_potential = False, plot_mass = False, plot_mag_field = False,return_shape = False):
    '''Returns operators Phi, H and P for a square 
    geometry boundary condition given by a magnetization that rotates parallel to the edge.'''
    
    Nx = parameters['Nx']     # Number of unit cells in x direction (should be odd)
    Ny = parameters['Ny']     # Number of unit cells in y direction (should be odd)
    kx = 0    # open system
    ky = 0    # open system
    
    #Generate Peierls phases
    #np.random.seed(0)
    if parameters['B1'] == 0:
        a_e = np.zeros((Ny,Nx))
        a_n = np.zeros((Ny,Nx))
    else:
        fluxes = make_fluxes(parameters, plot = plot_mag_field)
        a_e, a_n = vector_potential(parameters,fluxes)
        
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    
    
    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []    

    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
        
        #Peierls phases
        p_e = np.exp(1j*(a_e[y,x]))
        p_n = np.exp(1j*(a_n[y,x]))
        
        #Standard translations
        trs_e = np.exp(-(1j*kx*Nx)*((x+1)//Nx))
        trs_n = np.exp(-(1j*ky*Ny)*((y+1)//Ny))
        
        #Total phases
        phase_e = p_e*trs_e
        phase_n = p_n*trs_n
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e*(1-(x//(Nx-1)))] ################## Open boundaries in x direction
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n*(1-(y//(Ny-1)))] ################## Open boundaries in y direction
        
    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2

    potential_array = make_potential(parameters, plot = plot_potential).flatten()    
    pot = scipy.sparse.spdiags(potential_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    V = scipy.sparse.kron(csc_matrix(sigma_0), pot, format = "csc")    
    
    mass_array = make_mass(parameters, plot = plot_mass).flatten()    
    mass = scipy.sparse.spdiags(mass_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")

    H = H_0 + Phi.H@V@Phi + Phi.H@M@Phi

    
    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # Now rotate the spins on the edge
    def get_index(x,y,s):
        '''Returns the index of the orbital in x,y with spin s'''
        return int(Nx*Ny*s + Nx*y + x)

    edge_points, normal_angles = generate_square(L=min(Nx,Ny)-2)
    edge_points = edge_points + np.array([[Nx//2]*(len(edge_points[1])),[Ny//2]*(len(edge_points[1]))])
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
    theta = parameters['theta']
    indices_to_delete = []
    
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        
        #rotate
        rotation = spin_rotation([point[0],point[1]], theta, point[2]) 
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
        
        #book index to delete
        indices_to_delete.append(get_index(point[0],point[1],1))

    #Now we also have to delete the outer part
    amount_out = 0
    def discriminant(x,y):
        L = min(Nx,Ny)-2
        return (np.abs(x-y)+np.abs(x+y))/L
        
    X,Y = np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    for x,y in zip(X.ravel(),Y.ravel()):
        if discriminant(x-Nx//2,y-Ny//2)>1 and  get_index(x,y,1) not in indices_to_delete:
            indices_to_delete.append(get_index(x,y,0))
            indices_to_delete.append(get_index(x,y,1))
            amount_out += 1

            
    # Transforming the sparse matrix into dense to delete spins (could I avoid this?)
    H_aux = H.toarray()
    Phi_aux = Phi.toarray()
   
    
    H_aux = np.delete(H_aux, indices_to_delete, axis=0)
    H_aux = np.delete(H_aux, indices_to_delete, axis=1)
    
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=0)
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=1)
        
    H = csc_matrix(H_aux)
    Phi = csc_matrix(Phi_aux)
    P = Phi.H@Phi

    if return_shape:
        inside_indices = np.delete(np.arange(2*Nx*Ny), indices_to_delete)
        inside_x = inside_indices%(np.ones(len(inside_indices))*Nx)
        inside_y = (inside_indices//(np.ones(len(inside_indices))*Nx))%(np.ones(len(inside_indices))*Ny)
        inside_s = inside_indices//(np.ones(len(inside_indices))*Nx*Ny)
    
        return Phi, H, P, indices_to_delete, (inside_x[Nx*Ny-amount_out:],inside_y[Nx*Ny-amount_out:]), (inside_x[:Nx*Ny-amount_out],inside_y[:Nx*Ny-amount_out])
    else:
        return Phi, H, P, indices_to_delete

    
def solve_eigenproblem_square(parameters, energy = 1e-6, number_of_bands = int(1),plot_shape = True):
    Nx = parameters['Nx']     # Number of unit cells in x direction
    Ny = parameters['Ny']     # Number of unit cells in y direction
    #Generate Peierls phases
    np.random.seed(0)
    fluxes = make_fluxes(parameters)
    a_e, a_n = vector_potential(parameters,fluxes)
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    

    if plot_shape:
        Phi, H, P, deleted_indices, spinup_shape, spindown_shape = operators_square(parameters, return_shape = True)
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.scatter(spinup_shape[0],spinup_shape[1], s = 20)
        ax.scatter(spindown_shape[0],spindown_shape[1], s = 20,zorder=-1)
        ax.set_aspect('equal')
        fig.show()
    else:
        Phi, H, P, deleted_indices = operators_square(parameters, return_shape = False)

      #Solve generalised eigenproblem
    eigenvalues, eigenvectors = sla.eigsh(H, M=P, k = number_of_bands, tol = 0, sigma = energy, which = 'LM',return_eigenvectors = True)

    #Refill with zeros the deleted spins
    states = np.zeros((2*Nx*Ny,number_of_bands),dtype = complex)
    count = 0
    for index in range(2*Nx*Ny):
        if index not in deleted_indices:
            states[index] = (Phi@eigenvectors)[index-count]
        else:
            count += 1

    #Now make sure they are orthogonal
    overlaps = states.conjugate().transpose()@states
    ##The overlap can only be non-zero for degenerate states
    degenerate_indices = []
    bulk_indices = []    
    for i in range(overlaps.shape[0]):
        sorted = np.flip(np.sort(np.abs(overlaps[i])))
        if sorted[1]/sorted[0]<0.1: #This threshold (0.1) is a bit arbitrary
            bulk_indices.append(i)
        else:
            degenerate_indices.append(i)

    overlaps_deg = np.delete(overlaps, bulk_indices, axis=0)
    overlaps_deg = np.delete(overlaps_deg, bulk_indices, axis=1)
    overlaps_bulk = np.delete(overlaps, degenerate_indices, axis=0)
    overlaps_bulk = np.delete(overlaps_bulk, degenerate_indices, axis=1)

    states_deg = np.delete(states, bulk_indices, axis=1)
    states_bulk = np.delete(states, degenerate_indices, axis=1)

    evalues, orthogonal_coeff = np.linalg.eigh(overlaps_deg)
    orthogonal = np.append(states_deg@orthogonal_coeff, states_bulk , axis=1) #### These are finally the orthogonalised states
    norm = np.sqrt(np.diag(np.abs(orthogonal.conjugate().transpose()@orthogonal)))
    states = orthogonal/norm[None,:]
    
    # Rebuild state
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)

    ##### This following part is specific for the square ######
    edge_points, normal_angles = generate_square(L = min(Nx,Ny)-2)
    edge_points = edge_points + np.array([[Nx//2]*(len(edge_points[1])),[Ny//2]*(len(edge_points[1]))])
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
      # Now rotate back the spins on the edge
    theta = parameters['theta']
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        rotation = spin_rotation([point[0],point[1]], theta, point[2]+pi)
        states = rotation@states

    ###Now reshape
    states_shaped = np.reshape(states.flatten('F'), newshape = (number_of_bands,2,Ny,Nx), order = 'C')

    ### Now assign again energies
    energies = np.zeros(number_of_bands)
    for i in range(number_of_bands):
        if i in degenerate_indices:
            energies[i] = 0 #This should be zero
        else:
            energies[i] = eigenvalues[i]
    
    return energies, states_shaped, degenerate_indices


############## ELLIPSE

def generate_ellipse(r1,r2,npoints=1000):
    'Generates the set of points in the grid closest to an ellipse with radii r1 and r2 and the angle of the normal vector.'
    theta = np.linspace(0, 2*np.pi, npoints)
    x = r1*np.cos(theta)
    y = r2*np.sin(theta)
    angles = np.where(np.abs(theta-pi)<=pi/2, np.arctan((r1/r2)*np.tan(theta))+pi, np.arctan((r1/r2)*np.tan(theta)))
    z = np.exp(1j*angles)

    boundary_points, indices = np.unique(np.round(np.stack((x, y))), axis = 1, return_inverse=True)

    # Here I want to remove the points that are inside, what do I do with the normal? I guess I just ignore them, why not
    polar_angle = np.arctan2(boundary_points[1],boundary_points[0])
    boundary_points = boundary_points[:,np.argsort(polar_angle)]
    
    
    # Calculate the sum of elements in z that correspond to the same value in mask
    sums_real = np.bincount(indices, weights=z.real)
    sums_imag = np.bincount(indices, weights=z.imag)
    
    # Calculate the count of elements in b that correspond to the same value in mask
    counts = np.bincount(indices)
    
    # Calculate the average of elements in b that correspond to the same value in mask
    normal_angles = np.angle((sums_real + 1j*sums_imag)/counts)
    normal_angles = normal_angles[np.argsort(polar_angle)]

    second_displacements = np.roll(boundary_points,1,axis = 1)+np.roll(boundary_points,-1,axis = 1)-2*boundary_points #This is like a discrete second derivative.
    in_or_out = second_displacements[0]*np.cos(normal_angles) + second_displacements[1]*np.sin(normal_angles)

    #Get rid of points that are actually fully inside
    boundary_points = boundary_points[:,np.argwhere(in_or_out<=0.1).flatten()]
    normal_angles = normal_angles[np.argwhere(in_or_out<=0.1).flatten()]

    ##reapeat to get rid of points that are fully outside
    second_displacements = np.roll(boundary_points,1,axis = 1)+np.roll(boundary_points,-1,axis = 1)-2*boundary_points #This is like a discrete second derivative.
    in_or_out = second_displacements[0]*np.cos(normal_angles) + second_displacements[1]*np.sin(normal_angles) #Scalar product with the normal
    boundary_points = boundary_points[:,np.argwhere(in_or_out>=-1).flatten()]
    normal_angles = normal_angles[np.argwhere(in_or_out>=-1).flatten()]
    
    return boundary_points, normal_angles

def operators_ellipse(parameters, plot_potential = False, plot_mass = False, plot_mag_field = False,return_shape = False):
    '''Returns operators Phi, H and P for a square 
    geometry boundary condition given by a magnetization that rotates parallel to the edge.'''
    r1 = parameters['r1']
    r2 = parameters['r2']

    Nx = parameters['Nx']     # Number of unit cells in x direction (should be odd)
    Ny = parameters['Ny']     # Number of unit cells in y direction (should be odd)
    kx = 0    # open system
    ky = 0    # open system
    
    #Generate Peierls phases
    #np.random.seed(0)
    if parameters['B1'] == 0:
        a_e = np.zeros((Ny,Nx))
        a_n = np.zeros((Ny,Nx))
    else:
        fluxes = make_fluxes(parameters, plot = plot_mag_field)
        a_e, a_n = vector_potential(parameters,fluxes)
        
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    
    
    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []    

    print('Building matrix')
    
    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
        
        #Peierls phases
        p_e = np.exp(1j*(a_e[y,x]))
        p_n = np.exp(1j*(a_n[y,x]))
        
        #Standard translations
        trs_e = np.exp(-(1j*kx*Nx)*((x+1)//Nx))
        trs_n = np.exp(-(1j*ky*Ny)*((y+1)//Ny))
        
        #Total phases
        phase_e = p_e*trs_e
        phase_n = p_n*trs_n
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e*(1-(x//(Nx-1)))] ################## Open boundaries in x direction
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n*(1-(y//(Ny-1)))] ################## Open boundaries in y direction
    print('Done')
        
    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2

    potential_array = make_potential(parameters, plot = plot_potential).flatten()    
    pot = scipy.sparse.spdiags(potential_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    V = scipy.sparse.kron(csc_matrix(sigma_0), pot, format = "csc")    
    
    mass_array = make_mass(parameters, plot = plot_mass).flatten()    
    mass = scipy.sparse.spdiags(mass_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")

    H = H_0 + Phi.H@V@Phi + Phi.H@M@Phi

    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    # def spin_rotation(matrix, site, theta, phi):
    #     'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
    #     block = np.zeros((2,2),dtype = complex)
        
    #     spinup = int(site[0] + site[1]*Nx)
    #     spindown = int(site[0] + site[1]*Nx + Nx*Ny)

    #     rotation = np.array([[np.cos(theta/2),np.sin(theta/2)] , [-np.sin(theta/2)*np.exp(1j*phi),np.cos(theta/2)*np.exp(1j*phi)]])

    #     block[0,0] = matrix.toarray()[spinup, spinup]
    #     block[0,1] = matrix.toarray()[spinup, spindown]
    #     block[1,0] = matrix.toarray()[spindown, spinup]
    #     block[1,1] = matrix.toarray()[spindown, spindown]

    #     rotated_block = rotation.conjugate().transpose() @ block @ rotation

    #     rotated_block_coo = coo_matrix((rotated_block.ravel(), ([spinup,spinup,spindown,spindown], [spinup,spindown,spinup,spindown])), shape=matrix.shape)

    #     # Add the modified block back into the CSC matrix by converting it to CSC
    #     matrix += rotated_block_coo.tocsc()
        
    #     return 0

    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # Now rotate the spins on the edge
    def get_index(x,y,s):
        '''Returns the index of the orbital in x,y with spin s'''
        return int(Nx*Ny*s + Nx*y + x)
    print('Defining edge')
    edge_points, normal_angles = generate_ellipse(r1, r2)
    edge_points = edge_points + np.array([[Nx//2]*(len(edge_points[1])),[Ny//2]*(len(edge_points[1]))])
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
    theta = parameters['theta']
    indices_to_delete = []
    print('Rotating')
    for nnn,point in enumerate(zip(edge_points[0], edge_points[1], boundary_spin_projections)):
        # spin_rotation(H,[point[0],point[1]], theta, point[2])
        # spin_rotation(Phi,[point[0],point[1]], theta, point[2])

        #rotate
        rotation = spin_rotation([point[0],point[1]], theta, point[2]) 
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
        
        #book index to delete
        indices_to_delete.append(get_index(point[0],point[1],1))

    #Now we also have to delete the outer part
    amount_out = 0
    def discriminant(x,y):
        return (x/r1)**2 + (y/r2)**2
    X,Y = np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    print('Defining inside')
    for x,y in zip(X.ravel(),Y.ravel()):
        if discriminant(x-Nx//2,y-Ny//2)>=1 and  get_index(x,y,1) not in indices_to_delete:
            indices_to_delete.append(get_index(x,y,0))
            indices_to_delete.append(get_index(x,y,1))
            amount_out += 1
    print('Done')
            
    # Transforming the sparse matrix into dense to delete spins (could I avoid this?)
    H_aux = H.toarray()
    Phi_aux = Phi.toarray()
   
    
    H_aux = np.delete(H_aux, indices_to_delete, axis=0)
    H_aux = np.delete(H_aux, indices_to_delete, axis=1)
    
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=0)
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=1)
        
    H = csc_matrix(H_aux)
    Phi = csc_matrix(Phi_aux)
    P = Phi.H@Phi

    if return_shape:
        inside_indices = np.delete(np.arange(2*Nx*Ny), indices_to_delete)
        inside_x = inside_indices%(np.ones(len(inside_indices))*Nx)
        inside_y = (inside_indices//(np.ones(len(inside_indices))*Nx))%(np.ones(len(inside_indices))*Ny)
        inside_s = inside_indices//(np.ones(len(inside_indices))*Nx*Ny)
        print('Operators built')
        return Phi, H, P, indices_to_delete, (inside_x[Nx*Ny-amount_out:],inside_y[Nx*Ny-amount_out:]), (inside_x[:Nx*Ny-amount_out],inside_y[:Nx*Ny-amount_out])
    else:
        print('Operators built')
        return Phi, H, P, indices_to_delete

def solve_eigenproblem_ellipse(parameters, energy = 1e-6, number_of_bands = int(1),plot_shape = True):
    '''Calculate and plot bands in x direction.
        This function assumes that the only possible non-orthogonal degenerate states are at zero energy.'''
    r1 = parameters['r1']
    r2 = parameters['r2']
    
    Nx = int(round(2*r1) +3)
    Ny = int(round(2*r2) +3)
    parameters['Nx'] = Nx    # Number of unit cells in x direction
    parameters['Ny'] = Ny   # Number of unit cells in y direction
    #Generate Peierls phases
    np.random.seed(0)
    fluxes = make_fluxes(parameters)
    a_e, a_n = vector_potential(parameters,fluxes)
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    

    if plot_shape:
        Phi, H, P, deleted_indices, spinup_shape, spindown_shape = operators_ellipse(parameters, return_shape = True)
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.scatter(spinup_shape[0],spinup_shape[1], s = 20)
        ax.scatter(spindown_shape[0],spindown_shape[1], s = 20,zorder=-1)
        ax.set_aspect('equal')
        fig.show()
    else:
        Phi, H, P, deleted_indices = operators_ellipse(parameters, return_shape = False)

      #Solve generalised eigenproblem
    eigenvalues, eigenvectors = sla.eigsh(H, M=P, k = number_of_bands, tol = 0, sigma = energy, which = 'LM',return_eigenvectors = True)

    #Refill with zeros the deleted spins
    states = np.zeros((2*Nx*Ny,number_of_bands),dtype = complex)
    count = 0
    for index in range(2*Nx*Ny):
        if index not in deleted_indices:
            states[index] = (Phi@eigenvectors)[index-count]
        else:
            count += 1

    #Now make sure they are orthogonal
    overlaps = states.conjugate().transpose()@states
    ##The overlap can only be non-zero for degenerate states
    degenerate_indices = []
    bulk_indices = []    
    for i in range(overlaps.shape[0]):
        sorted = np.flip(np.sort(np.abs(overlaps[i])))
        if sorted[1]/sorted[0]<0.1: #This threshold (0.1) is a bit arbitrary
            bulk_indices.append(i)
        else:
            degenerate_indices.append(i)

    overlaps_deg = np.delete(overlaps, bulk_indices, axis=0)
    overlaps_deg = np.delete(overlaps_deg, bulk_indices, axis=1)
    overlaps_bulk = np.delete(overlaps, degenerate_indices, axis=0)
    overlaps_bulk = np.delete(overlaps_bulk, degenerate_indices, axis=1)

    states_deg = np.delete(states, bulk_indices, axis=1)
    states_bulk = np.delete(states, degenerate_indices, axis=1)

    evalues, orthogonal_coeff = np.linalg.eigh(overlaps_deg)
    orthogonal = np.append(states_deg@orthogonal_coeff, states_bulk , axis=1) #### These are finally the orthogonalised states
    norm = np.sqrt(np.diag(np.abs(orthogonal.conjugate().transpose()@orthogonal)))
    states = orthogonal/norm[None,:]
    
    # Rebuild state
    def spin_rotation_states(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return rotation

    ##### This following part is specific for the ellipse ######
    edge_points, normal_angles = generate_ellipse(r1, r2)
    edge_points = edge_points + np.array([[Nx//2]*(len(edge_points[1])),[Ny//2]*(len(edge_points[1]))])
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
      # Now rotate back the spins on the edge
    theta = parameters['theta']
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        rotation = spin_rotation_states([point[0],point[1]], theta, point[2]+pi)
        states = rotation@states

    ###Now reshape
    states_shaped = np.reshape(states.flatten('F'), newshape = (number_of_bands,2,Ny,Nx), order = 'C')

    ### Now assign again energies
    energies = np.zeros(number_of_bands)
    for i in range(number_of_bands):
        if i in degenerate_indices:
            energies[i] = 0 #This should be zero
        else:
            energies[i] = eigenvalues[i]
    
    return energies, states_shaped, degenerate_indices


############## STADIUM

def generate_stadium(r,L,npoints=1000000):
    'Generates the set of points in the grid closest to a stadium and the angle of the normal vector.'
    theta = np.linspace(0, 2*pi, npoints, endpoint = False)
    sign_theta = 2*np.heaviside(theta-pi,0)-1
    cos_theta_plus = (np.cos(theta))**2   *    (-L/r*np.tan(theta)  + np.sqrt( np.ones(npoints) - ((np.tan(theta))**2*L**2/r**2-1)*(np.tan(theta))**2 ) )
    cos_theta_minus = (np.cos(theta))**2   *    (-L/r*np.tan(theta)  - np.sqrt( np.ones(npoints) - ((np.tan(theta))**2*L**2/r**2-1)*(np.tan(theta))**2 ) )
    
    cos_theta_prime = np.where(sign_theta*np.cos(theta)<=0, cos_theta_plus, cos_theta_minus)
    
    x = np.zeros(npoints)
    y = np.zeros(npoints)
    angles = np.where(theta<=pi, np.arccos(cos_theta_prime),-np.arccos(-cos_theta_prime))
    

    x = np.where(np.abs(theta)<=np.arctan(r/L),L+r*cos_theta_prime,x)
    x = np.where(np.abs(theta-pi/2)<=np.arctan(L/r),r/np.tan(theta),x)
    x = np.where(np.abs(theta-pi)<=np.arctan(r/L),-L-sign_theta*r*cos_theta_prime,x)
    x = np.where(np.abs(theta-3*pi/2)<=np.arctan(L/r),-r/np.tan(theta),x)
    x = np.where(np.abs(theta-2*pi)<=np.arctan(r/L),L-r*cos_theta_prime,x)
    
    y = np.where(np.abs(theta)<=np.arctan(r/L),r*(np.sqrt(1-cos_theta_prime**2)),y)
    y = np.where(np.abs(theta-pi/2)<=np.arctan(L/r),r*np.ones(npoints),y)
    y = np.where(np.abs(theta-pi)<=np.arctan(r/L),-sign_theta*r*(np.sqrt(1-cos_theta_prime**2)),y)
    y = np.where(np.abs(theta-3*pi/2)<=np.arctan(L/r),-r*np.ones(npoints),y)
    y = np.where(np.abs(theta-2*pi)<=np.arctan(r/L),-r*(np.sqrt(1-cos_theta_prime**2)),y)
    
    
    angles = np.where(np.abs(theta-pi/2)<np.arctan(L/r),pi/2*np.ones(npoints),angles)
    angles = np.where(np.abs(theta-3*pi/2)<np.arctan(L/r),-pi/2*np.ones(npoints),angles)

    
    z = np.exp(1j*angles)


    boundary_points, indices = np.unique(np.round(np.stack((x, y))), axis = 1, return_inverse=True)


    polar_angle = np.arctan2(boundary_points[1],boundary_points[0])
    boundary_points = boundary_points[:,np.argsort(polar_angle)]


    # Calculate the sum of elements in z that correspond to the same value in mask
    sums_real = np.bincount(indices, weights=z.real)
    sums_imag = np.bincount(indices, weights=z.imag)
    
    # Calculate the count of elements in b that correspond to the same value in mask
    counts = np.bincount(indices)
    
    # Calculate the average of elements in b that correspond to the same value in mask
    normal_angles = np.angle((sums_real + 1j*sums_imag)/counts)
    normal_angles = normal_angles[np.argsort(polar_angle)]

    second_displacements = np.roll(boundary_points,1,axis = 1)+np.roll(boundary_points,-1,axis = 1)-2*boundary_points #This is like a discrete second derivative.
    in_or_out = second_displacements[0]*np.cos(normal_angles) + second_displacements[1]*np.sin(normal_angles)

    #Get rid of points that are actually fully inside
    boundary_points = boundary_points[:,np.argwhere(in_or_out<=0.1).flatten()]
    normal_angles = normal_angles[np.argwhere(in_or_out<=0.1).flatten()]

    ##reapeat to get rid of points that are fully outside
    second_displacements = np.roll(boundary_points,1,axis = 1)+np.roll(boundary_points,-1,axis = 1)-2*boundary_points #This is like a discrete second derivative.
    in_or_out = second_displacements[0]*np.cos(normal_angles) + second_displacements[1]*np.sin(normal_angles) #Scalar product with the normal
    boundary_points = boundary_points[:,np.argwhere(in_or_out>=-1).flatten()]
    normal_angles = normal_angles[np.argwhere(in_or_out>=-1).flatten()]
    
    return boundary_points, normal_angles

def operators_stadium(parameters, plot_potential = False, plot_mass = False, plot_mag_field = False,return_shape = False):
    '''Returns operators Phi, H and P for a stadium 
    geometry boundary condition given by a magnetization that rotates parallel to the edge.'''
    
    L = parameters['L']
    r = parameters['r']
    Nx = parameters['Nx']
    Ny = parameters['Ny']
    kx = 0    # open system
    ky = 0    # open system
    
    #Generate Peierls phases
    #np.random.seed(0)
    if parameters['B1'] == 0:
        a_e = np.zeros((Ny,Nx))
        a_n = np.zeros((Ny,Nx))
    else:
        fluxes = make_fluxes(parameters, plot = plot_mag_field)
        a_e, a_n = vector_potential(parameters,fluxes)
        
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    
    
    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []    

    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
        
        #Peierls phases
        p_e = np.exp(1j*(a_e[y,x]))
        p_n = np.exp(1j*(a_n[y,x]))
        
        #Standard translations
        trs_e = np.exp(-(1j*kx*Nx)*((x+1)//Nx))
        trs_n = np.exp(-(1j*ky*Ny)*((y+1)//Ny))
        
        #Total phases
        phase_e = p_e*trs_e
        phase_n = p_n*trs_n
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e*(1-(x//(Nx-1)))] ################## Open boundaries in x direction
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n*(1-(y//(Ny-1)))] ################## Open boundaries in y direction
        
    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2

    potential_array = make_potential(parameters, plot = plot_potential).flatten()    
    pot = scipy.sparse.spdiags(potential_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    V = scipy.sparse.kron(csc_matrix(sigma_0), pot, format = "csc")    
    
    mass_array = make_mass(parameters, plot = plot_mass).flatten()    
    mass = scipy.sparse.spdiags(mass_array, 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")

    H = H_0 + Phi.H@V@Phi + Phi.H@M@Phi

    
    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # Now rotate the spins on the edge
    def get_index(x,y,s):
        '''Returns the index of the orbital in x,y with spin s'''
        return int(Nx*Ny*s + Nx*y + x)

    edge_points, normal_angles = generate_stadium(L = L, r = r)
    edge_points = edge_points + np.array([[Nx//2]*(len(edge_points[1])),[Ny//2]*(len(edge_points[1]))])
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
    theta = parameters['theta']
    indices_to_delete = []
    
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        
        #rotate
        rotation = spin_rotation([point[0],point[1]], theta, point[2]) 
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
        
        #book index to delete
        indices_to_delete.append(get_index(point[0],point[1],1))

    #Now we also have to delete the outer part
    amount_out = 0
    def discriminant(x,y):
        if x<-L:
            return ((x+L)/r)**2+(y/r)**2
        elif x>L:
            return ((x-L)/r)**2+(y/r)**2
        else:
            return (y/r)**2
        
    X,Y = np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    for x,y in zip(X.ravel(),Y.ravel()):
        if discriminant(x-Nx//2,y-Ny//2)>1 and  get_index(x,y,1) not in indices_to_delete:
            indices_to_delete.append(get_index(x,y,0))
            indices_to_delete.append(get_index(x,y,1))
            amount_out += 1

            
    # Transforming the sparse matrix into dense to delete spins (could I avoid this?)
    H_aux = H.toarray()
    Phi_aux = Phi.toarray()
   
    
    H_aux = np.delete(H_aux, indices_to_delete, axis=0)
    H_aux = np.delete(H_aux, indices_to_delete, axis=1)
    
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=0)
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=1)
        
    H = csc_matrix(H_aux)
    Phi = csc_matrix(Phi_aux)
    P = Phi.H@Phi

    if return_shape:
        inside_indices = np.delete(np.arange(2*Nx*Ny), indices_to_delete)
        inside_x = inside_indices%(np.ones(len(inside_indices))*Nx)
        inside_y = (inside_indices//(np.ones(len(inside_indices))*Nx))%(np.ones(len(inside_indices))*Ny)
        inside_s = inside_indices//(np.ones(len(inside_indices))*Nx*Ny)
    
        return Phi, H, P, indices_to_delete, (inside_x[Nx*Ny-amount_out:],inside_y[Nx*Ny-amount_out:]), (inside_x[:Nx*Ny-amount_out],inside_y[:Nx*Ny-amount_out])
    else:
        return Phi, H, P, indices_to_delete

        
def solve_eigenproblem_stadium(parameters, energy = 1e-6, number_of_bands = int(1),plot_shape = True):
    ''' This function assumes that the only possible non-orthogonal degenerate states are at zero energy.'''
    L = parameters['L']
    r = parameters['r']
    Nx = 2*(L+r)+ 3     # Number of unit cells in x direction
    Ny = 2*r+ 3    # Number of unit cells in y direction
    parameters['Nx'] = Nx
    parameters['Ny'] = Ny
    #Generate Peierls phases
    np.random.seed(0)
    fluxes = make_fluxes(parameters)
    a_e, a_n = vector_potential(parameters,fluxes)
    parameters['a_e'] = a_e
    parameters['a_n'] = a_n
    

    if plot_shape:
        Phi, H, P, deleted_indices, spinup_shape, spindown_shape = operators_stadium(parameters, return_shape = True)
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.scatter(spinup_shape[0],spinup_shape[1], s = 20)
        ax.scatter(spindown_shape[0],spindown_shape[1], s = 20,zorder=-1)
        ax.set_aspect('equal')
        fig.show()
    else:
        Phi, H, P, deleted_indices = operators_stadium(parameters, return_shape = False)

      #Solve generalised eigenproblem
    eigenvalues, eigenvectors = sla.eigsh(H, M=P, k = number_of_bands, tol = 0, sigma = energy, which = 'LM',return_eigenvectors = True)

    #Refill with zeros the deleted spins
    states = np.zeros((2*Nx*Ny,number_of_bands),dtype = complex)
    count = 0
    for index in range(2*Nx*Ny):
        if index not in deleted_indices:
            states[index] = (Phi@eigenvectors)[index-count]
        else:
            count += 1

    #Now make sure they are orthogonal
    overlaps = states.conjugate().transpose()@states
    ##The overlap can only be non-zero for degenerate states
    degenerate_indices = []
    bulk_indices = []    
    for i in range(overlaps.shape[0]):
        sorted = np.flip(np.sort(np.abs(overlaps[i])))
        if sorted[1]/sorted[0]<0.1: #This threshold (0.1) is a bit arbitrary
            bulk_indices.append(i)
        else:
            degenerate_indices.append(i)

    overlaps_deg = np.delete(overlaps, bulk_indices, axis=0)
    overlaps_deg = np.delete(overlaps_deg, bulk_indices, axis=1)
    overlaps_bulk = np.delete(overlaps, degenerate_indices, axis=0)
    overlaps_bulk = np.delete(overlaps_bulk, degenerate_indices, axis=1)

    states_deg = np.delete(states, bulk_indices, axis=1)
    states_bulk = np.delete(states, degenerate_indices, axis=1)

    evalues, orthogonal_coeff = np.linalg.eigh(overlaps_deg)
    orthogonal = np.append(states_deg@orthogonal_coeff, states_bulk , axis=1) #### These are finally the orthogonalised states
    norm = np.sqrt(np.diag(np.abs(orthogonal.conjugate().transpose()@orthogonal)))
    states = orthogonal/norm[None,:]
    
    # Rebuild state
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)

    ##### This following part is specific for the stadium ######
    edge_points, normal_angles = generate_stadium(L = L, r = r)
    edge_points = edge_points + np.array([[Nx//2]*(len(edge_points[1])),[Ny//2]*(len(edge_points[1]))])
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
      # Now rotate back the spins on the edge
    theta = parameters['theta']
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        rotation = spin_rotation([point[0],point[1]], theta, point[2]+pi)
        states = rotation@states

    ###Now reshape
    states_shaped = np.reshape(states.flatten('F'), newshape = (number_of_bands,2,Ny,Nx), order = 'C')

    ### Now assign again energies
    energies = np.zeros(number_of_bands)
    for i in range(number_of_bands):
        if i in degenerate_indices:
            energies[i] = 0 #This should be zero
        else:
            energies[i] = eigenvalues[i]
    
    return energies, states_shaped, degenerate_indices


############## RECTANGLE

def generate_rectangle(Lx, Ly, plot_shape = False):
    'Generates the set of points in the grid closest to a rectangle with sides Lx and Ly and the angle of the normal vector.'

    x1 = Lx*np.ones(Ly-1)
    y1 = np.linspace(1,Ly, Ly-1, endpoint=False)
    angles1 = np.zeros(Ly-1)
    x2 = np.linspace(Lx-1,0, Lx-1, endpoint = False)
    y2 = Ly*np.ones(Lx-1)
    angles2 = (pi/2)*np.ones(Lx-1)
    
    x3 = np.zeros(Ly-1)
    y3 = np.linspace(Ly-1,0, Ly-1, endpoint=False)
    angles3 = pi*np.ones(Ly-1)
    x4 = np.linspace(1,Lx, Lx-1, endpoint = False)
    y4 = np.zeros(Lx-1)
    angles4 = -(pi/2)*np.ones(Lx-1)

    x = np.concatenate((x1,x2,x3,x4))
    y = np.concatenate((y1,y2,y3,y4))

    
    normal_angles = np.concatenate((angles1,angles2,angles3,angles4))
    boundary_points = np.stack((x,y)) + np.array([[1]*(len(normal_angles)),[1]*(len(normal_angles))])


    # Now the points inside. The following is only necessary to plot.
    # Nx = parameters['Nx']
    # Ny = parameters['Ny']
    
    # def discriminant(x,y):
    #     return x>1 and x<Lx+1 and y>1 and y<Ly+1
    
    # def get_index(x,y,s):
    #     '''Returns the index of the orbital in x,y with spin s'''
    #     return int(Nx*Ny*s + Nx*y + x)
    
    # indices_to_delete = []    
    # for point in zip(boundary_points[0], boundary_points[1]):
    #     #book index to delete from the edges
    #     indices_to_delete.append(get_index(point[0],point[1],1))
    # amount_out = 0
    # X,Y = np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    # for x,y in zip(X.ravel(),Y.ravel()):
    #     if not discriminant(x,y) and  get_index(x,y,1) not in indices_to_delete:
    #         indices_to_delete.append(get_index(x,y,0))
    #         indices_to_delete.append(get_index(x,y,1))
    #         amount_out += 1
    
    # inside_indices = np.delete(np.arange(2*Nx*Ny), indices_to_delete)
    # inside_x = inside_indices%(np.ones(len(inside_indices))*Nx)
    # inside_y = (inside_indices//(np.ones(len(inside_indices))*Nx))%(np.ones(len(inside_indices))*Ny)
    # inside_s = inside_indices//(np.ones(len(inside_indices))*Nx*Ny)
    
    # spinup_shape = (inside_x[Nx*Ny-amount_out:],inside_y[Nx*Ny-amount_out:])
    # spindown_shape = (inside_x[:Nx*Ny-amount_out],inside_y[:Nx*Ny-amount_out])

    # ##Plot shape

    # if plot_shape:

    #     fig = plt.figure(figsize = (7,7))
    #     ax = fig.add_subplot(111)
    #     #ax.scatter(spinup_shape[0],spinup_shape[1], s = 10)
    #     ax.scatter(spindown_shape[0],spindown_shape[1], s = 10,zorder=-1)
    #     ax.scatter(boundary_points[0],boundary_points[1], s = 4)
    #     ax.set_aspect('equal')
    
    #     return boundary_points, normal_angles, spinup_shape, spindown_shape, indices_to_delete

    
    return boundary_points, normal_angles

def operators_rectangle(parameters, return_shape = False):
    '''Returns operators Phi, H and P for a rectangle 
    geometry boundary condition given by a magnetization that rotates parallel to the edge.'''
    
    Nx = parameters['Nx']     # Number of unit cells in x direction
    Ny = parameters['Ny']     # Number of unit cells in y direction
    kx = 0    # open system
    ky = 0    # open system
    
    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []    

    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
                
        #Phases
        phase_e = np.exp(-(1j*kx*Nx)*((x+1)//Nx))
        phase_n = np.exp(-(1j*ky*Ny)*((y+1)//Ny))
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e*(1-(x//(Nx-1)))] ################## Open boundaries in x direction
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n*(1-(y//(Ny-1)))] ################## Open boundaries in y direction
        
    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2
       
    mass = scipy.sparse.spdiags(parameters['mass']*np.ones(Nx*Ny), 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")

    H = H_0 + Phi.H@M@Phi
        
    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # Now rotate the spins on the edge
    def get_index(x,y,s):
        '''Returns the index of the orbital in x,y with spin s'''
        return int(Nx*Ny*s + Nx*y + x)

    edge_points, normal_angles = generate_rectangle(Nx-2, Ny-2)
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
    theta = parameters['theta']
    indices_to_delete = []
    
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        
        #rotate
        rotation = spin_rotation([point[0],point[1]], theta, point[2]) 
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
        
        #book index to delete
        indices_to_delete.append(get_index(point[0],point[1],1))

    #Now we also have to delete the outer part
    amount_out = 0
    Lx = Nx-2
    Ly = Ny-2
    def discriminant(x,y):
        return x>1 and x<Lx+1 and y>1 and y<Ly+1
        
    X,Y = np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    for x,y in zip(X.ravel(),Y.ravel()):
        if not discriminant(x,y) and  get_index(x,y,1) not in indices_to_delete:
            indices_to_delete.append(get_index(x,y,0))
            indices_to_delete.append(get_index(x,y,1))
            amount_out += 1

            
    # Transforming the sparse matrix into dense to delete spins
    H_aux = H.toarray()
    Phi_aux = Phi.toarray()
   
    
    H_aux = np.delete(H_aux, indices_to_delete, axis=0)
    H_aux = np.delete(H_aux, indices_to_delete, axis=1)
    
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=0)
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=1)
        
    H = csc_matrix(H_aux)
    Phi = csc_matrix(Phi_aux)
    P = Phi.H@Phi

    if return_shape:
        inside_indices = np.delete(np.arange(2*Nx*Ny), indices_to_delete)
        inside_x = inside_indices%(np.ones(len(inside_indices))*Nx)
        inside_y = (inside_indices//(np.ones(len(inside_indices))*Nx))%(np.ones(len(inside_indices))*Ny)
        inside_s = inside_indices//(np.ones(len(inside_indices))*Nx*Ny)
    
        return Phi, H, P, indices_to_delete, (inside_x[Nx*Ny-amount_out:],inside_y[Nx*Ny-amount_out:]), (inside_x[:Nx*Ny-amount_out],inside_y[:Nx*Ny-amount_out])
    else:
        return Phi, H, P, indices_to_delete

def solve_eigenproblem_rectangle(parameters, energy = 1e-6, number_of_bands = int(1), plot_shape = True):
    Nx = parameters['Nx']     # Number of unit cells in x direction
    Ny = parameters['Ny']     # Number of unit cells in y direction
   
    if plot_shape:
        Phi, H, P, deleted_indices, spinup_shape, spindown_shape = operators_rectangle(parameters, return_shape = True)
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.scatter(spinup_shape[0],spinup_shape[1], s = 20)
        ax.scatter(spindown_shape[0],spindown_shape[1], s = 20,zorder=-1)
        ax.set_aspect('equal')
        fig.show()
    else:
        Phi, H, P, deleted_indices = operators_rectangle(parameters, return_shape = False)

      #Solve generalised eigenproblem
    eigenvalues, eigenvectors = sla.eigsh(H, M=P, k = number_of_bands, tol = 1e-8, sigma = energy, which = 'LM',return_eigenvectors = True)

    #Refill with zeros the deleted spins
    states = np.zeros((2*Nx*Ny,number_of_bands),dtype = complex)
    count = 0
    for index in range(2*Nx*Ny):
        if index not in deleted_indices:
            states[index] = (Phi@eigenvectors)[index-count]
        else:
            count += 1

    #Now make sure they are orthogonal
    overlaps = states.conjugate().transpose()@states
    ##The overlap can only be non-zero for degenerate states
    degenerate_indices = []
    bulk_indices = []    
    for i in range(overlaps.shape[0]):
        sorted = np.flip(np.sort(np.abs(overlaps[i])))
        if sorted[1]/sorted[0]<0.1: #This threshold (0.1) is a bit arbitrary
            bulk_indices.append(i)
        else:
            degenerate_indices.append(i)

    overlaps_deg = np.delete(overlaps, bulk_indices, axis=0)
    overlaps_deg = np.delete(overlaps_deg, bulk_indices, axis=1)
    overlaps_bulk = np.delete(overlaps, degenerate_indices, axis=0)
    overlaps_bulk = np.delete(overlaps_bulk, degenerate_indices, axis=1)

    states_deg = np.delete(states, bulk_indices, axis=1)
    states_bulk = np.delete(states, degenerate_indices, axis=1)

    evalues, orthogonal_coeff = np.linalg.eigh(overlaps_deg)
    orthogonal = np.append(states_deg@orthogonal_coeff, states_bulk , axis=1) #### These are finally the orthogonalised states
    norm = np.sqrt(np.diag(np.abs(orthogonal.conjugate().transpose()@orthogonal)))
    states = orthogonal/norm[None,:]
    
    # Rebuild state
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)

    ##### This following part is specific for the rectangle ######
    edge_points, normal_angles = generate_rectangle(Nx-2, Ny-2)
    # the parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
      # Now rotate back the spins on the edge
    theta = parameters['theta']
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        rotation = spin_rotation([point[0],point[1]], theta, point[2]+pi)
        states = rotation@states

    ###Now reshape
    states_shaped = np.reshape(states.flatten('F'), newshape = (number_of_bands,2,Ny,Nx), order = 'C')

    ### Now assign again energies
    energies = np.zeros(number_of_bands)
    for i in range(number_of_bands):
        if i in degenerate_indices:
            energies[i] = 0 #This should be zero
        else:
            energies[i] = eigenvalues[i]
    
    return energies, states_shaped, degenerate_indices