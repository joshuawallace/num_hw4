#Import stuff
import numpy as np
import matplotlib.pyplot as pp

a = 0.5 #A parameter defined in the problem statement

def initial_guess(x): #A function defining our initial guess
    return 1 + 2*x

def phi(x):  #A function defined in the problem statement
    return 20*np.pi*x*x*x

def phiprime(x):  #The derivative of the above function
    return 60*np.pi*x*x

def phidoubleprime(x):  #The derivative of the derivative of the two-above function
    return 120*np.pi*x

def f(x):  #u''(x) = f(x), which is this f
    return -20. + a*phidoubleprime(x)*np.cos(phi(x)) - a*(phiprime(x))**2 *np.sin(phi(x))

def true_solution(x):  #The analytic solution
    return 1 + 12*x - 10*x*x + a*np.sin(phi(x))


n_grid = 255 #Number of interior grid points
h = 1./float(n_grid+1) #size of grid spacing
u_0 = 1. #Left boundary condition
u_1 = 3. #Right boundary condition
epsilon = 1e-3  #The error tolerance I want to be within the true value before saying I've converged
number_of_iterations_per_level = 10 #Number of iterations per multigrid level


#In what follows, we are trying to solve the matrix equation A u = b

A_diagonal = -2.  #The diagonol elements of my A matrix
A_offdiagonal = 1. #The off-diagonol elements of my tridiagonol A matrix


u = np.zeros(n_grid) #the solution vector

x_binedges = [] #This list will store the x-values of the edges of the cells
for i in range(n_grid):
    x_binedges.append(float(i+1)/float(n_grid+1))

#initialize the solution vector:
for i in range(n_grid):
    u[i] = initial_guess(x_binedges[i])

b = np.zeros(n_grid)  #The RHS vector

for i in range(n_grid): #Initialize this
    b[i] = f(x_binedges[i])*h*h

b[0] = b[0] - u_0 #Add LHS boundary condition to b
b[n_grid-1] = b[n_grid-1] - u_1 #Add RHS boundary condition to b

#Next two lines is to plot the true solution
x_toplot = np.linspace(0,1,200)
pp.plot(x_toplot,true_solution(x_toplot),label="True Solution",lw=4)

#Begin the iteration
for k255 in range(number_of_iterations_per_level):

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * u[i+1]
    elif i== (len(u)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * u[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * u[i+1] + A_offdiagonal * u[i-1]
    u[i] = 1./A_diagonal * (b[i] - summation) #Update u based on Jacobi algorithm

#Calculate the residual
A = np.zeros( (n_grid,n_grid) )
for i in range(len(A)):
    for j in range(len(A[i])):
        if i == j:
            A[i][j] = A_diagonal
        elif j == (i-1) or j == (i+1):
            A[i][j] = A_offdiagonal
print A
tau255 = np.add(b, -1.*np.dot(A,u) )
print tau255

#############################################
#Now to iterate on the residual in the multigrid fashion
#First restrict to 2 times smaller than original grid
tau127 = np.zeros( (n_grid+1)/2 -1)  #The coarser tau
for i in range(len(tau127)):
    tau127[i] = tau255[i*2 + 1]  #Copy values at corresponding cell edges to coarsen

#b127 = np.zeros( (ngrid+1)/2)
#for i in range(len(b127)):
#    b127[i] = b[i*2]


error127 = np.zeros( (n_grid+1)/2 -1) #the error vector, set initally to zero

for k127 in range(number_of_iterations_per_level): #Solve A * error127 = tau127

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error127[i+1]
    elif i== (len(error127)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error127[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error127[i+1] + A_offdiagonal * error127[i-1]
    error127[i] = 1./A_diagonal * (tau127[i] - summation) #Update error127 based on Jacobi algorithm
    
A127 = np.zeros( ((n_grid+1)/2-1,(n_grid+1)/2-1) ) #Make a matrix A for the tau prime calculation
for i in range(len(A127)):
    for j in range(len(A127[i])):
        if i == j:
            A127[i][j] = A_diagonal
        elif j == (i-1) or j == (i+1):
            A127[i][j] = A_offdiagonal

tau127_prime = np.add(tau127, -1./h**2*np.dot(A127,error127)) #Calculate a tau prime to communicate to 
                                                           #Next restriction

#############################################
#Now restrict to 4 times smaller than original grid
tau63 = np.zeros( (n_grid+1)/4 -1)  #The coarser tau
for i in range(len(tau63)):
    tau63[i] = tau127_prime[i*2 + 1]  #Copy values at corresponding cell edges to coarsen

error63 = np.zeros( (n_grid+1)/4 -1) #the error vector, set initally to zero

for k63 in range(number_of_iterations_per_level): #Solve A * error63 = tau63

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error63[i+1]
    elif i== (len(error63)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error63[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error63[i+1] + A_offdiagonal * error63[i-1]
    error63[i] = 1./A_diagonal * (tau63[i] - summation) #Update error63 based on Jacobi algorithm
  
A63 = np.zeros( ((n_grid+1)/4 -1,(n_grid+1)/4 -1) ) #Make a matrix A for the tau prime calculation
for i in range(len(A63)):
    for j in range(len(A63[i])):
        if i == j:
            A63[i][j] = A_diagonal
        elif j == (i-1) or j == (i+1):
            A63[i][j] = A_offdiagonal

tau63_prime = np.add(tau63, -1./h**4*np.dot(A63,error63)) #Calculate a tau prime to communicate to 
                                                           #Next restriction

#############################################
#Now restrict to 8 times smaller than original grid
tau31 = np.zeros( (n_grid+1)/8 -1)  #The coarser tau
for i in range(len(tau31)):
    tau31[i] = tau63_prime[i*2 + 1]  #Copy values at corresponding cell edges to coarsen

error31 = np.zeros( (n_grid+1)/8 -1) #the error vector, set initally to zero

for k31 in range(number_of_iterations_per_level): #Solve A * error31 = tau31

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error31[i+1]
    elif i== (len(error31)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error31[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error31[i+1] + A_offdiagonal * error31[i-1]
    error31[i] = 1./A_diagonal * (tau31[i] - summation) #Update error31 based on Jacobi algorithm
    
A31 = np.zeros( ((n_grid+1)/8 -1,(n_grid+1)/8 -1) ) #Make a matrix A for the tau prime calculation
for i in range(len(A31)):
    for j in range(len(A31[i])):
        if i == j:
            A31[i][j] = A_diagonal
        elif j == (i-1) or j == (i+1):
            A31[i][j] = A_offdiagonal

tau31_prime = np.add(tau31, -1./h**8*np.dot(A31,error31)) #Calculate a tau prime to communicate to 
                                                           #Next restriction

#############################################
#Now restrict to 16 times smaller than original grid
tau15 = np.zeros( (n_grid+1)/16 -1)  #The coarser tau
for i in range(len(tau15)):
    tau15[i] = tau31_prime[i*2 + 1]  #Copy values at corresponding cell edges to coarsen

error15 = np.zeros( (n_grid+1)/16 -1) #the error vector, set initally to zero

for k15 in range(number_of_iterations_per_level): #Solve A * error15 = tau15

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error15[i+1]
    elif i== (len(error15)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error15[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error15[i+1] + A_offdiagonal * error15[i-1]
    error15[i] = 1./A_diagonal * (tau15[i] - summation) #Update error15 based on Jacobi algorithm
    
A15 = np.zeros( ((n_grid+1)/16 -1,(n_grid+1)/16 -1) ) #Make a matrix A for the tau prime calculation
for i in range(len(A15)):
    for j in range(len(A15[i])):
        if i == j:
            A15[i][j] = A_diagonal
        elif j == (i-1) or j == (i+1):
            A15[i][j] = A_offdiagonal

tau15_prime = np.add(tau15, -1./h**16*np.dot(A15,error15)) #Calculate a tau prime to communicate to 
                                                           #Next restriction

#############################################
#Now restrict to 32 times smaller than original grid
tau7 = np.zeros( (n_grid+1)/32 -1)  #The coarser tau
for i in range(len(tau7)):
    tau7[i] = tau15_prime[i*2 + 1]  #Copy values at corresponding cell edges to coarsen

error7 = np.zeros( (n_grid+1)/32 -1) #the error vector, set initally to zero

for k7 in range(number_of_iterations_per_level): #Solve A * error7 = tau7

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error7[i+1]
    elif i== (len(error7)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error7[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error7[i+1] + A_offdiagonal * error7[i-1]
    error7[i] = 1./A_diagonal * (tau7[i] - summation) #Update error7 based on Jacobi algorithm
    
A7 = np.zeros( ((n_grid+1)/32 -1,(n_grid+1)/32 -1) ) #Make a matrix A for the tau prime calculation
for i in range(len(A7)):
    for j in range(len(A7[i])):
        if i == j:
            A7[i][j] = A_diagonal
        elif j == (i-1) or j == (i+1):
            A7[i][j] = A_offdiagonal

tau7_prime = np.add(tau7, -1./h**32*np.dot(A7,error7)) #Calculate a tau prime to communicate to 
                                                           #Next restriction



#############################################
#Now restrict to 64 times smaller than original grid
tau3 = np.zeros( (n_grid+1)/64 -1)  #The coarser tau
for i in range(len(tau3)):
    tau3[i] = tau7_prime[i*2 + 1]  #Copy values at corresponding cell edges to coarsen

error3 = np.zeros( (n_grid+1)/64 -1) #the error vector, set initally to zero

for k3 in range(number_of_iterations_per_level): #Solve A * error3 = tau3

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error3[i+1]
    elif i== (len(error3)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error3[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error3[i+1] + A_offdiagonal * error3[i-1]
    error3[i] = 1./A_diagonal * (tau3[i] - summation) #Update error3 based on Jacobi algorithm
    
A3 = np.zeros( ((n_grid+1)/64 -1,(n_grid+1)/64 -1) ) #Make a matrix A for the tau prime calculation
for i in range(len(A3)):
    for j in range(len(A3[i])):
        if i == j:
            A3[i][j] = A_diagonal
        elif j == (i-1) or j == (i+1):
            A3[i][j] = A_offdiagonal

tau3_prime = np.add(tau3, -1./h**64*np.dot(A3,error3)) #Calculate a tau prime to communicate to 
                                                           #Next restriction




#####################################################################################
#####################################################################################
#####################################################################################

#### Now work our way back "up" the V


####################################
###Moving from 3 grids to 7 grids

error3_prolongated = np.zeros( (n_grid+1)/32 -1)  #Create the prolongated error vector for moving to finer grid
for i in range(len(error3_prolongated)):  #Do simple interpolation to prolongate the error vector
    if i == 0:
        error3_prolongated[0] = .5*(u_0 + error3[0])
    elif i == (len(error3_prolongated) -1):
        error3_prolongated[-1] = .5*(u_1 + error3[-1])
    elif i%2 == 1:
        error3_prolongated[i] = error3[(i-1)/2]
    else:
        error3_prolongated[i] = .5*(error3[i/2] + error3[i/2-1])

error7 = np.add(error7 , error3_prolongated)  #Update error7 based on error3_prolongated vector
 
for k7 in range(number_of_iterations_per_level): #Solve A * error7 = tau7

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error7[i+1]
    elif i== (len(error7)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error7[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error7[i+1] + A_offdiagonal * error7[i-1]
    error7[i] = 1./A_diagonal * (tau7[i] - summation) #Update error7 based on Jacobi algorithm

    
####################################
###Moving from 7 grids to 15 grids

error7_prolongated = np.zeros( (n_grid+1)/16 -1)  #Create the prolongated error vector for moving to finer grid
for i in range(len(error7_prolongated)):  #Do simple interpolation to prolongate the error vector
    if i == 0:
        error7_prolongated[0] = .5*(u_0 + error7[0])
    elif i == (len(error7_prolongated) -1):
        error7_prolongated[-1] = .5*(u_1 + error7[-1])
    elif i%2 == 1:
        error7_prolongated[i] = error7[(i-1)/2]
    else:
        error7_prolongated[i] = .5*(error7[i/2] + error7[i/2-1])

error15 = np.add(error15 , error7_prolongated)  #Update error15 based on error7_prolongated vector
 
for k15 in range(number_of_iterations_per_level): #Solve A * error15 = tau15

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error15[i+1]
    elif i== (len(error15)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error15[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error15[i+1] + A_offdiagonal * error15[i-1]
    error15[i] = 1./A_diagonal * (tau15[i] - summation) #Update error15 based on Jacobi algorithm


####################################
###Moving from 15 grids to 31 grids

error15_prolongated = np.zeros( (n_grid+1)/8 -1)  #Create the prolongated error vector for moving to finer grid
for i in range(len(error15_prolongated)):  #Do simple interpolation to prolongate the error vector
    if i == 0:
        error15_prolongated[0] = .5*(u_0 + error15[0])
    elif i == (len(error15_prolongated) -1):
        error15_prolongated[-1] = .5*(u_1 + error15[-1])
    elif i%2 == 1:
        error15_prolongated[i] = error15[(i-1)/2]
    else:
        error15_prolongated[i] = .5*(error15[i/2] + error15[i/2-1])

error31 = np.add(error31 , error15_prolongated)  #Update error31 based on error15_prolongated vector
 
for k31 in range(number_of_iterations_per_level): #Solve A * error31 = tau31

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error31[i+1]
    elif i== (len(error31)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error31[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error31[i+1] + A_offdiagonal * error31[i-1]
    error31[i] = 1./A_diagonal * (tau31[i] - summation) #Update error31 based on Jacobi algorithm


####################################
###Moving from 31 grids to 63 grids

error31_prolongated = np.zeros( (n_grid+1)/4 -1)  #Create the prolongated error vector for moving to finer grid
for i in range(len(error31_prolongated)):  #Do simple interpolation to prolongate the error vector
    if i == 0:
        error31_prolongated[0] = .5*(u_0 + error31[0])
    elif i == (len(error31_prolongated) -1):
        error31_prolongated[-1] = .5*(u_1 + error31[-1])
    elif i%2 == 1:
        error31_prolongated[i] = error31[(i-1)/2]
    else:
        error31_prolongated[i] = .5*(error31[i/2] + error31[i/2-1])

error63 = np.add(error63 , error31_prolongated)  #Update error63 based on error31_prolongated vector
 
for k63 in range(number_of_iterations_per_level): #Solve A * error63 = tau63

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error63[i+1]
    elif i== (len(error63)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error63[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error63[i+1] + A_offdiagonal * error63[i-1]
    error63[i] = 1./A_diagonal * (tau63[i] - summation) #Update error63 based on Jacobi algorithm


####################################
###Moving from 63 grids to 127 grids

error63_prolongated = np.zeros( (n_grid+1)/2 -1)  #Create the prolongated error vector for moving to finer grid
for i in range(len(error63_prolongated)):  #Do simple interpolation to prolongate the error vector
    if i == 0:
        error63_prolongated[0] = .5*(u_0 + error63[0])
    elif i == (len(error63_prolongated) -1):
        error63_prolongated[-1] = .5*(u_1 + error63[-1])
    elif i%2 == 1:
        error63_prolongated[i] = error63[(i-1)/2]
    else:
        error63_prolongated[i] = .5*(error63[i/2] + error63[i/2-1])

error127 = np.add(error127 , error63_prolongated)  #Update error127 based on error63_prolongated vector
 
for k127 in range(number_of_iterations_per_level): #Solve A * error127 = tau127

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error127[i+1]
    elif i== (len(error127)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error127[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error127[i+1] + A_offdiagonal * error127[i-1]
    error127[i] = 1./A_diagonal * (tau127[i] - summation) #Update error127 based on Jacobi algorithm


####################################
###Moving from 127 grids to 255 grids

error127_prolongated = np.zeros( (n_grid+1) -1)  #Create the prolongated error vector for moving to finer grid
for i in range(len(error127_prolongated)):  #Do simple interpolation to prolongate the error vector
    if i == 0:
        error127_prolongated[0] = .5*(u_0 + error127[0])
    elif i == (len(error127_prolongated) -1):
        error127_prolongated[-1] = .5*(u_1 + error127[-1])
    elif i%2 == 1:
        error127_prolongated[i] = error127[(i-1)/2]
    else:
        error127_prolongated[i] = .5*(error127[i/2] + error127[i/2-1])


"""error255 = np.add(error255 , error127_prolongated)  #Update error255 based on error127_prolongated vector
 
for k255 in range(number_of_iterations_per_level): #Solve A * error255 = tau255

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * error255[i+1]
    elif i== (len(error255)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * error255[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * error255[i+1] + A_offdiagonal * error255[i-1]
    error255[i] = 1./A_diagonal * (tau255[i] - summation) #Update error255 based on Jacobi algorithm"""


####Whew! Enough already!  Now for the final solution


u = np.add(u, error127_prolongated)  #Update u based on error255_prolongated vector
 
for k in range(number_of_iterations_per_level): #Solve A * u = b

    if i==0:  #If we're on the first row, only add A[i][i+1]
        summation = A_offdiagonal * u[i+1]
    elif i== (len(u)-1): #If we're on the last row, only add A[i][i-1]
        summation = A_offdiagonal * u[i-1]
    else:  #Only need to add two elements together
        summation = A_offdiagonal * u[i+1] + A_offdiagonal * u[i-1]
    u[i] = 1./A_diagonal * (b[i] - summation) #Update u based on Jacobi algorithm
                                                           
#Plot the final solution and save the figure
pp.plot(x_binedges,u,label='final iteration', lw=2.5)
pp.xlabel("x")
pp.ylabel("u(x)")
pp.legend(loc='best')
pp.xlim(0,1)
#pp.ylim(0,5.5)
pp.savefig('problem1b.pdf')
