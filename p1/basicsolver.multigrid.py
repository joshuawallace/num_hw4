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

iter_max = 50000 #Set a maximum number of iterations before quitting


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

true_sol = np.zeros(n_grid) #Make a vector to store the values of the true solution

for i in range(n_grid): #Calculate values of true solution
    true_sol[i] = true_solution(x_binedges[i])

#Next two lines is to plot the true solution
x_toplot = np.linspace(0,1,200)
pp.plot(x_toplot,true_solution(x_toplot),label="True Solution",lw=4)

#Begin the iteration
for k in range(iter_max):
    for i in range(len(u)):

        if i==0:  #If we're on the first row, only add A[i][i+1]
            summation = A_offdiagonal * u[i+1]
        elif i== (len(u)-1): #If we're on the last row, only add A[i][i-1]
            summation = A_offdiagonal * u[i-1]
        else:  #Only need to add two elements together
            summation = A_offdiagonal * u[i+1] + A_offdiagonal * u[i-1]
        u[i] = 1./A_diagonal * (b[i] - summation) #Update u based on Jacobi algorithm

    #check for convergence
    numconverge = 0 #count the number of elements of u which have converged
    for i in range(len(u)):
        if abs(u[i] - true_sol[i]) < epsilon: #If converged within epsilon
                numconverge += 1

    if numconverge == len(u):  #If all the elements of x have converged
        print "Converged!  After %d loops" % (k+1)
        break

    #Plot the 20th, 100th, and 1000th iteration
    if k==(20-1) or k==(100-1) or k==(1000-1):
        s = str(k+1) + " iterations"
        pp.plot(x_binedges,u,label=s,lw=2.5)


else: #If for loop completes with convergence not being acheived
    print "Convergence not achieved after %d number of iterations" % (k+1)

#Plot the final solution and save te figure
pp.plot(x_binedges,u,label='final iteration', lw=2.5)
pp.xlabel("x")
pp.ylabel("u(x)")
pp.legend(loc='best')
pp.xlim(0,1)
pp.savefig('problem1a.pdf')
