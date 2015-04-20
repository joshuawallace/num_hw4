#Import stuff
import numpy as np
import matplotlib.pyplot as pp

a = 0.5 #A parameter defined in the problem statement

def initial_guess(x):
    return 1 + 2*x

def phi(x):
    return 20*np.pi*x*x*x

def phiprime(x):
    return 60*np.pi*x*x

def phidoubleprime(x):
    return 120*np.pi*x

def f(x):
    return -20. + a*phidoubleprime(x)*np.cos(phi(x)) - a*(phiprime(x))**2 *np.sin(x)

def true_solution(x):
    return 1 + 12*x - 10*x*x + a*np.sin(phi(x))


n_grid = 255 #Number of interior grid points
h = 1./float(n_grid+1) #size of grid spacing
u_0 = 1. #Left boundary condition
u_1 = 3. #Right boundary condition
epsilon = 1e-7

iter_max = 10000 #Set a maximum number of iterations before quitting


A = np.zeros( (n_grid,n_grid) )
for i in range(len(A)):
    for j in range(len(A[i])):
        if i == j:
            A[i][j] = -2.
        elif j == (i-1) or j == (i+1):
            A[i][j] = 1.

u = np.zeros(n_grid) #the solution vector

x_binedges = []
for i in range(n_grid):
    x_binedges.append(float(i+1)/float(n_grid+1))
    #print x_binedges[i]

#initialize the solution vector:
for i in range(n_grid):
    u[i] = initial_guess(x_binedges[i])

b = np.zeros(n_grid)

for i in range(n_grid):
    b[i] = f(x_binedges[i])*h*h

b[0] = b[0] - u_0
b[n_grid-1] = b[n_grid-1] - u_1

true_sol = np.zeros(n_grid)

for i in range(n_grid):
    true_sol[i] = true_solution(x_binedges[i])

x_toplot = np.linspace(0,1,200)
pp.plot(x_toplot,true_solution(x_toplot),label="True Solution")

for k in range(iter_max):
    for i in range(len(u)):

        if i==0:  #If we're on the first row, only add A[i][i+1]
            summation = A[i][i+1] * u[i+1]
        elif i== (len(u)-1): #If we're on the last row, only add A[i][i-1]
            summation = A[i][i-1] * u[i-1]
        else:  #Only need to add two elements together
            summation = A[i][i+1] * u[i+1] + A[i][i-1] * u[i-1]
        u[i] = 1./A[i][i] * (b[i] - summation) #Update u based on Jacobi algorithm

    #check for convergence
    numconverge = 0 #count the number of elements of x which have converged
    for i in range(len(u)):
        if abs(u[i] - true_sol[i]) < epsilon: #If converged within epsilon
                numconverge += 1

    if numconverge == len(u):  #If all the elements of x have converged
        print "Converged!  After %d loops" % (k+1)
        break

    if k==(20-1) or k==(100-1) or k==(1000-1):
        if k == 19:
            ctype = 'red'
        if k == 99:
            ctype = 'green'
        if k == 999:
            ctype = 'black'
        s = str(k+1) + " iterations"
        pp.scatter(x_binedges,u,label=s,c=ctype)


else: #If for loop completes with convergence not being acheived
    print "Convergence not achieved after %d number of iterations" % (k+1)

pp.xlabel("x")
pp.ylabel("u(x)")
pp.legend(loc='best')
pp.xlim(0,1)
pp.savefig('problem1a.pdf')
