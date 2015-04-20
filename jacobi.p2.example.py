#Import stuff
import numpy as np
import matplotlib.pyplot as plt

#Import my own file, eigenstuffs.py, for eigenstuffs
import eigenstuffs as eg


n = 100  #Size of matrix
alpha=2. #Value of diagnol in matrix
iter_max = 1000 #Maximum number of iterations
j_ = 2 #j as defined in equations 1.1 and 1.2 in the homework
epsilon = 1e-8 #used to check for convergence


x = np.zeros(n)  #x is the solution vector
                 #it is initialized to my initial guess of all zeros

#Create and fill in the matrix T, defined in the problem statement
T = np.zeros( (n,n) )
for i in range(len(T)):
    for j in range(len(T[i])):
        if i == j:
            T[i][j] = alpha
        elif j == (i-1) or j == (i+1):
            T[i][j] = -1.
            
#calculate the eigenvalue appropriate for j_
evalue = eg.eigenvalue(j_,alpha,n)

#calculate the eigenvector appropriate for j_
evec = eg.eigenvec(j_,n)

#the vector b as defined in Ax = b
B = evalue*evec

#Create a list with which to store the error of each iteration
error = []

#Now let's iterate, Jacobi style
for k in range(iter_max):
    for i in range(len(x)): #for each element of the solution vector
        #The following if...elif...else statements are set up to exploit
        #the sparse nature of T. The summation used in the Jacobi algorithm
        #sums over a bunch of zeros so I cut those zeros out to speed up the code
        if i==0:  #If we're on the first row, only add T[i][i+1]
            summation = T[i][i+1] * x[i+1]
        elif i== (len(x)-1): #If we're on the last row, only add T[i][i-1]
            summation = T[i][i-1] * x[i-1]
        else:  #Only need to add two elements together
            summation = T[i][i+1] * x[i+1] + T[i][i-1] * x[i-1]
        x[i] = 1./T[i][i] * (B[i] - summation) #Update x based on Jacobi algorithm

    #update the error array
    error.append( np.max(np.absolute(np.subtract(evec,x))))
    
    #check for convergence
    numconverge = 0 #count the number of elements of x which have converged
    for i in range(len(x)):
        if abs(x[i] - evec[i]) < epsilon: #If converged within epsilon
                numconverge += 1

    if numconverge == len(x):  #If all the elements of x have converged
        print "Converged!  After %d loops" % (k+1)
        break
else: #If for loop completes with convergence not being acheived
    print "Convergence not achieved after %d number of iterations" % (k+1)


#Now plot the error as a function of iteration number
#I feel all these pyplot commands are sufficiently self-explanatory
#that I don't need to elaborate each one of them.
plt.plot(range(1,len(error)+1),error)
plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel('Error')
s = "Error plot for j=" + str(j_)
plt.title(s)
s = "errorplot_" + str(j_) + ".pdf"
plt.savefig(s)
