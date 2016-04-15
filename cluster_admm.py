import pandas as pd 
import numpy as np


def cluster(D):
	flag = 0
	m = D.shape[0]
	n = D.shape[1]
	r = np.ones((m,m,n))
	lamb = np.ones((m,m,n))
	X = D
	rho = 0.5
	alpha = 0.5
	iter = 0

	while True:
		r_new = min_r(X,lamb,rho)
		X_new = min_X(X,D,r_new,lamb,rho,alpha)
		lamb_new = update_lamb(X_new,r_new,lamb,rho)

		if (iter > 2):
			# flag = 1
			break

		iter = iter + 1
		print(iter)
		X = X_new
		lamb = lamb_new


	return r_new,X_new


def min_r(X,lamb,rho):
    
    m = X.shape[0] # row dimension from X
    n = X.shape[1] # colum dimemsion from X
    r = np.empty((m,m,n))
    for i in range(m):
        for j in range(m):
            xhat = -(X[i] - X[j] + lamb[i,j]/rho) 
            if np.linalg.norm(xhat,2) > 1./rho:
                r[i,j] = xhat * (1. - 1./(rho*np.linalg.norm(xhat,2)))
            else:
                r[i,j] = np.zeros(n)
                                              
            
    return r

def min_X(X,D,r,lamb,rho,alpha):
	m = X.shape[0]
	n = X.shape[1]
	X_new = np.zeros([m,n])
	for i in range(m):
		ind = list(range(m))
		del ind[i]
		gamma = 1./m*np.sum(lamb[i,ind] + 2. * rho * r[i,ind])
		d_bar = np.sum(D)/m
		x_bar = d_bar - gamma/alpha
		X_new[i] = 1/(2.*rho + alpha/m) * ( (alpha/m) * D[i] + 2 * rho * x_bar - gamma)

	return X_new


def update_lamb(X,r,lamb,rho):
	m = X.shape[0]
	n = X.shape[1]
	lamb_new = np.empty((m,m,n))
	for i in range(m):
		for j in range(m):
			x_hat = X[i] - X[j]
			lamb_new[i,j] = lamb[i,j] + rho * (r[i,j] + X[i] - X[j])
	return lamb_new


def get_clusters(X):
	m = X.shape[0]
	n = X.shape[1]
	# indices = list(range(m))
	labels = np.ones(m)
	tol = 1e-2
	X = np.concatenate((X,np.asarray(list(range(0,m))).reshape(m,1)),axis = 1) #ADD INDICES TO LAST COLUMN OF X
	X = X[X[:,0].argsort()] #SORT X BY FIRST COLUMN

	return X


df = pd.read_csv('PitchFxExample.csv')
D = df.iloc[:,3:].values

#GET X MATRIX 
r_new, X = cluster(D)
X_new = get_clusters(X)







