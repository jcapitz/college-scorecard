import pandas as pd 
import numpy as np


def cluster(D):
	flag = 0
	m = D.shape[0]
	n = D.shape[1]
	# r = np.ones((m,m,n))
	lamb = np.zeros((m,m,n))
	X = D
	rho = 1
	alpha = 0.5

	iter = 0

	while True:

		r_new = min_r(X,lamb,rho)
		X_new = min_X(X,D,r_new,lamb,rho,alpha)
		lamb_new = update_lamb(X_new,r_new,lamb,rho)

		print(np.linalg.norm(X-X_new,2)/np.linalg.norm(X_new,2))

		if (iter > 10):
			break

		iter = iter + 1
		# print(X_new[0])
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
	gamma = np.zeros(n)


	for i in range(m):
		ind = list(range(m))
		del ind[i]
		gamma = gamma + 1./m*np.sum(lamb[i,ind] + 2. * rho * r[i,ind])

	d_bar = np.sum(D)/m
	x_bar = d_bar - alpha/gamma

	for i in range(m):
		X_new[i] = 1/(2.*rho + alpha/m) * ( (alpha/m) * D[i] + 2. * rho * x_bar - gamma)

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
	X = X[X[:,0].argsort()] #SORT X BY FIRST COLUMN

	return X

# df = pd.read_csv('PitchFxExample.csv')
# D = df.iloc[:,3:].values
D1 = np.random.normal(0,1,[50,50])
D2 = np.random.normal(5,2,[50,50])
D = np.concatenate((D1,D2))


r, X = cluster(D)
# X_new = get_clusters(X)







