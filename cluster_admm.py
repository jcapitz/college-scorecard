import pandas as pd 
import numpy as np


def cluster(D):
	flag = 0
	n = D.shape[0]
	p = D.shape[1]
	# r = np.ones((n,n,p))
	lamb = np.zeros((n,n,p))
	X = D
	rho = 1
	alpha = 0.5

	iter = 0

	while True:

		r_new = min_r(X,lamb,rho)
		X_new = min_X(X,D,r_new,lamb,rho,alpha)
		lamb_new = update_lamb(X_new,r_new,lamb,rho)

		print(np.linalg.norm(X-X_new,2)/np.linalg.norm(X_new,2))

		if (iter > 20):
			break

		iter = iter + 1
		# print(X_new[0])
		X = X_new
		lamb = lamb_new


	return r_new,X_new


def min_r(X,lamb,rho):
    
    n = X.shape[0] # row dimension from X
    p = X.shape[1] # colum dimemsion from X
    r = np.empty((n,n,p))
    for i in range(n):
        for j in range(n):
            xhat = -(X[i] - X[j] + lamb[i,j]/rho) 
            if np.linalg.norm(xhat,2) > 1./rho:
                r[i,j] = xhat * (1. - 1./(rho*np.linalg.norm(xhat,2)))
            else:
                r[i,j] = np.zeros(p)     

    return r

def min_X(X,D,r,lamb,rho,alpha):
	n = X.shape[0]
	p = X.shape[1]
	X_new = np.zeros([n,p])
	gamma = np.zeros(p)


	for i in range(n):
		ind = list(range(n))
		del ind[i]
		gamma += 1./n*np.sum(lamb[i,ind] + 2. * rho * r[i,ind])

	d_bar = np.sum(D)/n
	x_bar = d_bar - alpha/gamma

	for i in range(n):
		ind = list(range(n))
		del ind[i]
		X_new[i] = 1/(2.*rho + alpha/n) * ( (alpha/n) * D[i] + 2. * rho * x_bar - 1/n*np.sum(lamb[i,ind] + 2. * rho * r[i,ind],axis = 0))

	return X_new


def update_lamb(X,r,lamb,rho):
	n = X.shape[0]
	p = X.shape[1]
	lamb_new = np.empty((n,n,p))
	for i in range(n):
		for j in range(n):
			lamb_new[i,j] = lamb[i,j] + rho * (r[i,j] + X[i] - X[j])

	return lamb_new


def get_clusters(X):
	n = X.shape[0]
	p = X.shape[1]
	X = X[X[:,0].argsort()] #SORT X BY FIRST COLUMN

	return X

# df = pd.read_csv('PitchFxExample.csv')
# D = df.iloc[:,3:].values
D1 = np.random.normal(0,1,[50,50])
D2 = np.random.normal(10,10,[50,50])
D = np.concatenate((D1,D2))


r, X = cluster(D)
# X_new = get_clusters(X)







