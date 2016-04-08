def cluster(D):
flag = 0
m = D.shape[0]
n = D.shape[1]
r = np.zeros([m,n])

x = np.random.rand([m,n])
lam = np.random.rand([m,n])


while(flag == 0):
	r_new = min_r(x,ro,lam)
	x_new = min_x(D,ro,lam,r_new,alpha)
	lam = update_lam(lam,ro,r_new,x_new)
	if (.....):
		flag = 1
	x = x_new

return x



def min_r(X,ro,lam):
	m = D.shape[0]
	n = D.shape[1]
	for i in range(0,m):
		for j = range(0,n):
			x_hat = x[j,:] - x[i,:]
			if x_hat + lam[i,j]/ro > 1./ro:
				r[i,j] = (x_hat - lam[i,j]/ro)*( 1. - 1./(ro * (x_hat + lam[i,j]/ro ))
			else:
				r[i,j] = 0
	return r

def min_x( D,ro,lam,r,alpha):
	m = D.shape[0]
	n = D.shape[1]
	X = np.zeros([m,n])
	s = np.zeros([m])
	d_bar = np.sum(D,axis = 0)/n
	for i in range (0,m):
		for j in range(1,n):
			if (j != i):
				s[i] = np.sum(lam[i,j] + 2. * ro * r[i,j])
				
	gamma = np.sum(g)
	x_bar = d_bar - gamma/alpha
	for i in range(0,m):
		X[i,:] = alpha/n * d[i,:] + 2 * ro * x_bar - 1/n * s[i]
	return X

def update_lam(lam,ro,r,X):
	m = D.shape[0]
	n = D.shape[1]
	new_lam = np.zeros([m,n])
	for i in range(0,m):
		for j in range(0,n):
			new_lam[i,j] = lam[i,j] + ro * (r[i,j] + X[i,:] - X[j,:])

	return new_lam


