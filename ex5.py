import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline

RANDOMSTATE = 43

np.random.seed(RANDOMSTATE)

max_degree = 9
datapoints = 40
num_bootstraps = datapoints
noise = 0.5

lmb_ridge = 4.52e-5  # best value from ex4
lmb_lasso = 3.59e-5  # best value form ex5

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

# Setting up the data
x = np.linspace(0, 1, datapoints)
y = np.linspace(0, 1, datapoints)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
z += noise * np.random.randn(z.shape[0], z.shape[1])

models = [LinearRegression(), 
		  make_pipeline(StandardScaler(), Ridge(lmb_ridge)),
		  make_pipeline(StandardScaler(), Lasso(lmb_lasso))]
bootstrap_MSE = np.zeros((len(models), max_degree))

i = 0
for model in models:
	for degree in tqdm(range(max_degree)):
		X = create_X(x, y, degree)

		X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), random_state=RANDOMSTATE)
		# every row of the matrix is a seperate prediction
		z_pred = np.zeros((z_test.shape[0], num_bootstraps))
		mse = np.zeros(num_bootstraps)
		for bootstrap in range(num_bootstraps):
			X_bootstrap, z_bootstrap = resample(X_train, z_train)
			model.fit(X_bootstrap, z_bootstrap)
			z_pred = model.predict(X_test)
			mse[bootstrap] = mean_squared_error(z_test, z_pred)
		#calculate errors for each degree
		bootstrap_MSE[i, degree] = np.mean(mse)
	i += 1

cv_score = np.zeros((len(models), max_degree))

i= 0
for model in models:
	for degree in tqdm(range(max_degree)):
		X = create_X(x, y, degree)

		cv_score[i, degree] = -np.mean(cross_val_score(model, X, z.reshape(-1, 1), scoring='neg_mean_squared_error', cv=KFold(10)))
	i += 1

fig, ax = plt.subplots()
ax.plot(range(max_degree), bootstrap_MSE[0, :], label='linreg')
ax.plot(range(max_degree), bootstrap_MSE[1, :], label='ridge')
ax.plot(range(max_degree), bootstrap_MSE[2, :], label='lasso')
ax.set_xlabel('Degree of polynomial')
ax.set_title('MSE for different regressiontechniques')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(range(max_degree), cv_score[0, :], label='linreg')
ax.plot(range(max_degree), cv_score[1, :], label='ridge')
ax.plot(range(max_degree), cv_score[2, :], label='lasso')
ax.set_xlabel('log10(lambda)')
ax.set_xlabel('Degree of polynomial')
ax.set_title('MSE for different regressiontechniques')
ax.legend()
plt.show()