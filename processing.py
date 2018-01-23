import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.svm import SVR

#load raw data
train = np.genfromtxt('data/train.csv', delimiter=',')[1:,]
test = np.genfromtxt('data/test.csv', delimiter=',')[1:,]

#generating target values
train=np.hstack((train,np.zeros([train.shape[0],2])))
test=np.hstack((test,np.zeros([test.shape[0],2])))

for i in range(1,81):
	train[train[:,0]==i,-2]=(train[train[:,0]==i][:,-2]).shape[0]

for i in range(81,101):
	test[test[:,0]==i,-2]=(test[test[:,0]==i][:,-2]).shape[0]

train[:,-1]=train[:,-2]-train[:,1]
test[:,-1]=test[:,-2]-test[:,1]



#separating test and train data
xtrain=train[:,1:-2]
xtest=test[:,1:-2]
ytrain=train[:,-1]
ytest=test[:,-1]



#standardising data
scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain=scaler.transform(xtrain) 
xtest=scaler.transform(xtest) 



#removing cols with constant values
#ind 1 5 10 16 18 19 are const
xtrain=np.delete(xtrain,[1,5,10,16,18,19],axis=1)
xtest=np.delete(xtest,[1,5,10,16,18,19],axis=1)



#linear reg

reg = linear_model.LinearRegression()
reg.fit (xtrain,ytrain)


print 'coeffs:',reg.coef_
print'\n'
print 'train metrics:'
pred=reg.predict(xtrain)
print("Root mean squared error: %.2f"% mean_squared_error(ytrain, pred)**0.5) #36.59
print('R2 score: %.2f' % r2_score(ytrain, pred)) #0.69
print'\n'
print 'test metrics:'
pred=reg.predict(xtest)
print("Root mean squared error: %.2f"% mean_squared_error(ytest, pred)**0.5) #53.14
print('R2 score: %.2f' % r2_score(ytest, pred)) #0.53

#poly reg
poly = PolynomialFeatures(3)
x2train=poly.fit_transform(xtrain)
x2test=poly.fit_transform(xtest)

reg = linear_model.LinearRegression()
reg.fit (x2train,ytrain)


print 'coeffs:',reg.coef_
print'\n'
print 'train metrics:'
pred=reg.predict(x2train)
print("Root mean squared error: %.2f"% mean_squared_error(ytrain, pred)**0.5) #32.50
print('R2 score: %.2f' % r2_score(ytrain, pred)) #0.76
print'\n'
print 'test metrics:'
pred=reg.predict(x2test)
print("Root mean squared error: %.2f"% mean_squared_error(ytest, pred)**0.5) #48.40
print('R2 score: %.2f' % r2_score(ytest, pred)) #0.61

'''
degree testrmse testr2
2		48.40	0.61
3		48.69	0.61
4		7785938831.53 -10094840957074756.00

'''

#SVM
svr_rbf = SVR(kernel='rbf', C=10, gamma=0.01)
svr_rbf.fit(xtrain, ytrain)

print 'test metrics:'
pred=svr_rbf.predict(xtest)
print("Root mean squared error: %.2f"% mean_squared_error(ytest, pred)**0.5) #53.14
print('R2 score: %.2f' % r2_score(ytest, pred)) #0.53

for c in [0.1,1,10]:
	for g in [0.01,0.1,1,10]:
		svr_rbf = SVR(kernel='rbf', C=c, gamma=g)
		svr_rbf.fit(xtrain, ytrain)
		'''
		print 'train metrics:'
		pred=svr_rbf.predict(xtrain)
		print("Root mean squared error: %.2f"% mean_squared_error(ytrain, pred)**0.5)
		print('R2 score: %.2f' % r2_score(ytrain, pred)) 
		print'\n'
		'''
		print 'test metrics:'
		pred=svr_rbf.predict(xtest)
		print("Root mean squared error: %.2f"% mean_squared_error(ytest, pred)**0.5) 
		print('R2 score: %.2f' % r2_score(ytest, pred)) 
		print "C: ",c
		print "Gamma: ",g
		print '\n'

'''
C 		gamma 		testrmse	testr2
0.1		0.01		56.97		0.46
0.1		0.1			60.82		0.38
0.1		1			79.73		-0.06
0.1		10			79.87		-0.06
1		0.01		51.86		0.55
1		0.1			
1		1
1		10
10		0.01		51.02		0.57
10		0.1			51.17		0.56
10		1
10		10

'''

#poly reg SVM

poly = PolynomialFeatures(2)
x2train=poly.fit_transform(xtrain)
x2test=poly.fit_transform(xtest)

svr_rbf = SVR(kernel='rbf', C=1000, gamma=10)
svr_rbf.fit(x2train, ytrain)


print 'test metrics:'
pred=svr_rbf.predict(x2test)
print("Root mean squared error: %.2f"% mean_squared_error(ytest, pred)**0.5) #53.14
print('R2 score: %.2f' % r2_score(ytest, pred)) #0.53


poly = PolynomialFeatures(3)
x2train=poly.fit_transform(xtrain)
x2test=poly.fit_transform(xtest)

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit (x2train,ytrain)

coef=reg.coef_
print 'coeffs:',coef
print'\n'
print 'train metrics:'
pred=reg.predict(x2train)
print("Root mean squared error: %.2f"% mean_squared_error(ytrain, pred)**0.5) #32.50
print('R2 score: %.2f' % r2_score(ytrain, pred)) #0.76
print'\n'
print 'test metrics:'
pred=reg.predict(x2test)
print("Root mean squared error: %.2f"% mean_squared_error(ytest, pred)**0.5) #48.40
print('R2 score: %.2f' % r2_score(ytest, pred)) #0.61

reg.fit (x2train[:,coef>2],ytrain)
print 'coeffs:',reg.coef_
print'\n'
print 'train metrics:'
pred=reg.predict(x2train[:,coef>2])
print("Root mean squared error: %.2f"% mean_squared_error(ytrain, pred)**0.5) #32.50
print('R2 score: %.2f' % r2_score(ytrain, pred)) #0.76
print'\n'
print 'test metrics:'
pred=reg.predict(x2test[:,coef>2])
print("Root mean squared error: %.2f"% mean_squared_error(ytest, pred)**0.5) #48.40
print('R2 score: %.2f' % r2_score(ytest, pred)) #0.61


