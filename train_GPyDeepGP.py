import numpy as np 
import GPy
import deepgp
import math
import torch 
from sklearn.preprocessing import StandardScaler
import pickle 
import time

#Define the path to the training data 
validation_data_path15 = '../model_learning_data_pickled/20201030bag15.pickled'
validation_data_path3  = '../model_learning_data_pickled/20201030bag3.pickled'
training_data_path     = '../model_learning_data_pickled/final_training_data.pickled'

#Loading Traning Data
training_data = pickle.load(open(training_data_path, "rb"))
x_train = training_data['features']
y_train = training_data['labels']
train_Fx = y_train[:,0].reshape(-1,1)
train_Fy = y_train[:,1].reshape(-1,1)
train_Fz = y_train[:,2].reshape(-1,1)

#Load The test Datasett 
test_data_15 = pickle.load(open(validation_data_path15, "rb"))
x_test_15 = test_data_15['features']
y_test_15 = test_data_15['labels']
test_Fx_15 = y_test_15[:,0].reshape(-1,1)
test_Fy_15= y_test_15[:,1].reshape(-1,1)
test_Fz_15 = y_test_15[:,2].reshape(-1,1)
test_data_3 = pickle.load(open(validation_data_path3, "rb"))
x_test_3 = test_data_3['features']
y_test_3 = test_data_3['labels']
test_Fx_3 = y_test_3[:,0].reshape(-1,1)
test_Fy_3 = y_test_3[:,1].reshape(-1,1)
test_Fz_3 = y_test_3[:,2].reshape(-1,1)

#Define the deep GP model
layers = [train_Fx.shape[1], 1, 1, 1,x_train.shape[1]]
inits = ['PCA']*(len(layers)-1)
kernels = []

for i in layers[1:]:
    kernels += [GPy.kern.RBF(i,ARD=True)]

mx = deepgp.DeepGP(layers,Y=train_Fx, X=x_train, 
                  inits=inits, 
                  kernels=kernels, # the kernels for each layer
                  num_inducing=40, back_constraint=False)
my = deepgp.DeepGP(layers,Y=train_Fy, X=x_train, 
                  inits=inits, 
                  kernels=kernels, # the kernels for each layer
                  num_inducing=40, back_constraint=False)
mz = deepgp.DeepGP(layers,Y=train_Fz, X=x_train, 
                  inits=inits, 
                  kernels=kernels, # the kernels for each layer
                  num_inducing=40, back_constraint=False)

models_path = './'

#Train and Predict 

print('training in the x direction')
mx.initialize_parameter()
mx.optimize(messages=True,max_iters=0)
np.save('model_save_x.npy', mx.param_array)
print('Testing in x direction on validation_data_path15')
mean_Fx_15,varx_15 = mx.predict(x_test_15)
std_Fx_15 = np.sqrt(varx_15).squeeze()
mean_Fx_15 = mean_Fx_15.squeeze()
print('Testing in x direction on validation_data_path3')
mean_Fx_3,varx_3 = mx.predict(x_test_3)
std_Fx_3 = np.sqrt(varx_3).squeeze()
mean_Fx_3 = mean_Fx_3.squeeze()


print('training in the y direction')
my.initialize_parameter()
my.optimize(messages=True,max_iters=0)
np.save('model_save_y.npy', my.param_array)
print('Testing in x direction on validation_data_path15')
mean_Fy_15,vary_15 = my.predict(x_test_15)
std_Fy_15 = np.sqrt(vary_15).squeeze()
mean_Fy_15 = mean_Fy_15.squeeze()
print('Testing in x direction on validation_data_path3')
mean_Fy_3,vary_3 = my.predict(x_test_3)
std_Fy_3 = np.sqrt(vary_3).squeeze()
mean_Fy_3 = mean_Fy_3.squeeze()


print('training in the z direction')
mz.initialize_parameter()
mz.optimize(messages=True,max_iters=0)
np.save('model_save_z.npy', mz.param_array)
print('Testing in x direction on validation_data_path15')
mean_Fz_15,varz_15 = mz.predict(x_test_15)
std_Fz_15 = np.sqrt(varz_15).squeeze()
mean_Fz_15 = mean_Fz_15.squeeze()
print('Testing in x direction on validation_data_path3')
mean_Fz_3,varz_3 = mz.predict(x_test_3)
std_Fz_3 = np.sqrt(varz_3).squeeze()
mean_Fz_3 = mean_Fz_3.squeeze()


experiments = 10
execution_times = []
for i in range(experiments):
	idx = np.random.randint(0,4000)
	print('Index',idx)
	test_point = x_test_3[idx,:].reshape(-1,1)
	print('Test_point',test_point.shape)
	start = time.process_time()
	pred_x = mx.predict(test_point.T)
	pred_y = my.predict(test_point.T)
	pred_z = mz.predict(test_point.T)
	stop = time.process_time()
	execution_times.append(stop-start)

print('The execution times were:',execution_times)



"""

def RMSE(y,mean): 
    if type(y) == torch.Tensor:
        y = y.numpy()
    if type(mean) == torch.Tensor:
        mean = mean.numpy()
        
    return np.sqrt(((y-mean)**2).mean()).item()

def NLL(y,mean,std):
    if type(y) == torch.Tensor:
        y = y.numpy()
    if type(mean) == torch.Tensor:
        mean = mean.numpy()
    if type(std) == torch.Tensor :
        std = std.numpy()
    return -0.5*((np.log(std**2)+((y-mean)/(std))**2).sum())


preds15 = torch.Tensor([mean_Fx_15,mean_Fy_15,mean_Fz_15])
std15 = torch.Tensor([std_Fx_15,std_Fy_15,std_Fz_15]) 
preds3 = torch.Tensor([mean_Fx_3,mean_Fy_3,mean_Fz_3])
std3 = torch.Tensor([std_Fx_3,std_Fy_3,std_Fz_3])

predpath = './PredictionVectors/'
torch.save(preds15,predpath + 'preds15.pt')
torch.save(std15,predpath + 'std15.pt')
torch.save(preds3,predpath + 'preds3.pt')
torch.save(std3,predpath + 'std3.pt')


#Compute statistics 
print('\n')
print('-------------------STATISTICS ON TEST 15 --------------------')
print('\n-------Force in X -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fx_15,mean_Fx_15))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fx_15,mean_Fx_15,std_Fx_15))
print('\n')
print('-------Force in Y -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fy_15,mean_Fy_15))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fy_15,mean_Fy_15,std_Fy_15))
print('\n')
print('\n')
print('-------Force in Z -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fz_15,mean_Fz_15))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fz_15,mean_Fz_15,std_Fz_15))
print('\n')
print('--------------------------------------------------')


print('\n')
print('-------------------STATISTICS ON TEST 3--------------------')
print('\n-------Force in X -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fx_3,mean_Fx_3))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fx_3,mean_Fx_3,std_Fx_3))
print('\n')
print('-------Force in Y -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fy_3,mean_Fy_3))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fy_3,mean_Fy_3,std_Fy_3))
print('\n')
print('\n')
print('-------Force in Z -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fz_3,mean_Fz_3))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fz_3,mean_Fz_3,std_Fz_3))
print('\n')
print('--------------------------------------------------')



print('\n')
print('-------------------STATISTICS ON TEST 3--------------------')
print('\n-------Force in X -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fx_3,mean_Fx_3))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fx_3,mean_Fx_3,std_Fx_3))
print('\n')
print('-------Force in Y -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fy_3,mean_Fy_3))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fy_3,mean_Fy_3,std_Fy_3))
print('\n')
print('\n')
print('-------Force in Z -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fz_3,mean_Fz_3))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fz_3,mean_Fz_3,std_Fz_3))
print('\n')
print('--------------------------------------------------')


"""
