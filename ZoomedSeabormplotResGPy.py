import GPy
import deepgp
import numpy as np
import torch
import pickle
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
sns.set()
sns.set_style("whitegrid")
sns.set_context("paper")
colors = sns.color_palette("tab10")
sns.set_palette(colors)
plt.rcParams["figure.figsize"] = [26, 4]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['legend.loc'] = "upper right"

SMALL_SIZE = 8
MEDIUM_SIZE = 20
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


########################STATISTICS##############################
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
    return 0.5*((np.log(std**2)+((y-mean)/(std))**2).sum()) 

#Define the path to the training data
validation_data_path15 = './20201030bag15.pickled'
validation_data_path3  = './20201030bag3.pickled'
training_data_path     = './final_training_data.pickled'

#Loading Traning Data
training_data = pickle.load(open(validation_data_path15, "rb"))
x_train = training_data['features']
y_train = training_data['labels']
train_Fx = y_train[:,0].reshape(-1,1)
train_Fy = y_train[:,1].reshape(-1,1)
train_Fz = y_train[:,2].reshape(-1,1)




#Load The test Datasett

test_data_3 = pickle.load(open(validation_data_path3, "rb"))
x_test_3 = test_data_3['features']
y_test_3 = test_data_3['labels']
time = test_data_3['labels_timestamp']
test_Fx = y_test_3[:,0].squeeze()
test_Fy = y_test_3[:,1].squeeze()
test_Fz = y_test_3[:,2].squeeze()

preds = torch.load('./PredictionVectorsRBFDeep/preds3.pt') 
std = torch.load('./PredictionVectorsRBFDeep/std3.pt') 

mean_Fx = preds[0,:].squeeze()
mean_Fy = preds[1,:].squeeze()
mean_Fz = preds[2,:].squeeze()
std_Fx = std[0,:].squeeze()
std_Fy = std[1,:].squeeze()
std_Fz = std[2,:].squeeze()

GP_pred = np.genfromtxt("../BaselineResults/GP_baseline_bag3.csv", delimiter=',')
baseline_time = np.genfromtxt("../BaselineResults/GP_x_axis_bag3.csv", delimiter=',')

idx_start_baseline = np.argwhere(baseline_time > time[0])[0].item()
idx_stop_baseline = idx_start_baseline + len(time)
print('idx_star_baseline',idx_start_baseline,'-------',idx_stop_baseline)

baseline_time = baseline_time[idx_start_baseline:idx_stop_baseline]

baseline_Fx = GP_pred[idx_start_baseline:idx_stop_baseline,0]
baseline_Fy = GP_pred[idx_start_baseline:idx_stop_baseline,1]
baseline_Fz = GP_pred[idx_start_baseline:idx_stop_baseline,2]


 

print('Test Shape',test_Fx.shape)
print('Pred Shape',mean_Fx.shape)
print('Std Shape',std_Fx.shape)


print('\n')
print('-------------------STATISTICS ON TEST SET BAG3--------------------')
print('\n-------Force in X -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fx,mean_Fx))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fx,mean_Fx,std_Fx))
print('\n')
print('-------Force in Y -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fy,mean_Fy))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fy,mean_Fy,std_Fy))
print('\n')
print('\n')
print('-------Force in Z -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fz,mean_Fz))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fz,mean_Fz,std_Fz))
print('\n')
print('--------------------------------------------------')


from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 
def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)
    
    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)


    [axins.spines[i].set_linewidth(2) for i in axins.spines]
    [axins.spines[i].set_color('black') for i in axins.spines]

    return pp, p1, p2

########################SUBPLOT ZOOM#############################
fig , (ax1,axins) = plt.subplots(2,1,figsize = (16,9))
#time = np.linspace(0,len(x_test_3),len(x_test_3))
#xlim = None 
#plt.suptitle('DGP - Predictions on the TEST set BAG3')
#Plot predictions in Force X 
ax1.set_title('Residual Torque in X direction')
ax1.plot(time,mean_Fx,label='Prediction',linewidth = 2)
ax1.plot(time,test_Fx,label = 'True_output',linewidth = 2)
ax1.plot(baseline_time,baseline_Fx,label = 'Baseline',linewidth = 2)
ax1.fill_between(time,mean_Fx-3*std_Fx,mean_Fx+3*std_Fx, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\Delta T_x$ [Nm]')
ax1.legend()
axins.plot(time,mean_Fx,label='Prediction',linewidth = 3)
axins.plot(time,test_Fx,label = 'True_output',linewidth = 3)
axins.plot(baseline_time,baseline_Fx,label = 'Baseline',linewidth = 3)
x1, x2, y1, y2 = 16, 19, 0, 0.4 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

plt.xticks(visible=True)
plt.yticks(visible=True)

mark_inset(ax1, axins, loc1a=2, loc1b=3, loc2a=1, loc2b=4, fc="black", ec="black",linewidth = 2)
########################PRERDICTTION##############################
fig , ax1 = plt.subplots(1,1)
#time = np.linspace(0,len(x_test_3),len(x_test_3))
#xlim = None 
#plt.suptitle('DGP - Predictions on the TEST set BAG3')
#Plot predictions in Force X 
ax1.set_title('Residual Torque in X direction')
ax1.plot(time,mean_Fx,label='Prediction',linewidth = 2)
ax1.plot(time,test_Fx,label = 'True_output',linewidth = 2)
ax1.plot(baseline_time,baseline_Fx,label = 'Baseline',linewidth = 2)
ax1.fill_between(time,mean_Fx-3*std_Fx,mean_Fx+3*std_Fx, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\Delta T_x$ [Nm]')
ax1.legend()
axins = zoomed_inset_axes(ax1,2.5,
                   bbox_to_anchor=(0, .6, .5, .4),
                   bbox_transform=ax1.transAxes, loc=2, borderpad=0)
axins.plot(time,mean_Fx,label='Prediction',linewidth = 3)
axins.plot(time,test_Fx,label = 'True_output',linewidth = 3)
axins.plot(baseline_time,baseline_Fx,label = 'Baseline',linewidth = 3)
x1, x2, y1, y2 = 17, 19, 0, 0.4 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax1, axins, loc1a=1, loc1b=2, loc2a=4, loc2b=3, fc="black", ec="black",linewidth = 2)
fig , ax2 = plt.subplots(1,1)
#Plot predictions in Fy axis 
ax2.set_title('Residual Torque in Y direction')
ax2.plot(time,mean_Fy,label='Prediction',linewidth = 2)
ax2.plot(time,test_Fy,label = 'True_output',linewidth = 2)
ax2.plot(baseline_time,baseline_Fy,label = 'Baseline',linewidth = 2)
ax2.fill_between(time,mean_Fy-3*std_Fy,mean_Fy+3*std_Fy, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel(r'$\Delta T_y$ [Nm]')
ax2.legend()
axins = zoomed_inset_axes(ax2,2.5,
                   bbox_to_anchor=(0, .6, .5, .4),
                   bbox_transform=ax2.transAxes, loc=2, borderpad=0)
axins.plot(time,mean_Fy,label='Prediction',linewidth = 3)
axins.plot(time,test_Fy,label = 'True_output',linewidth = 3)
axins.plot(baseline_time,baseline_Fy,label = 'Baseline',linewidth = 3)
x1, x2, y1, y2 = 14.5, 16.5, -0.4, -0 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax2, axins, loc1a=1, loc1b=2, loc2a=4, loc2b=4, fc="black", ec="black",linewidth = 2)

fig , ax3 = plt.subplots(1,1)
#Plot predictions in Fz axis 
ax3.set_title('Residual Torque in Z direction')
ax3.plot(time,mean_Fz,label='Prediction',linewidth = 2)
ax3.plot(time,test_Fz,label = 'True_output',linewidth = 2)
ax3.plot(baseline_time,baseline_Fz,label = 'Baseline',linewidth = 2)
ax3.fill_between(time,mean_Fz-3*std_Fz,mean_Fz+3*std_Fz, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel(r'$\Delta T_z$ [Nm]')
ax3.legend()
axins = zoomed_inset_axes(ax3,2.5,
                   bbox_to_anchor=(0, .6, .5, .4),
                   bbox_transform=ax3.transAxes, loc=2, borderpad=0)
axins.plot(time,mean_Fz,label='Prediction',linewidth = 3)
axins.plot(time,test_Fz,label = 'True_output',linewidth = 3)
axins.plot(baseline_time,baseline_Fz,label = 'Baseline',linewidth = 3)
x1, x2, y1, y2 = 22.5, 24, -0.4, 0.3 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax3, axins, loc1a=1, loc1b=2, loc2a=4, loc2b=3, fc="black", ec="black",linewidth = 2)
plt.draw()
print('First Plot created')

########################UCNCERTAINTY ERROR##############################

fig2 , ax11 = plt.subplots(1,1)
plt.suptitle('Error Uncertainty on the TEST set BAG3')
ax11.set_title(r'Error and Uncertainty $\Delta T_x$')
ax11.plot(time,np.abs(mean_Fx-test_Fx),label='Error',linewidth = 1)
ax11.fill_between(time,np.zeros(len(time)),3*std_Fx, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax11.set_xlabel('Time [s]')
ax11.set_ylabel('Prediction Error [Nm]')
ax11.legend()

fig2 , ax22 = plt.subplots(1,1)
ax22.set_title(r'Error and Uncertainty $\Delta T_y$')
ax22.plot(time,np.abs(mean_Fy-test_Fy),label='Error',linewidth = 1 )
ax22.fill_between(time,np.zeros(len(time)),3*std_Fy, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax22.set_xlabel('Time [s]')
ax22.set_ylabel('Prediction Error [Nm]')
ax22.legend()

fig2 , ax33 = plt.subplots(1,1)
ax33.set_title(r'Error and Uncertainty $\Delta T_z$')
ax33.plot(time,np.abs(mean_Fz-test_Fz),label='Error',linewidth = 1 )
ax33.fill_between(time,np.zeros(len(time)),3*std_Fz, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax33.set_xlabel('Time [s]')
ax33.set_ylabel('Prediction Error [Nm]')
ax33.legend()

plt.draw()

###### Plot the uncertainty heatmaps
labels_batch = np.array([test_Fx,test_Fy,test_Fz]).T
output_batch_mean = np.array([mean_Fx.numpy(),mean_Fy.numpy(),mean_Fz.numpy()]).T
output_batch_var = np.array([std_Fx.numpy()**2,std_Fy.numpy()**2,std_Fz.numpy()**2]).T


print('Labels bathc',labels_batch.shape)
print('Output batch mean',output_batch_var.shape)
print('Output batch var',output_batch_mean.shape)

error_np = np.absolute(labels_batch-output_batch_mean)
print('error np',error_np.shape)

min_error = np.zeros(3)
for ii in range(3):
    min_error[ii] = np.min(error_np[:,ii])

max_error = np.zeros(3)
for jj in range(3):
    max_error[jj] = np.max(error_np[:,jj])

error = pd.DataFrame(error_np)

var_np = output_batch_var
var = pd.DataFrame(var_np)
min_var = np.zeros(3)
for kk in range(3):
    min_var[kk] = np.min(var_np[:,kk])

max_var = np.zeros(3)
for ll in range(3):
    max_var[ll] = np.max(var_np[:,ll])
    
print('var np',var_np.shape)

error_bins_x = pd.cut(error.iloc[:,0], bins=10)
error_bins_y = pd.cut(error.iloc[:,1], bins=10)
error_bins_z = pd.cut(error.iloc[:,2], bins=10)

var_bins_x = pd.cut(var.iloc[:,0], bins=10)
var_bins_y = pd.cut(var.iloc[:,1], bins=10)
var_bins_z = pd.cut(var.iloc[:,2], bins=10)

error_counts_x = error_bins_x.value_counts().to_numpy()
#print(error_counts_x)

tuples_x = pd.DataFrame(dict(error = error.iloc[:,0],
                             variance = var.iloc[:,0]))
tuples_y = pd.DataFrame(dict(error = error.iloc[:,1],
                             variance = var.iloc[:,1]))
tuples_z = pd.DataFrame(dict(error = error.iloc[:,2],
                             variance = var.iloc[:,2]))

num_bins_error = 40
num_ticks_error = 40
#default definition
num_bins_error = 40
num_bins_var = 30

#define which extracts of which axis you want to use
extract_torque_x_error = 1
extract_torque_x_var = 1
extract_torque_y_error = 1
extract_torque_y_var = 1
extract_torque_z_error = 1
extract_torque_z_var = 1

#recalcuation of resolution
num_bins_error_x = int(np.ceil(num_bins_error/extract_torque_x_error))
num_bins_error_y = int(np.ceil(num_bins_error/extract_torque_y_error))
num_bins_error_z = int(np.ceil(num_bins_error/extract_torque_z_error))
num_bins_var_x = int(np.ceil(num_bins_var/extract_torque_x_var))
num_bins_var_y = int(np.ceil(num_bins_var/extract_torque_y_var))
num_bins_var_z = int(np.ceil(num_bins_var/extract_torque_z_var))


round_factor = 4
error_ticks_x = np.linspace(min_error[0],max_error[0],num_bins_error)
error_ticks_x = np.round(error_ticks_x,round_factor)
error_ticks_y = np.linspace(min_error[1],max_error[1],num_bins_error)
error_ticks_y = np.round(error_ticks_y,round_factor)
error_ticks_z = np.linspace(min_error[2],max_error[2],num_bins_error)
error_ticks_z = np.round(error_ticks_z,round_factor)

var_ticks_x = np.linspace(min_var[0],max_var[0],num_bins_var)
var_ticks_x = np.round(var_ticks_x,round_factor)
var_ticks_y = np.linspace(min_var[1],max_var[1],num_bins_var)
var_ticks_y = np.round(var_ticks_y,round_factor)
var_ticks_z = np.linspace(min_var[2],max_var[2],num_bins_var)
var_ticks_z = np.round(var_ticks_z,round_factor)

labels_error_x = np.arange(1,num_bins_error_x+1)
labels_var_x = np.arange(1,num_bins_var_x+1)
labels_error_y = np.arange(1,num_bins_error_y+1)
labels_var_y = np.arange(1,num_bins_var_y+1)
labels_error_z = np.arange(1,num_bins_error_z+1)
labels_var_z = np.arange(1,num_bins_var_z+1)

tuples_x_binned = tuples_x.assign(error_binned = pd.cut(tuples_x.error, bins=num_bins_error_x, labels=labels_error_x), 
                                  var_binned = pd.cut(tuples_x.variance, bins=num_bins_var_x, labels=labels_var_x))
tuples_x_binned = tuples_x_binned.assign(cartesian=pd.Categorical(
    tuples_x_binned.filter(regex='_binned').apply(tuple, 1)))

tuples_y_binned = tuples_y.assign(error_binned = pd.cut(tuples_y.error, bins=num_bins_error_y, labels=labels_error_y), 
                                  var_binned = pd.cut(tuples_y.variance, bins=num_bins_var_y, labels=labels_var_y))
tuples_y_binned = tuples_y_binned.assign(cartesian=pd.Categorical(
    tuples_y_binned.filter(regex='_binned').apply(tuple, 1)))

tuples_z_binned = tuples_z.assign(error_binned = pd.cut(tuples_z.error, bins=num_bins_error_z, labels=labels_error_z), 
                                  var_binned = pd.cut(tuples_z.variance, bins=num_bins_var_z, labels=labels_var_z))
tuples_z_binned = tuples_z_binned.assign(cartesian=pd.Categorical(
    tuples_z_binned.filter(regex='_binned').apply(tuple, 1)))

max_x_x = 30
max_y_x = 40

counts_x = tuples_x_binned['cartesian'].value_counts()
counts_x_max = np.max(counts_x)
hmap_x = np.zeros((num_bins_error_x,num_bins_var_x))
for kk in range(counts_x.shape[0]):
    coord = counts_x.index[kk]
    value = counts_x.iloc[kk]/counts_x_max
    hmap_x[coord[0]-1,coord[1]-1]=value
    
hmap_x = hmap_x[:max_y_x,:max_x_x]
new_max = np.max(hmap_x)
hmap_x = hmap_x/new_max
error_ticks_x = np.flip(error_ticks_x)
hmap_x = np.flip(hmap_x,axis=0)
hmap_x = pd.DataFrame(hmap_x, columns=var_ticks_x[:max_x_x], index=error_ticks_x[:max_y_x])
    
max_x_y = 30
max_y_y = 40

counts_y = tuples_y_binned['cartesian'].value_counts()
counts_y_max = np.max(counts_y)
hmap_y = np.zeros((num_bins_error_y,num_bins_var_y))
for kk in range(counts_y.shape[0]):
    coord = counts_y.index[kk]
    value = counts_y.iloc[kk]/counts_y_max
    hmap_y[coord[0]-1,coord[1]-1]=value
    
hmap_y = hmap_y[:max_y_y,:max_x_y]
new_max = np.max(hmap_y)
hmap_y = hmap_y/new_max
error_ticks_y = np.flip(error_ticks_y)
hmap_y = np.flip(hmap_y,axis=0)
hmap_y = pd.DataFrame(hmap_y, columns=var_ticks_y[:max_x_y], index=error_ticks_y[:max_y_y])

max_x_z = 30
max_y_z = 40    

counts_z = tuples_z_binned['cartesian'].value_counts()
counts_z_max = np.max(counts_z)
hmap_z = np.zeros((num_bins_error_z,num_bins_var_z))
for kk in range(counts_z.shape[0]):
    coord = counts_z.index[kk]
    value = counts_z.iloc[kk]/counts_z_max
    hmap_z[coord[0]-1,coord[1]-1]=value
    
hmap_z = hmap_z[:max_y_z,:max_x_z]
new_max = np.max(hmap_z)
hmap_z = hmap_z/new_max
error_ticks_z = np.flip(error_ticks_z)
hmap_z = np.flip(hmap_z,axis=0)
hmap_z = pd.DataFrame(hmap_z, columns=var_ticks_z[:max_x_z], index=error_ticks_z[:max_y_z])

x_ticks = 4
y_ticks = 3

fig , ax1 = plt.subplots(1,1,figsize=(10,9))       # Sample figsize in inches
plt.title('Error - Var Heatmap Torque X')
ax1 = sns.heatmap(hmap_x, annot=False, xticklabels=x_ticks, yticklabels=y_ticks,ax=ax1,linewidth=0,rasterized=True)
plt.setp(ax1.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.setp(ax1.get_yticklabels(), rotation=0, horizontalalignment='right')
ax1.set_xlabel('Variance')
ax1.set_ylabel('Error')
#plt.setp(ax1.get_yticks(), rotation=90)
#ax1.set_yticklabels(rotation = 90)
#ax1.savefig('heapmap_torque_x.png')

fig , ax2 = plt.subplots(1,1,figsize=(10,9))  
plt.title('Error - Var Heatmap Torque Y')
ax2 = sns.heatmap(hmap_y, annot=False, xticklabels=x_ticks, yticklabels=y_ticks,ax=ax2,linewidth=0,rasterized=True)
plt.setp(ax2.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.setp(ax2.get_yticklabels(), rotation=0, horizontalalignment='right')
ax2.set_xlabel('Variance')
ax2.set_ylabel('Error')
#ax2.set_yticklabels(rotation = 90)
#ax2.savefig('heapmap_torque_x.png')

fig , ax3 = plt.subplots(1,1,figsize=(10,9))  
plt.title('Error - Var Heatmap Torque Z')
ax3 = sns.heatmap(hmap_z, annot=False, xticklabels=x_ticks, yticklabels=y_ticks,ax=ax3,linewidth=0,rasterized=True)
plt.setp(ax3.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.setp(ax3.get_yticklabels(), rotation=0, horizontalalignment='right')
ax3.set_xlabel('Variance')
ax3.set_ylabel('Error')
#ax3.set_yticklabels(rotation = 90)
#ax3.savefig('heapmap_torque_x.png')
plt.draw()


test_data_15 = pickle.load(open(validation_data_path15, "rb"))
x_test_15 = test_data_15['features']
y_test_15 = test_data_15['labels']
time = test_data_15['labels_timestamp']
test_Fx = y_test_15[:,0].squeeze()
test_Fy = y_test_15[:,1].squeeze()
test_Fz = y_test_15[:,2].squeeze()

#Load predictions
preds = torch.load('./PredictionVectorsRBFDeep/preds15.pt') 
std = torch.load('./PredictionVectorsRBFDeep/std15.pt') 

mean_Fx = preds[0,:].squeeze()
mean_Fy = preds[1,:].squeeze()
mean_Fz = preds[2,:].squeeze()
std_Fx = std[0,:].squeeze()
std_Fy = std[1,:].squeeze()
std_Fz = std[2,:].squeeze()

#Loadd Baseline 
GP_pred = np.genfromtxt("../BaselineResults/GP_baseline_bag15.csv", delimiter=',')
baseline_time = np.genfromtxt("../BaselineResults/GP_x_axis_bag15.csv", delimiter=',')


idx_start_baseline = np.argwhere(baseline_time > time[0])[0].item()
idx_stop_baseline = idx_start_baseline + len(time)

baseline_time = baseline_time[idx_start_baseline:idx_stop_baseline]

baseline_Fx = GP_pred[idx_start_baseline:idx_stop_baseline,0]
baseline_Fy = GP_pred[idx_start_baseline:idx_stop_baseline,1]
baseline_Fz = GP_pred[idx_start_baseline:idx_stop_baseline,2]


print('Test Shape',test_Fx.shape)
print('Pred Shape',mean_Fx.shape)
print('Std Shape',std_Fx.shape)


print('\n')
print('-------------------STATISTICS ON TEST SET BAG15--------------------')
print('\n-------Force in X -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fx,mean_Fx))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fx,mean_Fx,std_Fx))
print('\n')
print('-------Force in Y -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fy,mean_Fy))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fy,mean_Fy,std_Fy))
print('\n')
print('\n')
print('-------Force in Z -----')
print('\n \n Root Mean Squared Error (RMSE)', RMSE(test_Fz,mean_Fz))
print('\n \n Negative Log Likelihood (NLL)', NLL(test_Fz,mean_Fz,std_Fz))
print('\n')
print('--------------------------------------------------')

fig , ax1 = plt.subplots(1,1)
#xlim = None 
plt.suptitle('Predictions on the TEST set BAG15')
#Plot predictions in Force X 
ax1.set_title('Residual Torque in X direction')
ax1.plot(time,mean_Fx,label='Prediction',linewidth = 1 )
ax1.plot(time,test_Fx,label = 'True_output',linewidth = 1 )
ax1.plot(baseline_time,baseline_Fx,label = 'Baseline',linewidth = 1)
ax1.fill_between(time,mean_Fx-3*std_Fx,mean_Fx+3*std_Fx, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$\Delta T_x $ [Nm]')
ax1.legend()

fig , ax2 = plt.subplots(1,1)
#Plot predictions in Fy axis 
ax2.set_title('Residual Torque in Y direction')
ax2.plot(time,mean_Fy,label='Prediction',linewidth = 1 )
ax2.plot(time,test_Fy,label = 'True_output',linewidth = 1 )
ax2.plot(baseline_time,baseline_Fy,label = 'Baseline',linewidth = 1)
ax2.fill_between(time,mean_Fy-3*std_Fy,mean_Fy+3*std_Fy, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel(r'$\Delta T_y $ [Nm]')
ax2.legend()


fig , ax3 = plt.subplots(1,1)
#Plot predictions in Fz axis 
ax3.set_title('Residual Torque in Z direction')
ax3.plot(time,mean_Fz,label='Prediction',linewidth = 1 )
ax3.plot(time,test_Fz,label = 'True_output',linewidth = 1 )
ax3.plot(baseline_time,baseline_Fz,label = 'Baseline',linewidth = 1)
ax3.fill_between(time,mean_Fz-3*std_Fz,mean_Fz+3*std_Fz, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel(r'$\Delta T_z $ [Nm]')
ax3.legend()

plt.draw()
print('First Plot created')

#PLOT UNCERTAINTY AND ERROR 

fig2 , ax11 = plt.subplots(1,1)
plt.suptitle('Predictions on the TEST set BAG15')

ax11.set_title(r'Error and Uncertainty $\Delta T_x$')
ax11.plot(time,np.abs(mean_Fx-test_Fx),label='Error',linewidth = 1)
ax11.fill_between(time,np.zeros(len(time)),3*std_Fx, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax11.set_xlabel('Time [s]')
ax11.set_ylabel('Prediction Error [Nm]')
ax11.legend()

fig2 , ax22 = plt.subplots(1,1)
ax22.set_title(r'Error and Uncertainty $\Delta T_y$')
ax22.plot(time,np.abs(mean_Fy-test_Fy),label='Error',linewidth = 1 )
ax22.fill_between(time,np.zeros(len(time)),3*std_Fy, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax22.set_xlabel('Time [s]')
ax22.set_ylabel('Prediction Error [Nm]')
ax22.legend()

fig2 , ax33 = plt.subplots(1,1)
ax33.set_title(r'Error and Uncertainty $\Delta T_z$')
ax33.plot(time,np.abs(mean_Fz-test_Fz),label='Error',linewidth = 1 )
ax33.fill_between(time,np.zeros(len(time)),3*std_Fz, alpha=0.2,label='Confidence Bounds 3std',edgecolor = 'black')
ax33.set_xlabel('Time [s]')
ax33.set_ylabel('Prediction Error [Nm]')
ax33.legend()

plt.draw()


###### Plot the uncertainty heatmaps
labels_batch = np.array([test_Fx,test_Fy,test_Fz]).T
output_batch_mean = np.array([mean_Fx.numpy(),mean_Fy.numpy(),mean_Fz.numpy()]).T
output_batch_var = np.array([std_Fx.numpy()**2,std_Fy.numpy()**2,std_Fz.numpy()**2]).T


print('Labels bathc',labels_batch.shape)
print('Output batch mean',output_batch_var.shape)
print('Output batch var',output_batch_mean.shape)

error_np = np.absolute(labels_batch-output_batch_mean)
print('error np',error_np.shape)

min_error = np.zeros(3)
for ii in range(3):
    min_error[ii] = np.min(error_np[:,ii])

max_error = np.zeros(3)
for jj in range(3):
    max_error[jj] = np.max(error_np[:,jj])

error = pd.DataFrame(error_np)

var_np = output_batch_var
var = pd.DataFrame(var_np)
min_var = np.zeros(3)
for kk in range(3):
    min_var[kk] = np.min(var_np[:,kk])

max_var = np.zeros(3)
for ll in range(3):
    max_var[ll] = np.max(var_np[:,ll])
    
print('var np',var_np.shape)

error_bins_x = pd.cut(error.iloc[:,0], bins=10)
error_bins_y = pd.cut(error.iloc[:,1], bins=10)
error_bins_z = pd.cut(error.iloc[:,2], bins=10)

var_bins_x = pd.cut(var.iloc[:,0], bins=10)
var_bins_y = pd.cut(var.iloc[:,1], bins=10)
var_bins_z = pd.cut(var.iloc[:,2], bins=10)

error_counts_x = error_bins_x.value_counts().to_numpy()
#print(error_counts_x)

tuples_x = pd.DataFrame(dict(error = error.iloc[:,0],
                             variance = var.iloc[:,0]))
tuples_y = pd.DataFrame(dict(error = error.iloc[:,1],
                             variance = var.iloc[:,1]))
tuples_z = pd.DataFrame(dict(error = error.iloc[:,2],
                             variance = var.iloc[:,2]))

num_bins_error = 40
num_ticks_error = 40
#default definition
num_bins_error = 40
num_bins_var = 30

#define which extracts of which axis you want to use
extract_torque_x_error = 1
extract_torque_x_var = 1
extract_torque_y_error = 1
extract_torque_y_var = 1
extract_torque_z_error = 1
extract_torque_z_var = 1

#recalcuation of resolution
num_bins_error_x = int(np.ceil(num_bins_error/extract_torque_x_error))
num_bins_error_y = int(np.ceil(num_bins_error/extract_torque_y_error))
num_bins_error_z = int(np.ceil(num_bins_error/extract_torque_z_error))
num_bins_var_x = int(np.ceil(num_bins_var/extract_torque_x_var))
num_bins_var_y = int(np.ceil(num_bins_var/extract_torque_y_var))
num_bins_var_z = int(np.ceil(num_bins_var/extract_torque_z_var))


round_factor = 4
error_ticks_x = np.linspace(min_error[0],max_error[0],num_bins_error)
error_ticks_x = np.round(error_ticks_x,round_factor)
error_ticks_y = np.linspace(min_error[1],max_error[1],num_bins_error)
error_ticks_y = np.round(error_ticks_y,round_factor)
error_ticks_z = np.linspace(min_error[2],max_error[2],num_bins_error)
error_ticks_z = np.round(error_ticks_z,round_factor)

var_ticks_x = np.linspace(min_var[0],max_var[0],num_bins_var)
var_ticks_x = np.round(var_ticks_x,round_factor)
var_ticks_y = np.linspace(min_var[1],max_var[1],num_bins_var)
var_ticks_y = np.round(var_ticks_y,round_factor)
var_ticks_z = np.linspace(min_var[2],max_var[2],num_bins_var)
var_ticks_z = np.round(var_ticks_z,round_factor)

labels_error_x = np.arange(1,num_bins_error_x+1)
labels_var_x = np.arange(1,num_bins_var_x+1)
labels_error_y = np.arange(1,num_bins_error_y+1)
labels_var_y = np.arange(1,num_bins_var_y+1)
labels_error_z = np.arange(1,num_bins_error_z+1)
labels_var_z = np.arange(1,num_bins_var_z+1)

tuples_x_binned = tuples_x.assign(error_binned = pd.cut(tuples_x.error, bins=num_bins_error_x, labels=labels_error_x), 
                                  var_binned = pd.cut(tuples_x.variance, bins=num_bins_var_x, labels=labels_var_x))
tuples_x_binned = tuples_x_binned.assign(cartesian=pd.Categorical(
    tuples_x_binned.filter(regex='_binned').apply(tuple, 1)))

tuples_y_binned = tuples_y.assign(error_binned = pd.cut(tuples_y.error, bins=num_bins_error_y, labels=labels_error_y), 
                                  var_binned = pd.cut(tuples_y.variance, bins=num_bins_var_y, labels=labels_var_y))
tuples_y_binned = tuples_y_binned.assign(cartesian=pd.Categorical(
    tuples_y_binned.filter(regex='_binned').apply(tuple, 1)))

tuples_z_binned = tuples_z.assign(error_binned = pd.cut(tuples_z.error, bins=num_bins_error_z, labels=labels_error_z), 
                                  var_binned = pd.cut(tuples_z.variance, bins=num_bins_var_z, labels=labels_var_z))
tuples_z_binned = tuples_z_binned.assign(cartesian=pd.Categorical(
    tuples_z_binned.filter(regex='_binned').apply(tuple, 1)))

max_x_x = 30
max_y_x = 40

counts_x = tuples_x_binned['cartesian'].value_counts()
counts_x_max = np.max(counts_x)
hmap_x = np.zeros((num_bins_error_x,num_bins_var_x))
for kk in range(counts_x.shape[0]):
    coord = counts_x.index[kk]
    value = counts_x.iloc[kk]/counts_x_max
    hmap_x[coord[0]-1,coord[1]-1]=value
    
hmap_x = hmap_x[:max_y_x,:max_x_x]
new_max = np.max(hmap_x)
hmap_x = hmap_x/new_max
error_ticks_x = np.flip(error_ticks_x)
hmap_x = np.flip(hmap_x,axis=0)
hmap_x = pd.DataFrame(hmap_x, columns=var_ticks_x[:max_x_x], index=error_ticks_x[:max_y_x])
    
max_x_y = 30
max_y_y = 40

counts_y = tuples_y_binned['cartesian'].value_counts()
counts_y_max = np.max(counts_y)
hmap_y = np.zeros((num_bins_error_y,num_bins_var_y))
for kk in range(counts_y.shape[0]):
    coord = counts_y.index[kk]
    value = counts_y.iloc[kk]/counts_y_max
    hmap_y[coord[0]-1,coord[1]-1]=value
    
hmap_y = hmap_y[:max_y_y,:max_x_y]
new_max = np.max(hmap_y)
hmap_y = hmap_y/new_max
error_ticks_y = np.flip(error_ticks_y)
hmap_y = np.flip(hmap_y,axis=0)
hmap_y = pd.DataFrame(hmap_y, columns=var_ticks_y[:max_x_y], index=error_ticks_y[:max_y_y])

max_x_z = 30
max_y_z = 40    

counts_z = tuples_z_binned['cartesian'].value_counts()
counts_z_max = np.max(counts_z)
hmap_z = np.zeros((num_bins_error_z,num_bins_var_z))
for kk in range(counts_z.shape[0]):
    coord = counts_z.index[kk]
    value = counts_z.iloc[kk]/counts_z_max
    hmap_z[coord[0]-1,coord[1]-1]=value
    
hmap_z = hmap_z[:max_y_z,:max_x_z]
new_max = np.max(hmap_z)
hmap_z = hmap_z/new_max
error_ticks_z = np.flip(error_ticks_z)
hmap_z = np.flip(hmap_z,axis=0)
hmap_z = pd.DataFrame(hmap_z, columns=var_ticks_z[:max_x_z], index=error_ticks_z[:max_y_z])


x_ticks = 4
y_ticks = 3

fig , ax1 = plt.subplots(1,1,figsize=(10,9))       # Sample figsize in inches
plt.title('Error - Var Heatmap Torque X')
ax1 = sns.heatmap(hmap_x, annot=False, xticklabels=x_ticks, yticklabels=y_ticks,ax=ax1,linewidth=0,rasterized=True)
plt.setp(ax1.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.setp(ax1.get_yticklabels(), rotation=0, horizontalalignment='right')
ax1.set_xlabel('Variance')
ax1.set_ylabel('Error')
#plt.setp(ax1.get_yticks(), rotation=90)
#ax1.set_yticklabels(rotation = 90)
#ax1.savefig('heapmap_torque_x.png')

fig , ax2 = plt.subplots(1,1,figsize=(10,9))  
plt.title('Error - Var Heatmap Torque Y')
ax2 = sns.heatmap(hmap_y, annot=False, xticklabels=x_ticks, yticklabels=y_ticks,ax=ax2,linewidth=0,rasterized=True)
plt.setp(ax2.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.setp(ax2.get_yticklabels(), rotation=0, horizontalalignment='right')
ax2.set_xlabel('Variance')
ax2.set_ylabel('Error')
#ax2.set_yticklabels(rotation = 90)
#ax2.savefig('heapmap_torque_x.png')

fig , ax3 = plt.subplots(1,1,figsize=(10,9))  
plt.title('Error - Var Heatmap Torque Z')
ax3 = sns.heatmap(hmap_z, annot=False, xticklabels=x_ticks, yticklabels=y_ticks,ax=ax3,linewidth=0,rasterized=True)
plt.setp(ax3.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.setp(ax3.get_yticklabels(), rotation=0, horizontalalignment='right')
ax3.set_xlabel('Variance')
ax3.set_ylabel('Error')
#ax3.set_yticklabels(rotation = 90)
#ax3.savefig('heapmap_torque_x.png')
plt.draw()

multipage('./ZoomedDeepGPs2Attempt.pdf')





