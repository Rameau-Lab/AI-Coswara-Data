# created by DebarpanB
# date 26th August, 2022

import argparse, configparser
import numpy as np
import pandas as pd
import pickle
import os
import random

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from pdb import set_trace as bp

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from TransformerAgeModel import SoundDataset, TransformerAgeModel
from Wav2VecPretrained import FT_Wav2Vec

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def score(reference_labels,sys_scores,thresholds=np.arange(0,1,0.0001)):

	# Arrays to store true positives, false positives, true negatives, false negatives
	TP = np.zeros((len(reference_labels),len(thresholds)))
	TN = np.zeros((len(reference_labels),len(thresholds)))
	for keyCnt in range(len(sys_scores)): # Repeat for each recording
		sys_labels = (sys_scores[keyCnt]>=thresholds)*1	# System label for a range of thresholds as binary 0/1
		gt = reference_labels[keyCnt]

		ind = np.where(sys_labels == gt) # system label matches the ground truth
		if gt==1:	# ground-truth label=1: True positives 
			TP[keyCnt,ind]=1
		else:		# ground-truth label=0: True negatives
			TN[keyCnt,ind]=1
		
	total_positives = sum(reference_labels)	# Total number of positive samples
	total_negatives = len(reference_labels)-total_positives # Total number of negative samples   

	TP = np.sum(TP,axis=0)	# Sum across the recordings
	TN = np.sum(TN,axis=0)

	TPR = TP/total_positives	# True positive rate: #true_positives/#total_positives
	TNR = TN/total_negatives	# True negative rate: #true_negatives/#total_negatives

	AUC = auc( 1-TNR, TPR )    	# AUC 

	return AUC, TPR, TNR

def get_data(file_list,feats_file,labels_file, label, shuffle=False,):
	#bp()
	#%% read the list of files
	file_list = open(file_list).readlines()
	file_list = [line.strip().split() for line in file_list]

	#%% read labels
	temp = open(labels_file).readlines()
	temp = [line.strip().split() for line in temp]
	labels={}
	# categories = ['female', 'male']
	# for fil,label in temp:
	# 	labels[fil]=categories.index(label)

	for fil, label in temp:
		labels[fil] = int(label)
	del temp

	#%% read feats.scp
	temp = open(feats_file).readlines()
	temp = [line.strip().split() for line in temp]
	feats={}
	for fil,filpath in temp:
		feats[fil]=filpath
	del temp


	#%% make examples
	egs = []
	for fil,_ in file_list:
		if feats.get(fil,None):
			F = pickle.load(open(feats[fil],'rb'))
			label = labels.get(fil,None)
			if label is not None:
				egs.append( F)
	# trucnate or pad
	min_length = min(len(x) for x in egs)
	egs = [x[:min_length] for x in egs]
	egs = np.vstack(egs)
	if shuffle:
		np.random.shuffle(egs)	
	return egs[:,:-1], egs[:,-1]

def expand(x, y, gap=1e-4):
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    return x1, y1

def main(config, datadir_name, audiocategory,  label, output_dir):
	# bp()
	print(audiocategory, label)
	datadir = os.path.join(datadir_name, audiocategory)
	# Training dataset
	# @TODO CHANGE LABELS TO all_age_group
	train_feats, train_labels = get_data(datadir+"/all.scp",datadir+"/feats.scp",datadir+"/all", label, shuffle=True)
	# breakpoint()
	# train_feats = train_feats[:, :int(train_feats.shape[1]/3)]
	
	X_train, X_test, y_train, y_test = train_test_split(train_feats, train_labels, test_size=0.3, random_state=SEED)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size =0.21, random_state=SEED)
	# breakpoint()
	if config['default']['classifier'] == 'Wav2Vec':
		print("w2v")
		train_dataset = SoundDataset(X_train, y_train)
		val_dataset = SoundDataset(X_val, y_val)
		test_dataset = SoundDataset(X_test, y_test)
		train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

		w2v = FT_Wav2Vec()
		w2v.train(train_loader, val_dataset, test_loader)
	elif config['default']['classifier'] == 'Transformer':
		print("Transformer")
		# scaler = StandardScaler(with_mean=True, with_std=False)
		# X_train = scaler.fit_transform(X_train)
		# X_val = scaler.transform(X_val)
		# X_test = scaler.transform(X_test)
		train_dataset = SoundDataset(X_train, y_train)
		val_dataset = SoundDataset(X_val, y_val)
		test_dataset = SoundDataset(X_test, y_test)
		train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
		test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
		model = TransformerAgeModel(input_dim=64,
									model_dim=256,
									num_heads=8,
									num_layers=1,
									output_dim=1,
									dropout_rate=0.04)
		optimizer = optim.Adam(model.parameters(), lr=0.0001)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

		model.learn(train_loader, val_loader, test_loader, optimizer, scheduler)
		# 9.89
	elif config['default']['classifier']=='LassoRegression':
		print("Lasso Regression")

		# Scaling the data (assuming that scaling is necessary, uncomment if needed)
		scaler = StandardScaler(with_mean=True, with_std=False)
		X_train = scaler.fit_transform(X_train)
		X_val = scaler.transform(X_val)
		X_test = scaler.transform(X_test)

		# Initialize the Lasso Regression model
		# Alpha is a hyperparameter that controls regularization; adjust as necessary
		alpha_value = 0.01  # Example alpha value, adjust based on your needs
		max_iterations = 10000  # Increased from the default
		clf = Lasso(alpha=alpha_value, max_iter=max_iterations)

		# Train the model
		clf.fit(X_train, y_train)

		# Make predictions on validation and test sets
		y_val_pred = clf.predict(X_val)
		y_test_pred = clf.predict(X_test)

		# Calculate MAE for the validation and test sets
		val_mae = mean_absolute_error(y_val, y_val_pred)
		print(f'Validation MAE: {val_mae}')

		test_mae = mean_absolute_error(y_test, y_test_pred)
		print(f'Test MAE: {test_mae}')

		# Calculate RMSE for the validation and test sets
		val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
		print(f'Validation RMSE: {val_rmse}')

		test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
		print(f'Test RMSE: {test_rmse}')

	elif config['default']['classifier']=='LinearRegression':
		print("Linear Regression")

		# Scaling the data (assuming that scaling is necessary, uncomment if needed)
		scaler = StandardScaler(with_mean=True, with_std=False)
		X_train = scaler.fit_transform(X_train)
		X_val = scaler.transform(X_val)
		X_test = scaler.transform(X_test)
		print(X_train.shape, X_val.shape, X_test.shape)

		# Initialize the Linear Regression model
		clf = LinearRegression()
		print(X_train.shape)
		print(y_train.shape)

		# Train the model
		clf.fit(X_train, y_train)

		# Make predictions on validation and test sets
		y_val_pred = clf.predict(X_val)
		y_test_pred = clf.predict(X_test)

		val_mae = mean_absolute_error(y_val, y_val_pred)
		print(f'Validation MAE: {val_mae}')

		test_mae = mean_absolute_error(y_test, y_test_pred)
		print(f'Test MAE: {test_mae}')

		# Calculate RMSE for the validation and test sets
		val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
		print(f'Validation RMSE: {val_rmse}')

		test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
		print(f'Test RMSE: {test_rmse}')

	elif config['default']['classifier']=='RandomForest':
		print(config['default']['classifier'])
		scaler = StandardScaler(with_mean=True, with_std=False)
		# X_train = scaler.fit_transform(X_train)
		# X_val = scaler.transform(X_val)
		# X_test = scaler.transform(X_test)

		clf = RandomForestClassifier(n_estimators=int(config[config['default']['classifier']]['n_estimators']),
									criterion=config[config['default']['classifier']]['criterion'],
									random_state=SEED)
		clf.fit(X_train, y_train)
		val_auc, _, _ = score(y_val.tolist(), clf.predict_proba(X_val)[:,1].tolist())
		print(f'Val auc: {val_auc}')
		# exit()
		test_auc, _, _ = score(y_test.tolist(), clf.predict_proba(X_test)[:,1].tolist())
		print(f'Test auc: {test_auc}')
		exit()
		test_accuracy = (np.count_nonzero(clf.predict(X_test)==y_test)*100)/y_test.shape[0]
		print(f'Test accuracy: {test_accuracy}')
		op_save_path = os.path.join(output_dir, config['default']['classifier'])
		if not os.path.exists(op_save_path): os.mkdir(op_save_path)
		np.save(os.path.join(op_save_path, 'y_true.npy'), y_test)
		np.save(os.path.join(op_save_path, 'y_pred.npy'), clf.predict(X_test))
		np.save(os.path.join(op_save_path, 'y_pred_proba.npy'), clf.predict_proba(X_test))
	elif config['default']['classifier']=='LogisticRegression':
		print(config['default']['classifier'])
		scaler = StandardScaler(with_mean=True, with_std=False)
		# X_train = scaler.fit_transform(X_train)
		# X_val = scaler.transform(X_val)
		# X_test = scaler.transform(X_test)
		clf = LogisticRegression(C=float(config[config['default']['classifier']]['C']),
									class_weight=config[config['default']['classifier']]['class_weight'],
									max_iter=int(config[config['default']['classifier']]['max_iter']),
									random_state=SEED)
		clf.fit(X_train, y_train)
		val_auc, _, _ = score(y_val.tolist(), clf.predict_proba(X_val)[:,1].tolist())
		print(f'Val auc: {val_auc}')
		# exit()
		test_auc, _, _ = score(y_test.tolist(), clf.predict_proba(X_test)[:,1].tolist())
		print(f'Test auc: {test_auc}')
		exit()
		# val_accuracy = (np.count_nonzero(clf.predict(X_val)==y_val)*100)/y_val.shape[0]
		# print(f'Val accuracy: {val_accuracy}')
		test_accuracy = (np.count_nonzero(clf.predict(X_test)==y_test)*100)/y_test.shape[0]
		print(f'Test accuracy: {test_accuracy}')
		op_save_path = os.path.join(output_dir, config['default']['classifier'])
		if not os.path.exists(op_save_path): os.mkdir(op_save_path)
		np.save(os.path.join(op_save_path, 'y_true.npy'), y_test)
		np.save(os.path.join(op_save_path, 'y_pred.npy'), clf.predict(X_test))
		np.save(os.path.join(op_save_path, 'y_pred_proba.npy'), clf.predict_proba(X_test))
	else:
		print('unknown classifier. exiting...')
		exit()

	#plot
	# cdict = {0: 'red', 1: 'blue', 2: 'green'}
	# fig, ax = plt.subplots()
	# for g in np.unique(train_labels):
	# 	ix = np.where(train_labels == g)
	# 	ax.scatter(X_proj[ix][:,0], X_proj[ix][:,1], c = cdict[g], label = g, s = 100, alpha=0.4*(2.1-g))
	# 	#ax.scatter(*expand(X_proj[ix][:,0], X_proj[ix][:,1]), lw=0, c = cdict[g], label = g, s = 100, alpha=0.5)
	# ax.legend()
	# save_plot_dir = outdir+'/'+config['default']['type']
	# if not os.path.exists(save_plot_dir): os.mkdir(save_plot_dir)
	# plt.savefig(save_plot_dir+'/plot.png')


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--classification_config', '-c', required=True)
	parser.add_argument('--datadir', '-d', required=True)
	parser.add_argument('--audiocategory', '-a', required=True)
	# parser.add_argument('--metadata_file', '-m', required=True)
	parser.add_argument('--output_dir', '-o', required=True)
	parser.add_argument('--label', '-l', required=True)
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.classification_config)

	main(config, args.datadir, args.audiocategory, args.label, args.output_dir)