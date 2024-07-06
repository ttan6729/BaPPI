import torch
import torch.nn as nn
import math
import random
import torch_geometric
import hyena
import torch.nn.functional as F
import torch_geometric.nn.conv as Conv
from torch_geometric.typing import OptTensor
import numpy as np
import sklearn
from torch.nn import Parameter

class BaPPI(torch.nn.Module): #hyena
	def __init__(self, in_len=512, d1=13, d2=0,layer_num=3,use_=False,pool_size=3,cnn_hidden=1,train_eps=True,feature_fusion='mul',class_num=7):
		super(BaPPI,self).__init__()
		self.models = torch.nn.ModuleList()#seven independent GNN models
		self.layer_num = layer_num
		self.class_num = class_num
		self.feature_fusion = feature_fusion
		self.f1_transform = 64
		self.layer_num = layer_num
		self.in_len = in_len
		self.d1 = d1
		
		self.long_conv = hyena.HyenaOperator(d_model=d1,l_max=in_len)

		self.conv1d = nn.Conv1d(in_channels=d1, out_channels=cnn_hidden, kernel_size=3, padding=1)
		self.res = torch.nn.Sequential(nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=1, stride=1),
			nn.ReLU(),nn.BatchNorm1d(cnn_hidden),
			nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
			nn.ReLU(),nn.BatchNorm1d(cnn_hidden),
			nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
			nn.ReLU(),nn.BatchNorm1d(cnn_hidden))

		self.act = nn.ReLU()
		self.bn1 = nn.BatchNorm1d(cnn_hidden)
		self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
		self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
		self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
		self.fc1 = nn.Linear(math.floor( in_len / pool_size),self.f1_transform )

		hidden = self.f1_transform 

		for i in range(class_num):
			newLayers = torch.nn.ModuleList()
			for j in range(self.layer_num-1):
				newLayers.append(torch_geometric.nn.models.GraphSAGE(hidden,hidden,1,act=nn.ReLU(),act_first=True,norm=nn.BatchNorm1d(hidden)))
			newLayers.append(Conv.AntiSymmetricConv(in_channels = hidden, act='tanh'))
			newLayers.append(nn.BatchNorm1d(hidden))
			self.models.append(newLayers)

		hidden3 = (1+class_num*layer_num)*hidden
		self.fc2 = get_classifier(hidden3,class_num,feature_fusion)

	def forward(self,f1,f2=None,edge_index=None,train_edge_id=None,edge_attr = None,p=0.5):  
		f1 = self.long_conv(f1)  #original shape, e.g. torch.Size([1690, 512, 33]) 
		f1 = f1.transpose(1, 2)
		f1 = self.conv1d(f1)
		f1 = f1 + self.res(f1)
		#f1 = self.act(f1)

		#f1 = self.bn1(f1)
		f1 = self.maxpool1d(f1)
		f1 = f1.transpose(1, 2)
		f1, _ = self.biGRU(f1)
		f1 = self.global_avgpool1d(f1)
		f1 = f1.squeeze()

		f1 = self.fc1(f1)
		x = f1
		output = [x]
		for i,m in enumerate(self.models):
			tmp = x
			for j in range(self.layer_num-1):
				tmp = self.models[i][j](tmp,edge_attr[i])
				output.append(tmp)
			tmp = self.models[i][-2](tmp,edge_attr[i])
			tmp = self.models[i][-1](tmp)#batch norm
			output.append(tmp)

		x = torch.cat(output,dim=1)

		node_id = edge_index[:, train_edge_id]
		x1 = x[node_id[0]]
		x2 = x[node_id[1]]

		if self.feature_fusion == 'CnM':
			x = torch.cat([torch.mul(x1, x2),x1, x2], dim=1)
		elif self.feature_fusion == 'concat':
			x = torch.cat([x1,x2],dim=1)
		elif self.feature_fusion == 'mul':
			x = torch.mul(x1, x2)
		x = self.fc2(x)
		return x




class DAE(nn.Module): #deep autoencoder
	def __init__(self,input_shape=20,hidden=-1,hidden2=-1):
		super(DAE, self).__init__()
		if hidden == -1:
			hidden = input_shape
		self.hidden2 = hidden2
	
		self.encoder = nn.Sequential(
			nn.Linear(input_shape, hidden),nn.ReLU(True),
			nn.Linear(hidden, hidden),nn.ReLU(True),
			nn.Linear(hidden, hidden),nn.ReLU(True),
			nn.Linear(hidden, 2*hidden),nn.ReLU(True),
			nn.Linear(2*hidden, hidden2))
		self.decoder = nn.Sequential(
			nn.Linear(hidden2, 2*hidden),nn.ReLU(True),
			nn.Linear(2*hidden, hidden),nn.ReLU(True),
			nn.Linear(hidden, hidden),nn.ReLU(True),
			nn.Linear(hidden, hidden),nn.ReLU(True),
			nn.Linear(hidden, input_shape))
		self.model = nn.Sequential(self.encoder, self.decoder)
	
	def encode(self, x):
		return self.encoder(x)

	def forward(self, x):
		x = self.model(x)
		return x

class ClusteringLayer(nn.Module):
	def __init__(self, n_clusters=10,hidden=-1, cluster_centers=None, alpha=1.0):
		super(ClusteringLayer, self).__init__()
		self.n_clusters = n_clusters
		self.alpha = alpha
		self.hidden = hidden
		if cluster_centers is None:
			initial_cluster_centers = torch.zeros(
			self.n_clusters,
			self.hidden,
			dtype=torch.float
			).cuda()
			nn.init.xavier_uniform_(initial_cluster_centers)
		else:
			initial_cluster_centers = cluster_centers
		self.cluster_centers = Parameter(initial_cluster_centers)

	def forward(self, x):
		norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
		numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
		power = float(self.alpha + 1) / 2
		numerator = numerator**power
		t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
		return t_dist

class DEC(nn.Module):
	def __init__(self, n_clusters=10,hidden=10,cluster_centers=None, alpha=1.0,autoencoder=None):
		super(DEC, self).__init__()
		self.n_clusters = n_clusters
		self.alpha = alpha
		self.hidden = hidden
		self.cluster_centers = cluster_centers
		self.autoencoder = autoencoder
		self.clusteringlayer = ClusteringLayer(n_clusters, hidden, cluster_centers, alpha)

	def target_distribution(self, q_):
		weight = (q_ ** 2) / torch.sum(q_, 0)
		return (weight.t() / torch.sum(weight, 1)).t()

	def forward(self, x):
		x = self.autoencoder.encode(x)
		return self.clusteringlayer(x)

	# def visualize(self, epoch,x): #for 2d graph
	# 	fig = plt.figure()
	# 	ax = plt.subplot(111)
	# 	x = self.autoencoder.encode(x).detach() 
	# 	x = x.cpu().numpy()[:2000]
	# 	x_embedded = TSNE(n_components=2).fit_transform(x)
	# 	plt.scatter(x_embedded[:,0], x_embedded[:,1])
	# 	fig.savefig('plots/mnist_{}.png'.format(epoch))
	# 	plt.close(fig)

def add_noise(data):
	#noise = torch.rand(data.size())*torch.randint(0,1,data.size())*0.2

	noise = torch.tensor(2*np.random.randint(2,size=data.size())-1,dtype=torch.float)
	noise = torch.rand(data.size())*noise*0.2
	noisy_data = data + data*noise
	return noisy_data


def get_classifier(hidden_layer,class_num,feature_fusion):
	fc = None
	if feature_fusion == 'CnM':
		fc = nn.Linear(3*hidden_layer,class_num)
	elif feature_fusion == 'concat':
		fc = nn.Linear(2*hidden_layer,class_num)
	elif feature_fusion == 'mul':
		fc = nn.Linear(1*hidden_layer,class_num)
	return fc

class AsymmetricLossOptimized(nn.Module):
	''' Notice - optimized version, minimizes memory allocation and gpu uploading,
	favors inplace operations'''

	def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
		super(AsymmetricLossOptimized, self).__init__()

		self.gamma_neg = gamma_neg
		self.gamma_pos = gamma_pos
		self.clip = clip
		self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
		self.eps = eps

		# prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
		self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

	def forward(self, x, y):
		""""
		Parameters
		----------
		x: input logits
		y: targets (multi-label binarized vector)
		"""
		self.targets = y
		self.anti_targets = 1 - y

		# Calculating Probabilities
		self.xs_pos = torch.sigmoid(x)
		self.xs_neg = 1.0 - self.xs_pos

		# Asymmetric Clipping
		if self.clip is not None and self.clip > 0:
			self.xs_neg.add_(self.clip).clamp_(max=1)

		# Basic CE calculation
		self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
		self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

		# Asymmetric Focusing
		if self.gamma_neg > 0 or self.gamma_pos > 0:
			if self.disable_torch_grad_focal_loss:
				torch.set_grad_enabled(False)
			self.xs_pos = self.xs_pos * self.targets
			self.xs_neg = self.xs_neg * self.anti_targets
			self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
										  self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
			if self.disable_torch_grad_focal_loss:
				torch.set_grad_enabled(True)
			self.loss *= self.asymmetric_w

		return -self.loss.sum()