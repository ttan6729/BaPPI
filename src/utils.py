import numpy as np
import random
import torch
import os
import math
import time
import Models
from collections import defaultdict 
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
def sorted_pair(id1,id2):
	if id1<id2:
		return [id1,id2]
	return [id2,id1]

def check_files_exist(fList:list):
	for fName in fList:
		if not os.path.isfile(fName):
			print(f'error,input file {fName} does not exist')
			exit()
	return
#encode binary vecotr into single num
def encode_inter(vec,num=7):
	result = 0
	for i,v in enumerate(vec):
		result += v*pow(2,num-1-i)
	return result
#		print([1,1,0,0,0,1,0])
#		a = utils.encode_inter([1,1,0,0,0,1,0])
def decode_inter(value,num=7):
	result = []
	for i in range(num):
		tmp = pow(2,num-1-i)
		v = math.floor(value/tmp)
		#print(f'{value} {tmp} {v}')
		value -= v*tmp
		result.append(v)
	return np.array(result,dtype=float)
def sort_dir_by_value(dict): #note: decesending order
	keys = list(dict.keys())
	values = list(dict.values())
	sorted_value_index = np.argsort(values)[::-1]
	sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
	return sorted_dict


class Metrictor_PPI:
	def __init__(self, pre_y, truth_y, is_binary=False):
		self.TP = 0
		self.FP = 0
		self.TN = 0
		self.FN = 0

		if is_binary:
			length = pre_y.shape[0]
			for i in range(length):
				if pre_y[i] == truth_y[i]:
					if truth_y[i] == 1:
						self.TP += 1
					else:
						self.TN += 1
				elif truth_y[i] == 1:
					self.FN += 1
				elif pre_y[i] == 1:
					self.FP += 1
			self.num = length

		else:
			N, C = pre_y.shape
			for i in range(N):
				for j in range(C):
					if pre_y[i][j] == truth_y[i][j]:
						if truth_y[i][j] == 1:
							self.TP += 1
						else:
							self.TN += 1
					elif truth_y[i][j] == 1:
						self.FN += 1
					elif truth_y[i][j] == 0:
						self.FP += 1
			self.num = N * C
	
	def append_result(self,path='test.txt',e=None):
		self.acc = (self.TP + self.TN) / (self.num + 1e-10)
		self.pre = self.TP / (self.TP + self.FP + 1e-10)
		self.recall = self.TP / (self.TP + self.FN + 1e-10)
		self.microF1 = 2 * self.pre * self.recall / (self.pre + self.recall + 1e-10)
		record = f'epoch {e},acc {self.acc:.4f}, microF1 {self.microF1:.4f}, precision {self.pre:.4f},recall {self.recall:.4f}'#,loss {loss}'
		with open(path,'a') as f:
			f.write(record+'\n')
		return record

	def show_result(self, is_print=False, file=None):
		self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
		self.Precision = self.TP / (self.TP + self.FP + 1e-10)
		self.Recall = self.TP / (self.TP + self.FN + 1e-10)
		self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)
		if is_print:
			print_file("Accuracy: {}".format(self.Accuracy), file)
			print_file("Precision: {}".format(self.Precision), file)
			print_file("Recall: {}".format(self.Recall), file)
			print_file("F1-Score: {}".format(self.F1), file)


def optimizedSmote(edgeList,interList,original_feature=None,seqsNum=0,portion=0.2,class_num=7,epoch=100): #edgeList: [0,1] interList [1 1 0 0 0 1 0] seqNum
	protein_type_num = 7 #assume seven protein types, antibodies, contractile proteins, enzymes, hormonal proteins, structural proteins, storage proteins, and transport proteins
	epcoh1,epoch2 = 50,20
	batch_size = 256 
	print('begin optimzied smote')
	encInter = [encode_inter(inter,class_num) for inter in interList] #encode length 7 binary vector into float, vice versa
	# neighs = defaultdict(list)
	# for edge in edgeList:
	#     a,b = edge[0],edge[1]
	#     neighs[a].append(b)
	#     neighs[b].append(a)

	# nDegree = np.zeros(seqsNum,dtype=float)
	# for k in neighs.keys():
	#     nDegree[k] = len(neighs[k])
	labelNum = pow(2,class_num)	
	labelFreq = np.zeros(pow(2,class_num),dtype=float)
	for i in encInter:
		labelFreq[i] += 1

		#class_freq = np.zeros(self.class_num,dtype=float)
	classFreq = np.sum(interList,axis=0,dtype=float)   
	classScore = np.array(classFreq)
		
	classScore /= (len(interList))
	alpha,beta = 5,0.3
	classScore = ([ (math.exp(-alpha*min(0,c-0.3))-1) for c in classScore])
	labelScore = []
	for i in range(len(labelFreq)):
		inter = decode_inter(i,class_num)
		tmp = np.sum( [ inter[j] * classScore[j] for j in range(len(inter))] ) #increase score for minor label
		tmp *= pow(labelFreq[i],beta) #decrease score for minor label combination
		labelScore.append(tmp)

	count=0 #in SHS27K, around 50% labels has more than 1 samples
	#decide added number of each label -- model for x1*x2 embedding -- find similar pair inside each label -- smote 
	totalLabelScore = 0.0
	threshold = 3
	n_cluster = 0
	for i,score in enumerate(labelScore):
		if labelFreq[i] > threshold:
			totalLabelScore += score
		if labelFreq[i] >= 1:
			n_cluster += 1        

	addedLabels = []
	totalAdded = int(portion * len(edgeList))  
	addedNum = []

	for i,score in enumerate(labelScore):
		if labelFreq[i] <= threshold: #skip label combination with too few sample
			addedNum.append(0)
		else:
			addedNum.append(int(totalAdded*score/totalLabelScore))

	totalScore = 0.0
	for inter in interList:
		totalScore += labelScore[encode_inter(inter)]


	node_info = defaultdict(lambda:-1) # key: node id, value [total score, node degree,[original edge index]]
	for i,edge in enumerate(edgeList):
		e1, e2 = edge[0], edge[1]
		score = labelScore[encode_inter(interList[i])]
		if node_info[e1] == -1:
			node_info[e1] = [0,0,[]]
		if node_info[e2] == -1:
			node_info[e2] = [0,0,[]]
		node_info[e1][0] += score
		node_info[e1][1] += 1
		node_info[e1][2].append(i)
		node_info[e2][0] += score
		node_info[e2][1] += 1
		node_info[e2][2].append(i)

	node_to_avg_score = {} 
	for key in node_info.keys():
		node_to_avg_score[key] = node_info[key][0]/node_info[key][1]
	node_to_avg_score = sort_dir_by_value(node_to_avg_score)

	node_num = len(node_to_avg_score)
	tmp = []

	current_freq = np.sum(np.array(interList),axis=0)
	#print(f'previous freq:{current_freq}')
	added = np.array([0,0,0,0,0,0,0],dtype=float)
	for i,num in enumerate(addedNum):
		curAdded = num * decode_inter(i)
		added += curAdded

	currentAddedScore = 0
	totalAddedScore = totalScore * portion


	tmp_count = 0
	added = np.array([0,0,0,0,0,0,0],dtype=float)
	factor = 4
	keys = list(node_to_avg_score.keys())
	addNodeId = []
	while currentAddedScore < totalAddedScore:
		random_value = math.exp(-factor*random.random()) - math.exp(-factor)
		nodeId = keys[int(random_value*node_num)]
		avg_score = node_to_avg_score[nodeId]
		degree = node_info[nodeId][1]
		if avg_score <= 2.0 or degree < 3:
			continue
		elif degree > 10 and random.random() > 0.7:
			continue
		else:
			addNodeId.append(nodeId)
			currentAddedScore += node_info[nodeId][0]
			for edgeId in node_info[nodeId][2]:
				added += np.array(interList[edgeId])

	inter_to_edge_id = defaultdict(list)
	for i,inter in enumerate(interList):
		inter_to_edge_id[encode_inter(inter)].append(i)

	node_to_seq = defaultdict(lambda:-1) # the id in edge_list to id in feature matrix
	feature = []
	for i,edge in enumerate(edgeList):
		e1,e2 = edge[0], edge[1]
		if node_to_seq[e1] == -1:
			node_to_seq[e1] = len(feature)
			feature.append(original_feature[e1])
		if node_to_seq[e2] == -1:
			node_to_seq[e2] = len(feature)
			feature.append(original_feature[e2])

	feature = np.array(feature)
	feature = torch.tensor(feature,dtype=torch.float)
	#feature = torch.tensor(original_feature,dtype=torch.float)
	device = torch.device('cpu')
	feature.to(device)

	start = time.time()
	train_loader = DataLoader(dataset=feature,batch_size=batch_size, shuffle=False)
	
	loss_fn = torch.nn.L1Loss().to(device)
	hidden = int(0.2*feature.size()[1])
	autoencoder = Models.DAE(input_shape=feature.size()[1],hidden2=hidden)
	optimizer = torch.optim.Adam(autoencoder.parameters(),lr=0.001, weight_decay=1e-5)
	for e in range(epcoh1):
		for i,data in enumerate(train_loader):
			nosiy_data = Models.add_noise(data).to(device)
			output = autoencoder(nosiy_data)		
			loss = loss_fn(output,data)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		#print(f'epoch {e}, loss:{loss.item():.4f}')	
	print(f'autoencoder time {(time.time()-start):.4f}')

	start = time.time()
	eFatures = []

	model = Models.DEC(n_clusters = protein_type_num,hidden = hidden,autoencoder=autoencoder)
	for i,batch in enumerate(train_loader): 
		batch = batch.to(device)
		eFatures.append(autoencoder.encode(batch).detach().cpu())
	eFatures = torch.cat(eFatures).to(device)

	kmeans = KMeans(n_clusters=protein_type_num, random_state=0).fit(eFatures) #sklearn kmeans
	cluster_centers = kmeans.cluster_centers_
	cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cpu()
	print(cluster_centers.size())
	model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
	loss_function = torch.nn.KLDivLoss(size_average=False)
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)
	for e in range(epoch2):
		#for i,data in enumerate(train_loader):
		output = model(feature)
		target = model.target_distribution(output).detach()
		out = output.argmax(1)
		loss = loss_fn(output.log(),target)/output.shape[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	output = model(feature)
	cluster_result = output.argmax(1).numpy()
	print(f'Deep embedding Clustering time {(time.time()-start):.4f}')

	seq_to_node  = {v: k for k, v in node_to_seq.items()}
	#addNodeId node_to_seq cluster_result feature
	start = time.time()
	addedNodeInfo = []

	seqCount = original_feature.shape[0]
	newEdges, newFeatures, newInter = [], [], []
	for i,nid in enumerate(addNodeId):
		seqId = find_nearest_neigh(node_to_seq[nid],cluster_result,np.array(feature))
		newNodeId = seq_to_node[seqId]
		rValue = 0.5*random.random()+0.5
		addedNodeInfo.append([seqCount,nid,newNodeId,rValue]) 
		for edgeId in node_info[nid][2]:
			newInter.append(interList[edgeId])
			newEdges.append([seqCount,edgeList[edgeId][1]])
		seqCount += 1

	return addedNodeInfo,newInter, newEdges #addedNodeInfo [newId, original id 1, original id 2, random value]

def find_nearest_neigh(nodeId,cluster_result,features):  #int, np array(int), np array (float)
	nodeFeature = features[nodeId]
	nodeCluster = cluster_result[nodeId]
	clusterIds = np.where(cluster_result==nodeCluster)[0]
	tmp = []
	feaNum = features.shape[0]

	for i in clusterIds:
		if cluster_result[i] == nodeCluster:
			if i == nodeId:
				tmp.append(1000)
			else:
				#tmp.append( np.sum(np.abs(nodeFeature-features[i]) ) )
				tmp.append(np.linalg.norm(nodeFeature-features[i]))
	min_index = np.argmin(np.array(tmp))


	return clusterIds[min_index]

def add_feature(addedNodeInfo,features):
	newFeatures = []
	for info in addedNodeInfo:
		id1,id2,rValue = info[1],info[2],info[3]
		newFeatures.append( features[id1]*rValue+features[id2]*(1-rValue) )
	newFeatures = np.array(newFeatures)
	print(newFeatures.shape)
	print(features.shape)
	features = np.concatenate((features,newFeatures),axis=0)
	print(features.shape)

	return features

#return: dict, key :encoded interaction id, value [edge]	
#edgeList [[0,3]...] inter_to_edge_id: {0:[0,2,3..],1[1,7,...]}
def create_synthetic_edges(edgeList,addedNum,cluster_result,inter_to_edge_id):
	result = []# each element [pair1,pair2,factor]
	for inter_id,num in enumerate(addedNum):
		if num == 0:
			continue
		edge_id = inter_to_edge_id[inter_id]
		#print(f'inter {decode_inter(inter_id)}, num {num}')
		cluster_ids = select_cluster([cluster_result[e] for e in edge_id]) #selected cluster id list for inter_id 
		clusters = defaultdict(lambda: None) #extract ids for the current inter_id within each cluster
		for c in cluster_ids:
			clusters[c] = []
		for eid in edge_id:
			cid = cluster_result[eid] 
			if clusters[cid] is not None:
				clusters[cid].append(eid)

		cluster_list = [] #transofrm dict to list
		for k in clusters.keys():
			if clusters[k] is not None:
				cluster_list.append(clusters[k])
	
		if len(cluster_list) == 0:
			#print(f'added num {num}, no cluster, edge num {len(edge_id)} {[cluster_result[e] for e in edge_id]}')
			cluster_list = [[cluster_result[e] for e in edge_id]]
		tmp_cluster_num = len(cluster_list)
		
		for i in range(num):
			cluster = cluster_list[random.randrange(0,tmp_cluster_num)]
			selected = np.random.choice(cluster,2,replace=False)
			selected = [selected[0],selected[1],np.random.random()]
			result.append(selected)
			if tmp_cluster_num == 1 and i > len(cluster_list[0]):
				continue


	return result

def select_cluster(data):
	output = defaultdict(lambda:0)
	for item in data:
		output[item] += 1

	ordered_cluster  = [i[0] for i in sorted(output.items(), key=lambda x:x[1])]
	ordered_cluster = list(reversed(ordered_cluster))

	
	selected_cluster = []
	for i,c in enumerate(ordered_cluster):
		if len(selected_cluster) < 3 or output[c] > 5:
			if output[c] > 3:
				selected_cluster.append(c)
	return selected_cluster
#split orginal label set into labels of long and short tail classes
def split_label(edge_index,label,n_data,label_num,ratio,train_mask):
	n_data = torch.tensor(n_data)
	sorted_n_data, indices = torch.sort(n_data,descending=True) #number of sample in eacch label
	print(n_data)
	print(sorted_n_data)
	print(indices)
	inv_indices = np.zeros(label_num,dtype=np.int64) #indices in n_data -> order of number
	for i in range(label_num):
		inv_indices[indices[i].item()] = i #
	print(inv_indices)
	mu = np.power(1/ratio,1/(label_num-1))
	n_round = []
	label_num_list = []
	for i in range(label_num):
		label_num_list.append(int(min(sorted_n_data)))
		if i < 1:
			n_round.append(1)
		else:
			n_round.append(10)
	print(label_num_list)
	label_num_list = np.array(label_num_list)
	label_num_list = label_num_list[inv_indices]
	print(n_round)
	return



def create_sythetic_data():
	label_num = 7
	sample_num = 100
	np.random.seed(0)

	return np.random.choice(label_num,sample_num,p=[0.25,0.25,0.3,0.05,0.05,0.05,0.05])

def normalize(fMatrix):
	max_values = fMatrix.max(axis=0)
	min_values = fMatrix.min(axis=0)
	row,col = fMatrix.shape[0],fMatrix.shape[1]
	for c in range(col):
		if (max_values[c]-min_values[c])<0.001:
			continue
		deno = max_values[c]-min_values[c]
		for r in range(row):
			fMatrix[r][c] =  (fMatrix[r][c]-min_values[c])/deno
	return fMatrix
