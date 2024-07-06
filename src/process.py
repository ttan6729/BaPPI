from collections import defaultdict 
import numpy as np
import random
import utils
import embedding
import re 
import torch
import json
import math
import torch_geometric
from gensim.models import Word2Vec
labelDir = {'reaction':0, 'binding':1, 'ptmod':2, 'activation':3,'inhibition':4, 'catalysis':5, 'expression':6}
amino_list = ['A','G','V','I','L','F','P','Y','M','T','S','H','N','Q','W','R','K','D','E','C']


#mul, mm: for multiplication of 2d matrix, bmm: mm+batch, matmul: matrix multiplication, spmm: sparse type
class PPIData(object):
	def __init__(self,seqPath,relPath,args,use_f2=None,class_num = 7):

		self.seqPath, self.relPath,self.args = seqPath, relPath, args
		self.class_num = class_num
		self.seqs, self.name2index = readSeqs(seqPath)
		self.seqsNum = len(self.name2index)
		self.pairList,self.interList,self.pair2index,self.neighIndex,self.edgeList  = readInteraction(relPath,self.name2index)
		self.encFeature = seqEncoding(self.seqs,args.L,PSSM=args.PSSM)
		self.seqFeature = seqEmbedding(self.seqs)
		 #np array
		self.weight = torch.from_numpy(np.ones(len(self.seqs)))
		self.split_dict = {}
		if args.m == 'bfs':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.split_dataset_bfs(test_percentage=0.2,edgeList=self.edgeList)
		elif args.m == 'dfs':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.split_dataset_dfs(test_percentage=0.2,edgeList=self.edgeList)
		elif args.m == 'random':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.split_dataset_random(test_percentage=0.2)			
		elif args.m == 'read':
			self.split_dict['train_index'], self.split_dict['valid_index'] = self.read_test_set(args.i3)		
		if args.m != 'read':
			self.save_valid_set(args)
		# if args.bs:
		# 	train_index = self.split_dict['train_index']
		# 	#addedNodeInfo [newId, original id 1, original id 2, random value]
		# 	addedNodeInfo,newInter, newEdges = utils.optimizedSmote([self.edgeList[t] for t in train_index],[self.interList[t] for t in train_index],self.seqFeature,self.seqsNum,portion=args.bp)
		# 	self.encFeature = utils.add_feature(addedNodeInfo,self.encFeature)
		# 	self.seqFeature = utils.add_feature(addedNodeInfo,self.seqFeature)
		# 	self.split_dict['train_index'].extend([x for x in range(len(self.edgeList),len(self.edgeList)+len(newEdges)) ])
		# 	self.interList.extend(newInter)
		# 	self.edgeList.extend(newEdges)
		if args.sv:
			print('save dataset split and synthetic dataset')

		edge_index = torch.tensor(self.edgeList, dtype=torch.long).transpose(0,1)
		edge_indices = create_edge_indices(self.edgeList,self.interList) #the edge list for each individual graph
		edge_indices = [torch.tensor(e,dtype=torch.long).transpose(0,1) for e in edge_indices]

		self.data = torch_geometric.data.Data(edge_index=np.array(edge_index,dtype=int),edge_attr=edge_indices,edge_attr_1 = torch.tensor(self.interList, dtype=torch.long)) 
		self.data.f1 = torch.tensor(self.encFeature,dtype=torch.float)
		self.data.f2 = torch.tensor(self.seqFeature,dtype=torch.float)
		self.data.train_mask = self.split_dict['train_index']
		self.data.val_mask = self.split_dict['valid_index']

	def save_valid_set(self,args):
		if args.m != 'read' or args.i3[-4:] == 'data':
			valid_pair = []
			train_pair = []
			for index in self.split_dict['valid_index']:
				valid_pair.append(self.pairList[index])
			for index in self.split_dict['train_index']:
				train_pair.append(self.pairList[index])
			result = {'valid_index':valid_pair,'train_index':train_pair,'seqPath':self.seqPath,'interPath':self.relPath}
			jsobj = json.dumps(result)
			with open(args.o+'.json', 'w') as f:
				f.write(jsobj)

	def split_dataset_bfs(self,node_to_edge_index=None,test_percentage=0.2,edgeList=None,src_path=None): #list of interaction, percentage of
		if not node_to_edge_index:
			node_to_edge_index = defaultdict(list)
			for i in range(len(edgeList)):
				id1, id2 = edgeList[i][0], edgeList[i][1]
				node_to_edge_index[id1].append(i)
				node_to_edge_index[id2].append(i)

		node_num = len( node_to_edge_index.keys() )
		test_size = int(len(edgeList) * test_percentage)
		test_set = []
		queue = []
		visited = []

		random_index = random.randint(0, node_num-1)
		while len( node_to_edge_index[random_index] ) > 5:
			random_index = random.randint(0, node_num-1)
		queue.append(random_index)
		print(f'root level {len( node_to_edge_index[random_index])}')
		count = 0
		#print(node_to_edge_index[random_index])

		while len(test_set) < test_size:
			if len(queue) == 0:
				print('bfs split meet root level 0, terminate process')
				exit()
				# while(random_index in visited):
				# 	random_index = random.randint(0, node_num-1)
				# queue.append(random_index)
			cur_node = queue.pop(0) 
			visited.append(cur_node)
			for edge_index in node_to_edge_index[cur_node]:
				if edge_index not in test_set:
					test_set.append(edge_index)
					id1,id2 = edgeList[edge_index][0],edgeList[edge_index][1]
					next_node = id1
					if id1 == cur_node:
						next_node = id2
					if next_node not in visited and next_node not in queue:
						queue.append(next_node)
				else:
					continue
		
		#test_set = np.array(test_set,dtype=int)
		training_set = self.construct_training_set(test_set)
		return training_set,test_set

	def split_dataset_dfs(self,node_to_edge_index=None,test_percentage=0.2,edgeList=None,src_path=None):
		if not node_to_edge_index:
			node_to_edge_index = defaultdict(list)
			for i in range(len(edgeList)):
				id1, id2 = edgeList[i][0], edgeList[i][1]
				node_to_edge_index[id1].append(i)
				node_to_edge_index[id2].append(i)

		node_num = len( node_to_edge_index.keys() )
		test_size = int(len(edgeList) * test_percentage)
		test_set = []
		stack = []
		visited = []

		random_index = random.randint(0, node_num-1)
		while len( node_to_edge_index[random_index] ) > 5:
			random_index = np.random.randint(0, node_num-1)
		print(f'random index {random_index},root level {len( node_to_edge_index[random_index])}')

		stack.append(random_index)

		while(len(test_set) < test_size):
			if len(stack) == 0:
				print('dfs split meet root level 0')
				exit()

			cur_node = stack[-1]
			if cur_node in visited:
				flag = True
				for edge_index in node_to_edge_index[cur_node]:
					if flag:
						id1,id2 = edgeList[edge_index][0],edgeList[edge_index][1]
						next_node = id1 if id2 == cur_node else id2
						if next_node in visited:
							continue
						else:
							stack.append(next_node)
							flag = False
					else:
						break
				if flag:
					stack.pop()
				continue
			else:
				visited.append(cur_node)
				for edge_index in node_to_edge_index[cur_node]:
					if edge_index not in test_set:
						test_set.append(edge_index)
		#test_set = np.array(test_set,dtype=int)
		training_set = self.construct_training_set(test_set)
		return training_set,test_set

	def split_dataset_random(self,test_percentage=0.2):
		all_indices = [i for i in range(len(self.edgeList))]
		test_size = int(test_percentage*len(all_indices))
		test_set = random.sample(all_indices,test_size)
		training_set = self.construct_training_set(test_set)
		return training_set,test_set


	def construct_training_set(self,test_indices):
		all_indices = [i for i in range(len(self.edgeList))]
		training_indices = list( set(all_indices).difference( set(test_indices) ) )
		assert len(self.edgeList) == (len(training_indices)+len(test_indices)), "error, the size of training and test set doesn't match"		
		return training_indices

	# 	return training_set,test_set
	def read_test_set(self,save_path):
		valid_index = []
		if save_path[-4:] == 'json':
			with open(save_path, 'r') as f:
				self.ppi_split_dict = json.load(f)
			print(len(self.ppi_split_dict['valid_index']))
			print(len(self.edgeList))

			for pair in self.ppi_split_dict['valid_index']:
				pair = pair.split('__')

				newPair = utils.sorted_pair(pair[0],pair[1])
				newPair = newPair[0] +'__'+ newPair[1]
				if self.pair2index[newPair] == -1:
					print(f'error, unfound pair {newPair}')
					exit()
				valid_index.append(self.pair2index[newPair])

		elif save_path[-4:] == 'data':
			with open(save_path, 'r') as f:
				fName1 = f.readline()
				fName2 = f.readline()
				lines = f.readlines()
				for line in lines:
					pair = line.strip().split('\t')
					newPair = utils.sorted_pair(pair[0],pair[1])
					newPair = newPair[0] +'__'+ newPair[1]
					if self.pair2index[newPair] == -1:
						print(f'error, unfound pair {newPair}')
						exit()
					valid_index.append(self.pair2index[newPair])
		print(len(valid_index))
		# if len(valid_index) < 4000:
		# 	exit()
		return self.construct_training_set(valid_index),valid_index


def readSeqs(seqPath):
	name2index = defaultdict(lambda:-1)
	seqs = []
	utils.check_files_exist([seqPath])
	count = 0
	with open(seqPath,'r') as file:
		lines = file.readlines()
		for line in lines:
			tmp=re.split(',|\t',line.strip('\n'))
			if name2index[tmp[0]] == -1:
				seqs.append(tmp[1])
				name2index[tmp[0]] = count
				count += 1
	#replace unrecognzied aminoa acid
	for i,seq in enumerate(seqs):
		tmp = []
		for s in seq:
			if s not in amino_list:
				tmp.append(s)
		if len(tmp) > 0:
			for t in tmp:
				seqs[i] = seqs[i].replace(t,'')	
	return seqs,name2index

def create_adj_tensors(neighIndex):
	seqNum = len(neighIndex)
	labelNum = len(neighIndex[0])
	result = np.zeros((labelNum,seqNum,seqNum),dtype=int)
	print(f'{len(neighIndex)} {len(neighIndex[0])} {len(neighIndex[0][0])}')

	for i in range(seqNum):
		for j in range(labelNum):
			for index in neighIndex[i][j]:
				result[j][i][index] = 1

	tmp = []
	for i in(range(labelNum)):
		tmp.append(torch.from_numpy(result[i]).to_sparse())
	# print(tmp[0].size())
	# print(tmp[0].indices()[0:10])
	return tmp

def readInteraction(relPath,name2index,level=3):
	labelNum = len(labelDir)
	#pairList and pair2index can be considered as inverse index
	pairList,interList = [],[]
	pair2index = defaultdict(lambda:-1) 
	utils.check_files_exist([relPath])
	edgeList = []
	with open(relPath,'r') as file:
		header  = file.readline() #assume the first line is header
		lines = file.readlines() 
		for line in lines:
			tmp=re.split(',|\t',line.strip('\n'))
			protein1,protein2,mode,is_dir = tmp[0], tmp[1], labelDir[tmp[2]], tmp[4]
			if name2index[protein1] == -1 or name2index[protein2] == -1:
				print(f'error, unrecognzied sequence {tmp}')
				exit()
			newPair = utils.sorted_pair(protein1,protein2)
			newIndex = [name2index[newPair[0]],name2index[newPair[1]]]
			newPair = newPair[0] +'__'+ newPair[1]
			if pair2index[newPair] == -1:
				pair2index[newPair] = len(pairList)
				pairList.append(newPair)
				interList.append(np.zeros((labelNum,),dtype=int))
				edgeList.append(newIndex)
			interList[pair2index[newPair]][mode] = 1

	adj = []
	for i in range(len(name2index)):
		adj.append([])
		for j in range(labelNum):
			adj[i].append([])

	for i in range(len(interList)):	
		interaction, pair = interList[i],edgeList[i]	
		for j,inter in enumerate(interaction):
			if inter == 1:
				adj[pair[0]][j].append(pair[1])
				adj[pair[1]][j].append(pair[0]) 

	return pairList,interList,pair2index,adj,edgeList


def seqEmbedding(seqs:list,w2vPath=None,PSSMPath=None):
	fMatrix = embedding.CalAAC(seqs)
	fMatrix = np.concatenate((fMatrix,embedding.CalCJ(seqs)),axis=1) #dimension 343
	#fMatrix = np.concatenate((fMatrix,embedding.CalDPC(seqs)),axis=1) #dimension 400
	fMatrix = np.concatenate((fMatrix,embedding.CalPAAC(seqs)),axis=1)  #dimension 50
	fMatrix = np.concatenate((fMatrix,embedding.CalCTDT(seqs)),axis=1) #dimension 39
	fMatrix = np.concatenate((fMatrix,embedding.CalProtVec(seqs)),axis=1) #dimension 1
	fMatrix = np.concatenate((fMatrix,embedding.CalPos(seqs)),axis=1)  #dimension 1
	fMatrix = fMatrix.astype(float)
	# norms = fMatrix.max(axis=0)
	# nonzero = norms > 0
	# fMatrix[:,nonzero] /= norms[nonzero]
	fMatirx = utils.normalize(fMatrix)

	return fMatrix



def seqEncoding(seqs:list,maxLen=512,modelPath='src/vec5_CTC.txt',w2vPath = 'src/wv_swissProt_size_20_window_16.model',PSSM=None):
	result = []
	paddedSeq = seq_padding(seqs,maxLen)
	v1 = w2v(paddedSeq,w2vPath,20,maxLen)
	v2 = word2type(paddedSeq,modelPath,maxLen)
	result = np.concatenate((v1,v2),axis=2)
	if PSSM:
		v3 = get_PSSM(seqs,PSSM,size=20,maxLen=maxLen)
		result = np.concatenate((result,v3),axis=2)
	return result

def word2type(paddedSeq,modelPath=None,maxLen=512):
	seqNum = len(paddedSeq)
	model = {}
	size = None
	with open(modelPath,'r') as file:
		for line in file:
			line = re.split(' |\t',line)
			model[line[0]] = np.array([float(x) for x in line[1:]])
			if size is None:
				size = len(line[1:])
	
	result = []
	for i in range(seqNum):
		tmp = []
		for j in range(maxLen):
			if paddedSeq[i][j] == ' ':
				tmp.append(np.zeros(size)) 
			else:
				tmp.append(model[paddedSeq[i][j]])
		result.append(tmp)
	result = np.array(result,dtype=float)
	return result


def w2v(paddedSeq ,modelPath=None,size=20,maxLen=512):
	model = Word2Vec.load(modelPath)
	result = []
	seqNum =  len(paddedSeq)
	size = len(model.wv[paddedSeq[0][0]])
	print(f'embedding size {size}')
	print(f'padding with maxLen {maxLen}')	

	for i in range(seqNum):
		tmp = []
		for j in range(maxLen):
			if paddedSeq[i][j] == ' ':
				tmp.append(np.zeros(size))
			else:
				tmp.append(model.wv[paddedSeq[i][j]])
		result.append(np.array(tmp))

	return np.array(result)

import sys
sys.path.append('src')
import pssmpro
def generatePSSM(seqFile:str,output:str,blast_db:str,blast_path='/home/user1/code/software/ncbi-install/bin/psiblast'):
	print(f'\nPSSM input {seqFile} {output} {blast_path} {blast_db}')

	number_of_cores = 8
	pssmpro.create_pssm_profile(seqFile,output,blast_path,blast_db,number_of_cores)
	return


def get_PSSM(idDict,PSSMPath=None,size=20,maxLen=512):
	seqNum = len(idDict)
	fList = [s for s in range(seqNum)]
	for k in idDict.keys():
		fList[idDict[k]] = k
	for i in range(seqNum):
		fList[i] = PSSMPath+'/'+fList[i] +'.pssm'

	check_files_exist(fList)

	result = []
	for fName in fList:
		#print(f'PSSM file {fName}')
		file = open(fName,'r')
		for i in range(3):
			file.readline()	
		count = 0
		tmp = []
		while True:
			line = file.readline()
			if line == '\n' or count >= maxLen:
				break
			count +=1
			line = line.split()[2:22]
			scores = np.array([float(l) for l in line],dtype=float)
			tmp.extend(scores)
		if count < maxLen: #padding
			for i in range(count,maxLen):
				tmp.extend(np.zeros(20,dtype=float))  
		result.append(tmp)
	result = np.array(result,dtype=float)
	# scaler = sklearn.preprocessing.StandardScaler().fit(result)
	# result = scaler.transform(result)
	#'/home/user1/code/PPIKG/method/AFTGAN/string_both_PSSM/9606.ENSP00000000233.pssm'
	return result.reshape((seqNum,-1,size)) 


def seq_padding(seqList,maxLen=512):
	paddedSeq = []
	seqNum = len(seqList)
	for i in range(seqNum):
		if len(seqList[i]) >= maxLen:
			paddedSeq.append(seqList[i][0:maxLen])
		else:
			paddedSeq.append(seqList[i])
			paddedSeq[i] += ' '*(maxLen-len(seqList[i]))
	return paddedSeq

def readSeq(seqPath=""):	
	idDict = dict()
	seqList = []
	check_files_exist([seqPath])
	sFile = open(seqPath,'r')

	count = 0
	while True:
		line = sFile.readline().strip('\n')
		if not line:
			break
		tmp = re.split(',|\t',line)
		if tmp[0] not in idDict: #seq ID
			seqList.append(tmp[1])
			idDict[tmp[0]] = count
			count += 1
	for i,seq in enumerate(seqList):
		tmp = []
		for s in seq:
			if s not in amino_list:
				tmp.append(s)
		if len(tmp) > 0:
			for t in tmp:
				seqList[i] = seqList[i].replace(t,'')
	sFile.close()
	return seqList, idDict


def create_edge_indices(edgeList,interList,class_num = 7):
	result = []
	for i in range(class_num):
		result.append([])
	for i,inter in enumerate(interList):
		for j in range(class_num):
			if inter[j] == 1:
				result[j].append(edgeList[i])
	return result
