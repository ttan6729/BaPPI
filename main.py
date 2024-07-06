import sys
import os
import argparse
import torch
import torch.nn as nn
import math
import random
sys.path.append('src')
import Models
import process
import utils



def str2bool(v):
	"""
	Converts string to bool type; enables command line 
	arguments in the format of '--arg1 true --arg2 false'
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	return



#graph:f1(3D) f2(2D) train_mask val_mask edge_index, edge_attr_1
def train(model,graph,loss_fn,optimizer,device,result_prefix=None,batch_size=512,epochs=100,scheduler=None,global_best_f1=0.0,args=None):
	best_f1,best_epoch = 0.0,0
	result = None
	scaler = torch.cuda.amp.GradScaler()
	for epoch in range(epochs):
		f1_sum,loss_sum ,recall_sum,precision_sum = 0.0,0.0,0.0,0.0
		steps = math.ceil(len(graph.train_mask)/batch_size)
		model.train()
		random.shuffle(graph.train_mask)
		for step in range(steps):
			torch.cuda.empty_cache()
			if step == steps-1:
				train_edge_id = graph.train_mask[step*batch_size:]
			else:
				train_edge_id = graph.train_mask[step*batch_size:(step+1)*batch_size]

			output = model(graph.f1,graph.f2,graph.edge_index,train_edge_id,graph.edge_attr) #edge index: list
			label = graph.edge_attr_1[train_edge_id]
			label = label.type(torch.FloatTensor).to(device)
			loss = loss_fn(output,label)

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()			

		#validation
		model.eval()	
		valid_pre_result_list = []
		valid_label_list = []
		valid_loss_sum = 0.0
		#torch.save()#save model
		steps = math.ceil(len(graph.val_mask) / batch_size)
		saved_pred = []

		with torch.no_grad(): #validation set
			for step in range(steps):
				if step == steps-1:
					valid_edge_id = graph.val_mask[step*batch_size:]
				else:
					valid_edge_id = graph.val_mask[step*batch_size:(step+1)*batch_size]

				output = model(graph.f1,graph.f2,graph.edge_index,valid_edge_id,graph.edge_attr)
				
				label = graph.edge_attr_1[valid_edge_id]
				label = label.type(torch.FloatTensor).to(device)
				loss = loss_fn(output,label)
				valid_loss_sum += loss.item()

				m = nn.Sigmoid()
				pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

				valid_pre_result_list.append(pre_result.cpu().data)
				valid_label_list.append(label.cpu().data)
				saved_pred.append(m(output).to(device).cpu().data )

		valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
		valid_label_list = torch.cat(valid_label_list, dim=0)

		saved_pred = torch.cat(saved_pred,dim=0)
		metrics = utils.Metrictor_PPI(valid_pre_result_list, valid_label_list)
		record = metrics.append_result(result_prefix+'.txt',epoch+1)
		print(record)
		recall_sum += metrics.recall
		precision_sum += metrics.pre
		f1_sum += metrics.microF1
		loss_sum += loss.item()
		valid_loss = valid_loss_sum / steps

		if best_f1 < metrics.microF1:
			best_f1 = metrics.microF1
			best_epoch = epoch
			result =  {'pred':saved_pred,'actual':valid_label_list}
	
	if global_best_f1 < best_f1:
		global_best_f1 = best_f1
		torch.save(result,result_prefix+'.pt')

	return global_best_f1



def get_args_parser():
	parser = argparse.ArgumentParser('BaPPI',add_help=False)
	parser.add_argument('-m',default=None,type=str,help='mode, optinal value: read,bfs,dfs,rand,')
	parser.add_argument('-o',default='output',type=str)
	parser.add_argument('-i',default=None,type=str,help='')
	parser.add_argument('-i1',default=None,type=str,help='sequence file')
	parser.add_argument('-i2',default=None,type=str,help='relation file')
	parser.add_argument('-i3',default=None,type=str,help='file path of test set indices (for read mode)')
	parser.add_argument('-e',default=100,type=int,help='epochs')
	parser.add_argument('-b', default=256, type=int,help='batch size')
	parser.add_argument('-ln', default=2, type=int,help='graph layer num')
	parser.add_argument('-L', default=512, type=int,help='length for sequence padding')
	parser.add_argument('-Loss', default='AS', type=str,help='loss function')
	parser.add_argument('-jk', default=False, type=str2bool,help='use jump knowledege to fuse pair or not')
	parser.add_argument('-ff', default='CnM', type=str,help='option for protein pair representaion')
	parser.add_argument('-hl', default=512, type=int,help='hidden layer')
	parser.add_argument('-sv',default=False,type=str2bool,help='if save dataset path')
	parser.add_argument('-cuda',default=True,type=str2bool,help='if use cuda')
	parser.add_argument('-force',default=True,type=str2bool,help='if write to existed output file')
	parser.add_argument('-PSSM',default=False,type=str2bool,help='if use PSSM')
	# parser.add_argument('-bs', default=False, type=str2bool,help='if use balanced smote')
	# parser.add_argument('-bp', default=0.1, type=float,help='portion for balanced smote')
	return parser

if __name__ == "__main__":
	parser = argparse.ArgumentParser('PPIM', parents=[get_args_parser()])
	args = parser.parse_args()	
	if args.i:
		with open(args.i,'r') as f:
			args.i1 = f.readline().strip()
			args.i2 = f.readline().strip()
	if not args.force:
		if os.path.isfile(args.o+'.txt'):
			print('output name already exists')
			exit()

	device = torch.device('cuda' if torch.cuda.is_available and args.cuda else 'cpu')

	PPIData = process.PPIData(args.i1,args.i2,args)
	graph = PPIData.data #		f1(3D) f2(2D) train_mask val_mask edge_index, edge_attr_1
	#device = torch.device('cpu')
	graph.to(device) 

	# if args.t == 'BaPPI':
	# 	model = Models.BaPPI(in_len=args.L,d1=graph.f1.size()[-1],d2=graph.f2.size()[-1],
	# 	layer_num=args.ln,pool_size=3,cnn_hidden=1,feature_fusion = args.ff).to(device)
	model = Models.BaPPI(in_len=args.L,d1=graph.f1.size()[-1],d2=graph.f2.size()[-1],
	layer_num=args.ln,pool_size=3,cnn_hidden=1,feature_fusion = args.ff).to(device)
	optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
	
	if args.Loss=='CE':
		loss_fn = nn.BCEWithLogitsLoss().to(device)
	elif args.Loss=='AS':
		loss_fn = Models.AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)

	print('begin training')
	best_f1 = train(model,graph,loss_fn,optimizer,device,args.o,args.b,args.e,None,0.0)
	print('end training')
	print(f'output save to file with prefix {args.o}')
	if args.o:
		with open(args.o+'.txt','r+') as file:
			file_data = file.read()
			file.seek(0,0)
			command = ' '.join(arg for arg in sys.argv)
			line = f'command: {command}\n'
			line += f'best_f1: {best_f1}\n'
			line += f'mode: {args.m}\n'		
			line += f'layer num: {args.ln}\n'	
			line += f'filePath: {args.i1} {args.i2}\n'
			if args.i3:
				line += f'valid set path: {args.i3}\n'
			#line += f'model: {args.t}\n'
			line +=f'Loss function: {args.Loss}\n'
			line +=f'max length of seqs: {args.L}\n'
			line +=f'feature 1 shape {graph.f1.size()}\n'# feature 2 shape {graph.f2.size()}\n'
			line +=f'epoch: {args.e}\n'
			line +=f'feature fusion mode: {args.ff}\n'

			file.write(line + '\n' + file_data)

