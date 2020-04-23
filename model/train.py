# Pytorch Files
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

#
import numpy as np
import argparse

# Project Files
import model
import dataloading

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 8

def train(data_path, act):

	if act=="classify":
		classifier = model.Vanilla_Classify_Net(output_dim=16)
		classifier.to(DEVICE)
		train_data = dataloading.ShapeNetClassify(data_path, 'train')
		train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,
			sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
			pin_memory=True, drop_last=True, timeout=0,
			worker_init_fn=None)

		val_data = dataloading.ShapeNetClassify(data_path, 'val')
		val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True,
			sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
			pin_memory=True, drop_last=True, timeout=0,
			worker_init_fn=None)

		loss_func = nn.CrossEntropyLoss()
		optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0)
		print('{0}, {1}, {2}, {3}, {4}'.format('Epoch', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
		epoch = 0
		prev_val_accuracy = 0
		val_accuracy = 0
		epoch_delta = -999
		while (np.abs(epoch_delta)>0.1) and (epoch < EPOCHS):
			running_loss = 0.0
			train_accuracy = 0.0
			total_correct = 0
			total_samples = 0
			for step, (inputs, labels) in enumerate(train_loader):
				classifier.train()
				# inputs = inputs.permute(0,2,1)
				# print("Input Shape: {}".format(inputs.shape))
				# print("labels: {}".format(labels.shape))
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				optimizer.zero_grad()

				outputs = classifier(inputs)
				# print("Output Shape: {}".format(outputs.shape))
				loss = loss_func(outputs, labels)
				loss.backward()
				optimizer.step()
				running_loss += loss.item()/labels.size(0)
				_, predicted = torch.max(outputs.data, 1)
				total_samples += labels.size(0)
				# print("Total Samples: {}".format(total_samples))
				total_correct += (predicted == labels).sum().item()
			train_accuracy = 100 * total_correct / total_samples

			val_accuracy = 0
			val_correct = 0
			val_total = 0
			val_running_loss = 0
			with torch.no_grad():
				for _, (inputs, labels) in enumerate(val_loader):
					classifier.eval()
					# inputs = inputs.permute(0,2,1)
					inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
					outputs = classifier(inputs)
					val_loss = loss_func(outputs, labels)
					val_running_loss += val_loss.item()/labels.size(0)
					_, predicted = torch.max(outputs.data, 1)
					val_total += labels.size(0)
					val_correct += (predicted == labels).sum().item()
			val_accuracy = 100 * val_correct / val_total
			epoch_delta = val_accuracy - prev_val_accuracy
			prev_val_accuracy = val_accuracy
			epoch += 1
			print('{0:5d}, {1:10.3f}, {2:11.3f}, {3:9.3f}, {4:10.3f}'.format(epoch, running_loss, train_accuracy, val_running_loss, val_accuracy))

		print("Training terminated. Saving model...")
		torch.save(classifier.state_dict(), "./model/pn_classify.pt")

	if act=="segment":
		classifier = model.Vanilla_Segment_Net(output_dim=6)
		classifier.to(DEVICE)
		train_data = dataloading.ShapeNetSegment(data_path, 'train')
		train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,
			sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
			pin_memory=True, drop_last=True, timeout=0,
			worker_init_fn=None)

		val_data = dataloading.ShapeNetSegment(data_path, 'val')
		val_loader = data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True,
			sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
			pin_memory=True, drop_last=True, timeout=0,
			worker_init_fn=None)

		loss_func = nn.CrossEntropyLoss()
		optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0)
		print('{0}, {1}, {2}, {3}, {4}'.format('Epoch', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
		epoch = 0
		prev_val_accuracy = 0
		val_accuracy = 0
		epoch_delta = -999
		while (np.abs(epoch_delta)>0.1) and (epoch < EPOCHS):
			running_loss = 0.0
			train_accuracy = 0.0
			total_correct = 0
			total_samples = 0
			for step, (inputs, labels, _) in enumerate(train_loader):
				batch_size = labels.size(0)
				point_num = labels.size(1)
				classifier.train()
				# inputs = inputs.permute(0,2,1)
				# print("Input Shape: {}".format(inputs.shape))
				# print("labels: {}".format(labels.shape))
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				optimizer.zero_grad()

				outputs = classifier(inputs)
				# print("Output Shape: {}".format(outputs.shape))
				loss = loss_func(outputs, labels)
				loss.backward()
				optimizer.step()
				t_samps = labels.view(batch_size*point_num, -1).squeeze()
				# print("tsamps: {}".format(len(t_samps)))
				running_loss += loss.item()/batch_size
				# print("runningloss: {}".format(running_loss))
				_, predicted = torch.max(outputs.data, 1)
				# print(labels.shape)
				# print("labels.size(0): {}".format(labels.size(0)))
				total_samples += len(t_samps)
				# print("Total Samples: {}".format(total_samples))
				total_correct += (predicted == labels).sum().item()
			train_accuracy = 100 * total_correct / total_samples
			# print("train_accuracy: {}".format(train_accuracy))

			val_accuracy = 0
			val_correct = 0
			val_total = 0
			val_running_loss = 0
			with torch.no_grad():
				for _, (inputs, labels, _) in enumerate(val_loader):
					batch_size = labels.size(0)
					point_num = labels.size(1)
					classifier.eval()
					# inputs = inputs.permute(0,2,1)
					inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
					outputs = classifier(inputs)
					val_loss = loss_func(outputs, labels)
					t_samps = labels.view(batch_size*point_num, -1).squeeze()
					val_running_loss += val_loss.item()/batch_size
					_, predicted = torch.max(outputs.data, 1)
					val_total += len(t_samps)
					val_correct += (predicted == labels).sum().item()
			val_accuracy = 100 * val_correct / val_total
			epoch_delta = val_accuracy - prev_val_accuracy
			prev_val_accuracy = val_accuracy
			epoch += 1
			print('{0:5d}, {1:10.3f}, {2:11.3f}, {3:9.3f}, {4:10.3f}'.format(epoch, running_loss, train_accuracy, val_running_loss, val_accuracy))

		print("Training terminated. Saving model...")
		torch.save(classifier.state_dict(), "./model/pn_Segment.pt")


if __name__== '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='shapenet', type=str)
	parser.add_argument('--action', default='segment', type=str) #option classify | segment
	parser.add_argument('--path', default='ShapeNet', type=str)

	args = parser.parse_args()

	train(args.path, args.action)

	
