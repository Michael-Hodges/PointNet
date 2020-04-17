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

def train(data_path, act):

	if act=="classify":
		classifier = model.Vanilla_Classify_Net(output_dim=16)
		train_data = dataloading.ShapeNetClassify(data_path, 'train')
		train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True,
			sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
			pin_memory=True, drop_last=True, timeout=0,
			worker_init_fn=None)

		val_data = dataloading.ShapeNetClassify(data_path, 'val')
		val_loader = data.DataLoader(dataset=val_data, batch_size=64, shuffle=True,
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
				print("dataloader_inputs: {}".format(inputs.shape))
				classifier.train()
				inputs = inputs.permute(0,2,1)
				print("labels: {}".format(labels.shape))
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				optimizer.zero_grad()

				outputs = classifier(inputs)
				loss = loss_func(outputs, labels)
				loss.backward()
				optimizer.step()
				running_loss += loss.item()/labels.size(0)
				_, predicted = torch.max(outputs.data, 1)
				total_samples += labels.size()
				total_correct += (predicted == labels).sum().item()
			train_accuracy = 100 * total_correct / total

			val_accuracy = 0
			val_correct = 0
			val_total = 0
			val_running_loss = 0
			with torch.no_grad():
				for _, (inputs, labels) in enumerate(val_loader):
					inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
					outputs = classifier(inputs)
					val_loss = loss(outputs, labels)
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
		torch.save(net.state_dict(), "./model/pn_classify.pt")

if __name__== '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='shapenet', type=str)
	parser.add_argument('--action', default='classify', type=str)
	parser.add_argument('--path', default='ShapeNet', type=str)

	args = parser.parse_args()

	train(args.path, args.action)

	
