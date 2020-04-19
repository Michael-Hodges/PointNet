import os
import numpy as np
import pptk
import argparse
import torch

import model
import dataloading
from train import DEVICE

def load_model(model_type, model_path):
	if model_type == 'Vanilla_Classify_Net':
		classifier = model.Vanilla_Classify_Net(output_dim=16)
		classifier.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
	return classifier



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--path', default ='ShapeNet/test_data/02691156/016871.pts', type=str)
	# parser.add_argument('--model', default ='Vanilla_Classify_Net', type=str)
	# parser.add_argument('--load', default= 'model/pn_classify.pt', type=str)
	args = parser.parse_args()
	