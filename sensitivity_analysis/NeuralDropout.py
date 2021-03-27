import os
import sys
import torch
import torchvision.models as models
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sensitivity_analysis.BlackBoxAnalysis import BlackBoxAnalysis
from scripts.timing import print_time
from torchvision import transforms, utils


'''
The BlackBoxAnalysis class takes as inputs:
1) A link to a model architecture (pretrained or otherwise):        model_path
2) A set of one or more input images:                               tnsr
3) Additional user inputs in the form of arguments:                 args

BlackBoxAnalysis is a parent class to the analysis classes (e.g. PixelDropout),
providing shared features and parameters with all children classes.
'''


class NeuralDropout(BlackBoxAnalysis):
    # Initialization of image_name and model_path
    # as well as arguments (args) stored persistently
    def __init__(self, image_path, model_path, args):
        
        super(NeuralDropout, self).__init__(image_path, model_path, args)
        
        self.results_dict = {}

        # Always keep the last box and its contents in memory
        self.last_pixel = None
        self.last_delta = torch.tensor([0, 0, 0])

        # Initialise ImageNet ordered class list (used in score tracking)
        # Establish fixed class list based on highest confidence prediction
        self.output, self.percentages = self.get_prediction()
        _, self.fixed_class_list = torch.sort(self.output, descending=True)
    
    
    def get_filter_from_layer(self, layer):
        return self.state_dict[layer].data.cpu().clone() 


    def plot_filter(self, layer, grid, rows, args, nrow):        
        plt.figure(figsize=(nrow,rows))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.ioff()
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        plt.savefig(f"{args.output_folder}/{layer}.png")
        plt.close()


    def plot_filters(self, args, ch=0, allkernels=False, nrow=8, padding=1):
        # initialise NeuralPathwaysAnalysis object
        for layer in self.state_dict.keys():          
            try:
                filter = self.get_filter_from_layer(layer)
                n,c,w,h = filter.shape
                if allkernels: filter = filter.view(n*c, -1, w, h)
                elif c != 3: filter = filter[:,ch,:,:].unsqueeze(dim=1)
                rows = np.min((filter.shape[0] // nrow + 1, 64))    
                grid = utils.make_grid(filter, nrow=nrow, normalize=True, padding=padding)  
                self.plot_filter(layer, grid, rows, args, nrow=nrow)
            except:
                print(f"Something up with layer {layer}")
                pass


    def model_confidence(self):
        self.output, self.percentages = self.get_prediction()
        _, self.fixed_class_list = torch.sort(self.output, descending=True)
        self.confidence = self.percentages[self.fixed_class_list[0][0]].item()
        return self.confidence