import os
import sys
import torch
import itertools
import argparse
import time
from scripts.timing import print_time
from sensitivity_analysis.BlackBoxAnalysis import BlackBoxAnalysis
from tqdm import tqdm


'''
The PixelDropout class is an object that applies iterative pixel-by-pixel 
transformations onto an image to enable analysis of the sensitivity of
model inference to pixel dropout.
'''


class PixelDropout(BlackBoxAnalysis):
    # Initialization of image_name and model_path
    # as well as arguments (args) stored persistently
    def __init__(self, image_path, model_path, args):
        
        super(PixelDropout, self).__init__(image_path, model_path, args)
        
        self.results_dict = {}

        # Always keep the last box and its contents in memory
        self.last_pixel = None
        self.last_delta = torch.tensor([0, 0, 0])

        # ImageNet ordered class list (used in score tracking)
        self.fixed_class_list = None

    def pdrop_experiment(self, args):
        # Here we output the top (n) prediction's index, class and confidence
        n = args.num_outputs
        x = args.x
        y = args.y
        
        try:
            user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
        except KeyError:
            user_paths = []

        with open(user_paths[0] + '/' + args.imagenet_classes) as f:
        
            classes = [line.strip() for line in f.readlines()]
            self.pdrop_outcome(classes, n=n, px_idx=None)
            
            pbar = tqdm(total=(x*y))
            for i, j in itertools.product(range(x), range(y)):
        
                self.pdrop_outcome(classes, n=n, px_idx=[i, j])
                pbar.update(1)
            pbar.close()
        
        # Undo final operation (as it never gets undone otherwise)
        l_i, l_j = self.last_pixel
        self.tnsr[:, l_i, l_j] = self.last_delta.clone()


    def pdrop_outcome(self, classes, n=1, px_idx=None):        
        try:
            self.pdrop(px_idx=px_idx)
            output, percentage = self.get_prediction()       
            if px_idx is not None:       
                i, j = px_idx       
                if n != 1:       
                    self.results_dict[f"[{i},{j}]"] = [
                        float(percentage[idx].item()) for idx in self.fixed_class_list]
                else:       
                    self.results_dict[f"[{i},{j}]"] = float(
                        percentage[self.fixed_class_list[0][0]].item())       
            else:       
                _, self.fixed_class_list = torch.sort(output, descending=True)
                if n != 1:       
                    self.results_dict["[x,x]"] = [
                        float(percentage[idx].item()) for idx in self.fixed_class_list]
                else:       
                    self.results_dict[f"[x,x]"] = float(
                        percentage[self.fixed_class_list[0][0]].item())
        except RuntimeError:            
            pass

    # this functions takes an image (or the original image)
    def pdrop(self, px_idx=None):
        try:        
            if px_idx is not None:                
                # undo previous delta at pixel (l_i, l_j)
                if self.last_pixel is not None:        
                    l_i, l_j = self.last_pixel
                    self.tnsr[:, l_i, l_j] = self.last_delta.clone()
        
                # apply next delta at pixel (i, j)
                i, j = px_idx
                self.last_delta = self.tnsr[:, i, j].clone()
                self.last_pixel = [i, j]
                self.tnsr[:, i, j] = torch.tensor([0, 0, 0]).clone()        
            else:        
                pass        
        except RuntimeError:        
            pass