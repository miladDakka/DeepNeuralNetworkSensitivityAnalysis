import os
import sys
import torch
import itertools
import argparse
import time
import math
from scripts.timing import print_time
from sensitivity_analysis.BlackBoxAnalysis import BlackBoxAnalysis
from tqdm import tqdm
from PIL import Image
import numpy as np
import psutil


'''
The BisectionDropout class is an object that applies iterative bisection dropout 
transformations onto an image to enable analysis of the sensitivity of
model inference to bisection dropout.
'''


def counts_from_phases(phases):
    
    counts = 0
    for i in range(phases):

        counts += 4 ** (i+1)

    return counts


class BisectionDropout(BlackBoxAnalysis):
    # Initialization of image_name and model_path
    # as well as arguments (args) stored persistently
    def __init__(self, image_path, model_path, args):
        
        super(BisectionDropout, self).__init__(image_path, model_path, args)

        self.results_dict = {}

        # Always keep the last box and its contents in memory
        self.last_box = None
        self.last_delta = None

        # ImageNet ordered class list (used in score tracking)
        self.fixed_class_list = None

        # Initialise (c)urrent rectangle height and width, which is fixed
        # and (current) rectangle dimensions,

        self.dims = torch.tensor([self.h, self.w])
        self.dims_c = self.dims

        # Start counter at 0 (this represents a full bisection cycle)
        self.counter = 0

        # Initialize four bisection dropout rectangles
        self.box = torch.tensor([[0,0], self.dims_c])


    def bdrop_experiment(self, args):
        # Here we output the top (n) prediction's index, class and confidence
        num_outputs = args.num_outputs

        with open(args.imagenet_classes) as f:
            # Begin with no dropout and full dropout for baselines
            classes = [line.strip() for line in f.readlines()]
            self.bdrop_outcome(classes, n=num_outputs, box=None)
            self.bdrop_outcome(classes, n=num_outputs, box=self.box)

            # Here we calculate the theoretical number of iterations (see Confluence on Bisection Dropout)
            phases = math.ceil(math.log2(max(self.w, self.h))) # +1 for initial round
            counts = counts_from_phases(phases)
            pbar = tqdm(total=counts)
            
            for _ in range(phases):
                # Split boxes in 4
                self.bisection()
                # Calculate number (range) of iterations for height and width
                hrange = math.ceil(self.h / self.dims_c[0])
                wrange = math.ceil(self.w / self.dims_c[1])

                for i, j in itertools.product(range(hrange), range(wrange)):
                    
                    # Refresh the round score
                    self.get_next_box(i, j)
                    self.bdrop_outcome(classes, n=num_outputs, box=self.box)                        
                    pbar.update(1)

                print(f"Dimensions: {self.dims_c.tolist()}. Number of iterations: {hrange * wrange}")
                if self.dims_c.tolist() == [1,1]:
                    break
                    pbar.close()

            pbar.close()
                
        # Undo final operation (as it never gets undone otherwise)
        [[lx1, ly1],[lx2,ly2]] = self.last_box
        self.tnsr[:, lx1:lx2, ly1:ly2] = self.last_delta.clone()

    # Get model prediction for a particular box drop
    def bdrop_outcome(self, classes, n=1, box=None):

        try:
            self.bdrop(box=box)            
            if box is not None:  
                self.last_box = box
                self.get_norm_result(box, n)
            else:
                self.get_norm_result(None, n)
        except Exception as e:
            print(f"Error during bdrop_outcome() with exception: {e}")

    # Transform the image by dropping out "box"
    def bdrop(self, box=None):
        try:
            if box is not None:
                # undo previous delta at box [(lx1, ly1), (lx2, ly2)]
                if self.last_box is not None:
                    [[lx1, ly1], [lx2, ly2]] = self.last_box
                    self.tnsr[:, lx1:lx2, ly1:ly2] = self.last_delta.clone()

                # apply next delta at box [(x1, y1), (x2, y2)]
                [[x1, y1], [x2, y2]] = box
                self.last_delta = self.tnsr[:, x1:x2, y1:y2].clone()
                self.last_box = box

                # The new tensor to inference on, with dropout applied
                self.tnsr[:, x1:x2, y1:y2] = torch.zeros(3, x2-x1, y2-y1).clone()

            else:
                pass

            # self.img_norm = Image.fromarray(((self.tnsr.transpose(0,2).numpy())*255.999).astype(np.uint8))
            # self.img_norm = self.img_norm.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
            # self.img_norm.show()
            # # time.sleep(0.3)
            # import ipdb; ipdb.set_trace()
            # for proc in psutil.process_iter():
            #     if proc.name() == "display":
            #         proc.kill()
        except Exception as e:
            print(f"Error during bdrop() with exception: {e}")

    # Bisect (2D) current boxes to produce 4*more boxes
    def bisection(self):
        if self.dims_c[0] != 1:
            self.dims_c[0] = max(self.dims_c[0]//2,1)        
        if self.dims_c[1] != 1:
            self.dims_c[1] = max(self.dims_c[1]//2,1)
        
    
    def get_next_box(self, i, j):
        dims_c = self.dims_c.tolist()
        self.box = torch.tensor(
            [[min(  i * dims_c[0], self.h), min( j * dims_c[0], self.w)], \
             [min((i+1)*dims_c[0], self.h),min((j+1)*dims_c[0], self.w)]])


    # Return area in number of pixels
    def get_box_area(self, box):
        
        if box is not None:
            return abs(np.prod(np.array(box[1])-np.array(box[0])))
        else:
            return 1.0


    def get_norm_result(self, box, n):        
        output, percentage = self.get_prediction()
        
        if box == None:
                
                box_label = "[[x,x], [x,x]]"
                _, self.fixed_class_list = torch.sort(output, descending=True)
        
        else:

            box_label = str(box.tolist())

        if n != 1:
            
            self.results_dict[box_label] = [
                float(percentage[idx].item()) for idx in self.fixed_class_list]

        else:
            self.results_dict[box_label] = float(
                percentage[self.fixed_class_list[0][0]].item())


        # Add normalised score to dict        
        if box is None:
            
            self.original_prediction = self.results_dict[box_label]

        drop_area = self.get_box_area(box)
        sigma = self.results_dict[box_label] - self.original_prediction
        
        self.norm_results[box_label] = sigma / drop_area
        self.counter += 1