import os
import sys
import torch
import torchvision.models as models
import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import time
from PIL import Image
from scripts.timing import print_time
from torchvision import transforms


'''
The BlackBoxAnalysis class takes as inputs:
1) A link to a model architecture (pretrained or otherwise):        model_path
2) A set of one or more input images:                               tnsr
3) Additional user inputs in the form of arguments:                 args

BlackBoxAnalysis is a parent class to the analysis classes (e.g. PixelDropout),
providing shared features and parameters with all children classes.
'''


class BlackBoxAnalysis:
    # User initialised
    def __init__(self, image_path, model_path, args):
        # t1 = time()
        self.args = args

        self.device_id = self.args.device_id
        self.device = (self.args.device + ':' + str(self.device_id)
                       ) if self.args.device == 'cuda' else (self.args.device)
        self.device = torch.device(self.device)

        self.model_path = model_path
        self.model_filename = os.path.basename(self.model_path)
        self.model_name = os.path.splitext(self.model_filename)[0]
        self.model_folder = os.path.dirname(self.model_path)
        if self.model_filename in os.listdir(self.model_folder):
            self.model = eval(f"models.{self.model_name}()")
        else:
            sys.exit(f"{self.model_name} not found in {self.model_folder} folder.")
        self.state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(self.state_dict)

        self.model.to(self.device)
        # t2 = time(); print_time(t2,t1,msg="Load model")
        self.model.eval()
        # t3 = time(); print_time(t3,t2,msg="Evaluate model")

        self.image_path = image_path
        self.img = Image.open(self.image_path).convert('RGB')
        self.tnsr = self.img2tnsr()

        # Finally, initialise the results list
        self.norm_results = {}

    # TODO: allow more transform options and user interfaceability
    def img2tnsr(self):
        # transforms for model trained on ImageNet data
        # t1 = time()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.w = 224
        self.h = 224
        self.resize = 256
        transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop((self.w, self.h)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std
            )
            ])
        # print_time(time(),t1,msg="Transform image")
        return transform(self.img)

    def get_prediction(self):
        # TODO: if isinstance(tns,int) or if (tns == None) and isinstance(self.tnsr,int):
        # t1 = time()
        batch_t = torch.unsqueeze(self.tnsr, 0).to(self.device)
        # t2 = time(); print_time(t2,t1,msg="unsqueeze")
        with torch.no_grad():
            output = self.model(batch_t)
            # t3 = time(); print_time(t3,t2,msg="output")
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        return output, percentage

    # ensure that all lingering stored memory is cleared
    def terminate(self):
        self.img.close()
        del self.img
        del self.tnsr
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()


    class UnNormalize(object):    
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tnsr):
            """
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:
                Tensor: Normalized image.
            """
            for t, m, s in zip(tnsr, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return tnsr

    def unNormalize_image(self):
        # TODO: Make sure unnormalise works here the same way it does on the .ipynb version
        unorm = self.UnNormalize(mean=self.mean, std=self.std)
        self.img_unnorm = Image.fromarray(((unorm(self.tnsr).transpose(0,2).numpy())*255.999).astype(np.uint8))
        self.img_unnorm = self.img_unnorm.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
        self.img_norm = Image.fromarray(((self.tnsr.transpose(0,2).numpy())*255.999).astype(np.uint8))
        self.img_norm = self.img_norm.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)

    