import os
import sys
import csv
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
from time import time
from pathlib import Path

from app.app import App
from scripts.timing import print_time
from sensitivity_analysis.NeuralDropout import NeuralDropout as NPA


def get_injection_points(injection_count, width=200, height=200):
    injection_points = []
    # Generate injection_count number of random int points
    for i in range(injection_count):
        x, y = np.random.randint(width), np.random.randint(height)
        injection_points.append([x, y])
    return injection_points


def update_results(npa, i, j, injection_points, filter, score_dict, score_diff_dict, remove):
    for injection_point in injection_points:
        r, c = injection_point
        if remove:
            npa.tnsr[:, r:r+filter[i, j, :, :].shape[0], c:c +
                     filter[i, j, :, :].shape[1]] -= filter[i, j, :, :]
        else:
            npa.tnsr[:, r:r+filter[i, j, :, :].shape[0], c:c +
                     filter[i, j, :, :].shape[1]] += filter[i, j, :, :]
            post_score = npa.model_confidence()
            score_dict[(i+1, j+1)] = post_score
            score_diff_dict[(i+1, j+1)] = post_score - score_dict[(0, 0)]
    return score_dict, score_diff_dict


def neural_dropout(args):
    # get current working directory
    cwd = os.getcwd()

    # define full path to image
    image_name = args.image_name
    image_path = Path(cwd + '/' + image_name)

    # define full path to model
    model_dir = Path("pretrained-models/")
    model_path = model_dir / args.model_name

    npa = NPA(image_path, model_path, args)

    injection_points = get_injection_points(args.injection_count)
    filter = npa.get_filter_from_layer('conv1.weight')

    score_dict = {}
    score_diff_dict = {}
    score_dict[(0, 0)] = npa.model_confidence()
    score_diff_dict[(0, 0)] = 0

    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            score_dict, score_diff_dict = update_results(
                npa, i, j, injection_points, filter, score_dict, score_diff_dict, remove=False)
            score_dict, score_diff_dict = update_results(
                npa, i, j, injection_points, filter, score_dict, score_diff_dict, remove=True)

    for key, value in score_diff_dict.items():
        print(f"For key {key} score differential is {round(value, 2)}.")