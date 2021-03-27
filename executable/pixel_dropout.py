import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

from app.app import App
from scripts.timing import print_time
from sensitivity_analysis.PixelDropout import PixelDropout as PD
from visualization.visualization import draw_target, df2ptval, df2hist1d, df2hist2d


def pixel_dropout(args):
    # get current working directory
    cwd = os.getcwd()

    # define full path to image
    image_name = args.image_name
    image_path = Path(cwd + '/' + image_name)

    # define full path to model
    model_dir = Path("pretrained-models/")
    model_path = model_dir / args.model_name

    # initialise PixelDropout object
    p = PD(image_path, model_path, args)

    # get pdrop experiment results
    p.pdrop_experiment(args)

    # unnormalize image
    p.unNormalize_image()
    
    # set up dataframes
    df = pd.DataFrame(list(p.results_dict.items()),
                      columns=["Pixel", "Prediction"])
    csv_name = os.path.basename(args.df_output)
    df.to_csv(args.output_folder + csv_name, index=False)
    pdrop_scores = df.iloc[1:]

    # generate and save 1D (histogram) and 2D (density) plots
    _, values = df2ptval(pdrop_scores)
    values = np.array(values)
    df2hist1d(values, os.path.basename(image_name))
    df2hist2d(values, os.path.basename(image_name))

    # get top n positive and negative pixels
    n = args.n
    lg = pdrop_scores.nlargest(n, 'Prediction')
    sm = pdrop_scores.nsmallest(n, 'Prediction')
    max_pts, _ = df2ptval(lg)
    min_pts, _ = df2ptval(sm)

    # draw top n positive (green) and negative (red) pixels
    for pt in max_pts:
        draw_target(p.img_norm, [pt[0], pt[1]], fill=(255, 0, 0))
    for pt in min_pts:
        draw_target(p.img_norm, [pt[0], pt[1]], fill=(0, 255, 0))

    # save image and terminate instantiation
    p.img_norm.save(args.output_folder + os.path.basename(args.img_output))
    p.terminate()