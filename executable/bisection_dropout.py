import os
import sys
import csv
import argparse
import pandas as pd
from time import time
from pathlib import Path

from app.app import App
from scripts.timing import print_time
from visualization.visualization import draw_target, df2ptval
from sensitivity_analysis.BisectionDropout import BisectionDropout as BD


def bisection_dropout(args):
    # get current working directory
    cwd = os.getcwd()

    # define full path to image
    image_name = args.image_name
    image_path = Path(cwd + '/' + image_name)

    # define full path to model
    model_dir = Path("pretrained-models/")
    model_path = model_dir / args.model_name

    # initialise BisectionDropout object
    b = BD(image_path, model_path, args)

    # get pdrop experiment results
    b.bdrop_experiment(args)

    # unnormalize image
    b.unNormalize_image()

    # set up dataframes
    df = pd.DataFrame(list(b.norm_results.items()),
                      columns=["Box", "Prediction"])
    df.to_csv(args.df_output, index=False)
    bdrop_scores = df.iloc[1:]

    # get top n positive and negative pixels
    n = args.n
    lg = bdrop_scores.nlargest(n, 'Prediction')
    sm = bdrop_scores.nsmallest(n, 'Prediction')
    max_pts, _ = df2ptval(lg)
    min_pts, _ = df2ptval(sm)

    # draw top n positive (green) and negative (red) pixels
    for pt in max_pts:
        draw_target(b.img_norm, [pt[0], pt[1]], fill=(255, 0, 0))
    for pt in min_pts:
        draw_target(b.img_norm, [pt[0], pt[1]], fill=(0, 255, 0))

    # display image and terminate instantiation
    b.img_norm.show()
    b.img_norm.save(args.img_output)
    b.terminate()
