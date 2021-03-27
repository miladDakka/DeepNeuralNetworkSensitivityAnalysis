import os
import sys
from PIL import Image, ImageDraw
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kde
from ast import literal_eval
from math import sqrt


def draw_target(img, px=[10, 10], fill=(0, 0, 255)):
    w = img.width
    h = img.height
    draw = ImageDraw.Draw(img)
    for i, j in itertools.product(range(-w, w), range(-h, h)):
        if i**2+j**2 == 0:
            draw.point([px[1]+j, px[0]+i], fill)
    del draw


def df2ptval(df):
    pts = []
    vals = []
    for index, row in df.iterrows():
        # Convert the Pixel and Prediction columns
        pts.append(literal_eval(row['Pixel']))
        vals.append(row['Prediction'])
    return pts, vals


def df2hist1d(values, filename, n=50, dpi=300, xlabel="Prediction score (%)", ylabel="Frequency (counts)"):
    plt.close()
    plt.hist(values, bins=n)
    plt.title(filename)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('/home/polydatum/polydatum/polydatum/testing_outputs/gpdrop_histogram_' + filename, dpi=dpi)
    plt.close()


def df2hist2d(values, filename, dpi=300, legendlabel = "Prediction score (%)"):
    plt.close()
    n = int(sqrt(len(values)))
    values_grid = values.reshape(n, n)
    color_map = plt.cm.get_cmap('jet').reversed()
    plt.imshow(values_grid, cmap=color_map, interpolation='none')
    cbar = plt.colorbar().set_label(legendlabel, rotation=90)
    plt.title(filename)
    plt.savefig('/home/polydatum/polydatum/polydatum/testing_outputs/gpdrop_densityplot_' + filename, dpi=dpi)
    plt.close()