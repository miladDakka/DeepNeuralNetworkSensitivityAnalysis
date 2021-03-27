import os
import sys
from time import time

from app.app import App
from scripts.timing import print_time
from executable.pixel_dropout import *
from executable.neural_dropout import *
from executable.bisection_dropout import *


def main(args):
    
    if args.algorithm == "pixel_dropout" or args.algorithm == "p":
        pixel_dropout(args)
     
    if args.algorithm == "neural_dropout" or args.algorithm == "n":
        neural_dropout(args)

    if args.algorithm == "bisection_dropout" or args.algorithm == "b":
        bisection_dropout(args)


if __name__ == "__main__":
    t1 = time()

    app = App()
    app.run()
    main(app.args)

    t2 = time()
    print_time(t2, t1, msg="Start to finish!")