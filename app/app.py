import argparse

class App:
    def __init__(self):
        
        self.args = None

    def run(self):
        # define argument parser
        self.parser = argparse.ArgumentParser(
            description='Retrieve model predictions for an image.')

        self.parser.add_argument('--algorithm',
                            metavar='AG', default="pixel_dropout",
                            help='Sensitivity analysis algorithm (default: pixel_dropout).')
        self.parser.add_argument('--model-name',
                            metavar='MN', default="resnet18.pth",
                            help='Filepath to model.')
        self.parser.add_argument('--image-name',
                            metavar='IP', default="sample_inputs/dog.png", type=str,
                            help='Filepath to image.')
        self.parser.add_argument('--x',
                            metavar='X', default=224, type=int,
                            help='X component of pixel index.')
        self.parser.add_argument('--y',
                            metavar='Y', default=224, type=int,
                            help='Y component of pixel index.')
        self.parser.add_argument('--num-outputs',
                            metavar='NO', default=1, type=int,
                            help='Number of output classes.')
        self.parser.add_argument('--df-output',
                            metavar='DO', default="output.csv", type=str,
                            help='Output path for dataframe (Default: "output.csv").')
        self.parser.add_argument('--img-output',
                            metavar='IO', default="pixel_dropout.png", type=str,
                            help='Output path for image (Default: "pixel_dropout.png").')
        self.parser.add_argument('--device',
                            metavar='DV', default="cuda", type=str,
                            help='Device (Default: "cuda").')
        self.parser.add_argument('--device-id',
                            metavar='DV', default=0, type=int,
                            help='Device id (Default: 0).')
        self.parser.add_argument('--n',
                            metavar='N', default=10, type=int,
                            help='Top n pixels displayed in visualization.')
        self.parser.add_argument('--output-folder',
                            metavar='OF', default="temporary_folder",
                            help='General flag for output folder useful in many scripts.')
        self.parser.add_argument('--imagenet-classes',
                            metavar='IC', default="sample_inputs/imagenet_classes.txt", type=str,
                            help='Location of imagenet_classes.txt or similar file.')
        self.parser.add_argument("--injection-count", type=str, default=10,
                            help="Number of injections to randomly insert into image.")
    

        self.args = self.parser.parse_args()