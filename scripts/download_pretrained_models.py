import os
import torch
import torchvision.models as models

from pathlib import Path

# this snippet is taken from "https://pytorch.org/docs/stable/torchvision/models.html"
# to download pretrained models from online
resnet18 =              models.resnet18(pretrained=True)
alexnet =               models.alexnet(pretrained=True)
squeezenet1_0 =         models.squeezenet1_0(pretrained=True)
vgg16 =                 models.vgg16(pretrained=True)
densenet161 =           models.densenet161(pretrained=True)
googlenet =             models.googlenet(pretrained=True)
shufflenet_v2_x1_0 =    models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2 =          models.mobilenet_v2(pretrained=True)
resnext50_32x4d =       models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 =       models.wide_resnet50_2(pretrained=True)
mnasnet1_0 =            models.mnasnet1_0(pretrained=True)

cwd = os.getcwd()
model_path = "/pretrained-models/"
model_dir = Path(cwd + model_path)

model_names = [
    "resnet18.pth","alexnet.pth","squeezenet1_0.pth","vgg16.pth",
    "densenet161.pth","googlenet.pth","shufflenet_v2_x1_0.pth","mobilenet_v2.pth",
    "resnext50_32x4d.pth","wide_resnet50_2.pth","mnasnet1_0.pth"
]

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(resnet18.state_dict(),           model_dir / model_names[0] ) #44.7M
torch.save(alexnet.state_dict(),            model_dir / model_names[1] )  #233M
torch.save(squeezenet1_0.state_dict(),      model_dir / model_names[2] ) #4.79M
torch.save(vgg16.state_dict(),              model_dir / model_names[3] )  #528M
torch.save(densenet161.state_dict(),        model_dir / model_names[4] )  #110M
torch.save(googlenet.state_dict(),          model_dir / model_names[5] ) #49.7M
torch.save(shufflenet_v2_x1_0.state_dict(), model_dir / model_names[6] ) #8.79M
torch.save(mobilenet_v2.state_dict(),       model_dir / model_names[7] ) #13.6M
torch.save(resnext50_32x4d.state_dict(),    model_dir / model_names[8] ) #95.8M
torch.save(wide_resnet50_2.state_dict(),    model_dir / model_names[9] )  #132M
torch.save(mnasnet1_0.state_dict(),         model_dir / model_names[10]) #16.9M