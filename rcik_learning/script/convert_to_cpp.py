import torch
from model import RCIK
import numpy as np

# An instance of your model.
# folder = '/media/mincheul/db/rcik_dataset/fetch_arm/'
folder = './'
name = 'batch_sampler_1000_best.pt'
model = RCIK(7)

checkpoint = torch.load(folder + "models/" + name)
model.load_state_dict(checkpoint)
sm = torch.jit.script(model)
sm.save("/home/mincheul/Desktop/" + name)
