import torch
import torch.nn as nn
import model
import datasets
from critic import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder=model.model1_encoder
decoder=model.model1_decoder