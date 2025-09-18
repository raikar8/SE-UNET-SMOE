"""
Train interface for speech enhancement!
You can just run this file.
"""
import os
import argparse
import torch
import options
import utils
import datetime
import random
import numpy as np
import time
from dataloader import create_dataloader


######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
print(opt)