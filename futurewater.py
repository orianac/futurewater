# imports
from pathlib import Path
from typing import Tuple, List

import os
import glob
import gcsfs
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import xarray as xr
import netCDF4
import scipy
import nc_time_axis
import cftime

# Globals
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # This line checks if GPU is available
WATERFORAG_ROOT = Path('/pool0/data/orianac/waterforag/')
