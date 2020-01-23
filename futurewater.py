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

def load_historical_forcing():
    ds = 'an xarray dataset with all variables for the subbasin of interest'
    return ds



def load_hindcasted_forcing():
    forcing_path = WATERFORAG_ROOT / 'meteorology' / 'hindcast'
    ds = 'an xarray dataset with all forecasted variables for the subbasin of interest'
    return ds
    

class gridmet_NMME(Dataset):
# this class will house the datasets to be used for training and validation for a selected basin
# it is specific to the gridmet historical dataset and the NMME climate forecasts
  #  def __init__():
    def __init():
        return 'hello'
