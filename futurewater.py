# imports
from pathlib import Path
from typing import Tuple, List

import os
import glob
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

def load_hindcasted_forcing(ensemble_member: str):
    forcing_path = WATERFORAG_ROOT / 'meteorology' / 'hindcasts' / 'netcdf'
    files = list(glob.glob(f"{str(forcing_path)}/YAKIMA_bcsd_nmme_hindcasts_CFSv2_ENS{ensemble_member}_*.nc"))
    print(forcing_path)
    if len(files) == 0:
        raise RuntimeError(f'No forcing file file found for ensemble member {ensemble_member}')
    ds = xr.open_mfdataset(files)
    return ds

class gridmet_NMME(Dataset):
# this class will house the datasets to be used for training and validation for a selected basin
# it is specific to the gridmet historical dataset and the NMME climate forecasts
    def __init__(self, ensemble_member: str, period: str=None):
        """Initialize Dataset containing the data of a single basin.

        :param ensemble_member: string of integer from 1 to 24 (non zero-padded)
                                denoting the ensemble member of the data in use.
        """
        self.ensemble_member = ensemble_member
        self.period = period
        self.meteorology = self._load_data()

    def _load_data(self):
        """Load input and output data from text files."""
        ds = load_hindcasted_forcing(self.ensemble_member)
        return ds
ensemble_member = '23'
ds_train = gridmet_NMME(ensemble_member)
