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

def load_historical_forcing(end_date: str):
    '''This will grab the historical forcing file and grab the observed
    meteorology for the 180 days leading up to the end_date specified. This
    will be combined with the forecasts to provide a continuous timeseries
    of meteorology.
    '''
    ds = 'an xarray dataset with all variables for the subbasin of interest'
    # extract the time period that runs up until the first day of hindcast
    # denoted by "end_date"
    # return the extracted dataset in a format aligning with the hindcast
   
    return ds

def load_hindcasted_forcing(ensemble_member: str):
    forcing_path = WATERFORAG_ROOT / 'meteorology' / 'hindcasts' / 'netcdf'
    files = list(glob.glob(f"{str(forcing_path)}/YAKIMA_bcsd_nmme_hindcasts_CFSv2_ENS{ensemble_member}_*.nc"))
    if len(files) == 0:
        raise RuntimeError(f'No forcing file file found for ensemble member {ensemble_member}')
    ds = xr.open_mfdataset(files)
    return ds

def load_historical_streamflow(

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
        # want to add the historical forcings as well like this:
        # ds_historical = load_historical_forcing(self.end_date)
        
        # here we'll want to subset the forcing for the specific basin we're working with
        # like this:
        # ds_subset = mask(ds, basin_mask)
        # if mean:
            # convert to basin mean values

        # add the code that converts the forcings into the shape appropriate for pytorch
        # load_streamflow for the location 
        return ds

class Model(nn.Module):
    """Implementation of a single layer LSTM network. Taken directly from sample notebook"""
    
    def __init__(self, hidden_size: int, dropout_rate: float=0.0):
        """Initialize model
        
        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # create required layer
        self.lstm = nn.LSTM(input_size=5, hidden_size=self.hidden_size, 
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.
        
        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.
        
        :return: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)
        
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1,:,:]))
        return pred

def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.tqdm_notebook(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        
def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.
    
    :return: Two torch Tensors, containing the observations and 
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
            
    return torch.cat(obs), torch.cat(preds)
        
def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val


ensemble_member = '23'
period = 'train'
ds_train = gridmet_NMME(ensemble_member=ensemble_member, period=period)


