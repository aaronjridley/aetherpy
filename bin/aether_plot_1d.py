#!/usr/bin/env python3
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md

"""A super-simple Block-based model visualization routine."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import datetime as dt
from pylab import cm
import os
from netCDF4 import Dataset


# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

def get_args():

    parser = argparse.ArgumentParser(
        description = 'Plot Aether results - super simple!')
    
    parser.add_argument('-list',  \
                        action='store_true', default = False, \
                        help = 'list variables in file')

    parser.add_argument('-var',  \
                        default = "Temperature_neutral", \
                        help = 'variable to plot')

    parser.add_argument('-log',  \
                        action='store_true', default = False, \
                        help = 'Take the log of the variable')
    
    parser.add_argument('filelist', nargs='+', \
                        help = 'list files to use for generating plots')
    
    args = parser.parse_args()

    return args

# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

def epoch_to_datetime(epoch_time):
    """Convert from epoch seconds to datetime.

    Parameters
    ----------
    epoch_time : int
        Seconds since 1 Jan 1965

    Returns
    -------
    dtime : dt.datetime
        Datetime object corresponding to `epoch_time`

    Notes
    -----
    Epoch starts at 1 Jan 1965.

    """

    dtime = dt.datetime(1965, 1, 1) + dt.timedelta(seconds=epoch_time)

    return dtime


# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------


def read_nc_file(filename, file_vars=None):

    """Read all data from a blocked Aether netcdf file.

    Parameters
    ----------
    filename : str
        An Aether netCDF filename
    file_vars : list or NoneType
        List of desired variable neames to read, or None to read all
        (default=None)

    Returns
    -------
    data : dict
        A dictionary containing all data from the netCDF file, including:
        filename - filename of file containing header data
        nlons - number of longitude grids per block
        nlats - number of latitude grids per block
        nalts - number of altitude grids per block
        nblocks - number of blocks in file
        vars - list of data variable names
        time - datetime for time of file
        The dictionary also contains a read_routines.DataArray keyed to the
        corresponding variable name. Each DataArray carries both the variable's
        data from the netCDF file and the variable's corresponding attributes.

    Raises
    --------
    IOError
        If the input file does not exist
    KeyError
        If any expected dimensions of the input netCDF file are not present

    Notes
    -----
    This routine only works with blocked Aether netCDF files.

    """

    # Checks for file existence
    if not os.path.isfile(filename):
        raise IOError(f"unknown aether netCDF blocked file: {filename}")

    # NOTE: Includes header information for easy access until
    #       updated package structure is confirmed
    # Initialize data dict with defaults (will remove these defaults later)
    data = {'filename': filename,
            'units': '',
            'long_name': None}

    with Dataset(filename, 'r') as ncfile:
        # Process header information: nlons, nlats, nalts, nblocks
        data['nlons'] = len(ncfile.dimensions['lon'])
        data['nlats'] = len(ncfile.dimensions['lat'])
        data['nalts'] = len(ncfile.dimensions['z'])
        data['nblocks'] = len(ncfile.dimensions['block'])

        # Included for compatibility
        data['vars'] = [var for var in ncfile.variables.keys()
                        if file_vars is None or var in file_vars]

        # Fetch requested variable data
        if (not (file_vars is None)):
            for key in data['vars']:
                var = ncfile.variables[key]  # key is var name
                data[key] = np.array(var)

        data['time'] = epoch_to_datetime(np.array(ncfile.variables['time'])[0])

    return data

# ----------------------------------------------------------------------
# Determine if the grid is 1D (assume vertical 1D...)
# ----------------------------------------------------------------------

def determine_isoned(lonData):
    nLons = len(lonData['lon'][0, :, 0, 0])
    nLats = len(lonData['lon'][0, 0, :, 0])
    nGCs = 2
    if ((nLons == 2*nGCs + 1) & (nLats == 2*nGCs + 1)):
        return True
    else:
        return False

# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

if __name__ == '__main__':

    # Get the input arguments
    args = get_args()
    
    if (args.list):
        header = read_nc_file(args.filelist[0])
        for k, v in header.items():
            if (k != 'vars'):
                print(k, '-> ', v)
            else:
                print('vars : ')
                for i, var in enumerate(v):
                    print(i, var)
        exit()

    altData = read_nc_file(args.filelist[0], 'z')
    lonData = read_nc_file(args.filelist[0], 'lon')
    latData = read_nc_file(args.filelist[0], 'lat')

    is1D = determine_isoned(lonData)
    print("  -> is a 1d grid? ", is1D)

    if (not is1D):
        print('This code is for 1d files.....')
        exit()

    alts = altData['z'][0, 2, 2, :]/1000.0
    nAlts = len(alts)
    
    var = args.var
    varAltered = var

    nTimes = len(args.filelist)

    allData = np.zeros((nTimes, nAlts))
    allTimes = []
    
    for iTime, file in enumerate(args.filelist):
        valueData = read_nc_file(file, var)
        allData[iTime, :] = valueData[var][0, 2, 2, :]
        allTimes.append(valueData['time'])

    if (args.log):
        allData = np.log10(allData)
        varAltered = 'log(' + varAltered + ')'
        
    mini = np.min(allData)
    maxi = np.max(np.abs(allData))

    if (mini < 0):
        cmap = cm.bwr
        mini = -maxi
    else:
        cmap = cm.plasma
        
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_axes([0.075, 0.1, 0.90, 0.8])
    y2d, x2d = np.meshgrid(alts, allTimes)
    cax = ax.pcolormesh(x2d, y2d, allData,
                        vmin = mini, vmax = maxi, cmap = cmap)
                        
    ax.set_ylabel('Altitude (km)')
    
    cbar = fig.colorbar(cax, ax = ax, shrink = 0.75, pad = 0.02)
    cbar.set_label(varAltered, rotation=90)

    var_name_stripped = var.replace(" ", "")
    outfile = var_name_stripped + '_1d.png'
    print('Writing file : ' + outfile)
    plt.savefig(outfile)
    plt.close()
    
