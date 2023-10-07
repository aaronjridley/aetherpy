#!/usr/bin/env python3
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md

"""A super-simple Block-based model visualization routine."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from netCDF4 import Dataset
import os
import datetime as dt
from pylab import cm

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
                        default = "Temperature", \
                        help = 'variable to plot')

    parser.add_argument('-alt', metavar = 'alt', default =200, type = int, \
                        help = 'altitude :  alt in km (closest)')
    
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
    alts = altData['z'][0,0,0,:]/1000.0
    d = np.abs(alts - args.alt)
    iAlt = np.argmin(d)

    lonData = read_nc_file(args.filelist[0], 'lon')
    latData = read_nc_file(args.filelist[0], 'lat')
    
    print('Taking a slice at alt = ',alts[iAlt], ' km')

    nBlocks = altData['nblocks']
    nLons = altData['nlons']
    nLats = altData['nlats']

    var = args.var

    for file in args.filelist:
    
        valueData = read_nc_file(file, var)

        title = var + ' at %4.0f km at ' % alts[iAlt] + \
            valueData['time'].strftime('%B %d, %Y; %H:%M:%S UT')
    
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_axes([0.075, 0.1, 0.95, 0.8])

        mini = np.min(valueData[var][:, 1:-1, 1:-1, iAlt])
        maxi = np.max(np.abs(valueData[var][:, 1:-1, 1:-1, iAlt]))
        if (mini < 0):
            cmap = cm.bwr
            mini = -maxi
        else:
            cmap = cm.plasma

        for iBlock in range(nBlocks):
            lon2d = lonData['lon'][iBlock, 1:-1, 1:-1, iAlt]
            lat2d = latData['lat'][iBlock, 1:-1, 1:-1, iAlt]
            v2d = valueData[var][iBlock, 1:-1, 1:-1, iAlt]
            cax = ax.pcolormesh(lon2d, lat2d, v2d, \
                                vmin = mini, vmax = maxi, cmap = cmap)
            
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_ylim([-90.0, 90.0])
        ax.set_xlim([0.0, 360.0])
        ax.set_title(title)
        cbar = fig.colorbar(cax, ax=ax, shrink = 0.75, pad=0.02)
        cbar.set_label(var,rotation=90)

        var_name_stripped = var.replace(" ", "")
        sAlt = 'alt%03d' % iAlt
        sTime = valueData['time'].strftime('%Y%m%d_%H%M%S')
        outfile = var_name_stripped + '_' + sTime + '_' + sAlt + '.png'

        print('Writing file : ' + outfile)
        plt.savefig(outfile)
        plt.close()
