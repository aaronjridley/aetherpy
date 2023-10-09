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

    parser.add_argument('-cut',  \
                        default = "alt", \
                        help = 'plane to plot (alt, lat, lon)')

    parser.add_argument('-alt', metavar = 'alt', default = 200, type = int, \
                        help = 'altitude :  alt in km (closest)')
    parser.add_argument('-lat', metavar = 'lat', default = 0, type = int, \
                        help = 'latitude :  lat in deg (closest)')
    parser.add_argument('-lon', metavar = 'lon', default = 180, type = int, \
                        help = 'longitude :  lon in deg (closest)')
    
    parser.add_argument('-scatter',  \
                        action='store_true', default = False, \
                        help = 'make scatter plot (instead of contour)')
    
    
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

def plot_alt_plane(valueData, lonData, latData, var, iAlt, \
                   doScatter = False):

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
        if (doScatter):
            cax = ax.scatter(lon2d, lat2d, c = v2d, \
                             vmin = mini, vmax = maxi, cmap = cmap)
        else:
            cax = ax.pcolormesh(lon2d, lat2d, v2d, \
                                vmin = mini, vmax = maxi, cmap = cmap)
            
    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_ylim([-90.0, 90.0])
    ax.set_xlim([0.0, 360.0])

    return cax


# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

def plot_lon_plane(valueData, lonData, latData, altData, var, lon, \
                   doScatter = False):

    # need to cycle through all of the blocks to get the min and max:

    nBlocksPlotted = 0
    for iBlock in range(nBlocks):
        lons = lonData['lon'][iBlock, 1:-1, 0, 0]
        print(lons)
        if ((lon >= np.min(lons)) & (lon < np.max(lons))):

            d = np.abs(lons - lon)
            iLon = np.argmin(d)

            mini = np.min(valueData[var][:, iLon, 1:-1, 1:-1])
            maxi = np.max(np.abs(valueData[var][:, iLon, 1:-1, 1:-1]))
            nBlocksPlotted += 1

    if (nBlocksPlotted == 0):
        print('Cycled through all of the blocks and could not find ')
        print('requested longitude of : ', lon)
        exit()

    if (mini < 0):
        cmap = cm.bwr
        mini = -maxi
    else:
        cmap = cm.plasma

    nBlocksPlotted = 0
    for iBlock in range(nBlocks):
        lons = lonData['lon'][iBlock, 1:-1, 0, 0]
        if ((lon >= np.min(lons)) & (lon < np.max(lons))):

            d = np.abs(lons - lon)
            iLon = np.argmin(d)

            sPos = '%4.0f deg longitude ' % lons[iLon]
            sPosFile = 'lon%03d' % iLon
            alt2d = altData['z'][iBlock, iLon, 1:-1, 1:-1]
            lat2d = latData['lat'][iBlock, iLon, 1:-1, 1:-1]
            print(lat2d[10,:])
            v2d = valueData[var][iBlock, iLon, 1:-1, 1:-1]
            if (doScatter):
                cax = ax.scatter(lat2d, alt2d, c = v2d, \
                                 vmin = mini, vmax = maxi, cmap = cmap)
            else:
                cax = ax.pcolormesh(lat2d, alt2d, v2d, \
                                    vmin = mini, vmax = maxi, cmap = cmap)
            
    ax.set_xlabel('Altitude (km)')
    ax.set_xlabel('Latitude (deg)')
    ax.set_xlim([-90.0, 90.0])
    #ax.set_xlim([0.0, 360.0])

    return cax, sPos, sPosFile

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
    
    nBlocks = altData['nblocks']
    nLons = altData['nlons']
    nLats = altData['nlats']

    var = args.var
    doScatter = args.scatter

    for file in args.filelist:
    
        valueData = read_nc_file(file, var)
    
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_axes([0.075, 0.1, 0.95, 0.8])

        if (args.cut == 'alt'):
            alts = altData['z'][0,0,0,:]/1000.0
            d = np.abs(alts - args.alt)
            iAlt = np.argmin(d)
            print('Taking a slice at alt = ',alts[iAlt], ' km')
            sPos = '%4.0f km ' % alts[iAlt]
            sPosFile = 'alt%03d' % iAlt
            cax = plot_alt_plane(valueData, lonData, latData, var, iAlt, doScatter)
        if (args.cut == 'lon'):
            cax, sPos, sPosFile = plot_lon_plane(valueData, lonData, latData, altData, var, args.lon, doScatter)

        title = var + ' at ' + sPos + ' at ' + \
            valueData['time'].strftime('%B %d, %Y; %H:%M:%S UT')
        ax.set_title(title)
        cbar = fig.colorbar(cax, ax=ax, shrink = 0.75, pad=0.02)
        cbar.set_label(var,rotation=90)

        var_name_stripped = var.replace(" ", "")
        sTime = valueData['time'].strftime('%Y%m%d_%H%M%S')
        outfile = var_name_stripped + '_' + sTime + '_' + sPosFile + '.png'

        print('Writing file : ' + outfile)
        plt.savefig(outfile)
        plt.close()
