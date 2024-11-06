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
import glob

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

    parser.add_argument('-cut',  \
                        default = "alt", \
                        help = 'plane to plot (alt, lat, lon)')

    parser.add_argument('-alt', metavar = 'alt', default = 200, type = int, \
                        help = 'altitude :  alt in km (closest)')
    parser.add_argument('-lat', metavar = 'lat', default = 0, type = int, \
                        help = 'latitude :  lat in deg (closest)')
    parser.add_argument('-lon', metavar = 'lon', default = 180, type = int, \
                        help = 'longitude :  lon in deg (closest)')
    
    parser.add_argument('-polar',  \
                        action='store_true', default = False, \
                        help = 'plot polar plots also (3 plots!)')
    
    parser.add_argument('-scatter',  \
                        action='store_true', default = False, \
                        help = 'make scatter plot (instead of contour)')
    
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

    print('Reading file : ', filename)
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
# This is a function that will examine the last block to determine
# whether the grid is a cubesphere or not.  The last block should
# be the north polar region.  If it is a spherical grid, the spacing
# should be uniform in longitude.  If it is a cubesphere grid, it
# won't be uniform
# ----------------------------------------------------------------------

def determine_cubesphere(lonData):
    nLatsD2 = int(len(lonData['lon'][-1, 0, :, 0])/2)
    # Only take true cells, so we can not have to worry about wrapping:
    lon1d = lonData['lon'][-1, 2:-2, nLatsD2, 0]
    if (len(lon1d) == 1):
        return False
    dLon = lon1d[1:-1] - lon1d[0:-2]
    mindLon = np.min(dLon)
    maxdLon = np.max(dLon)
    if ((maxdLon - mindLon) > 0.01):
        return True
    else:
        return False

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
# This adds labels to the polar plots, including latitudes, local times,
# and a label in the upper left
# ----------------------------------------------------------------------

def set_labels_polar(axis, label, isSouth = False, \
                     no00 = False, no06 = False, no12 = False, no18 = False):

    xlabels = ['06', '12', '18', '00']
    if (no00):
        xlabels[3] = ''
    if (no06):
        xlabels[0] = ''
    if (no12):
        xlabels[1] = ''
    if (no18):
        xlabels[2] = ''
    if (isSouth):
        ylabels = ['-80', '-70', '-60', '-50']
    else:
        ylabels = ['80', '70', '60', '50']
    axis.set_xticks(np.arange(0,2*np.pi,np.pi/2))
    axis.set_yticks(np.arange(10,45,10))
    axis.set_xticklabels(xlabels)
    axis.set_yticklabels(ylabels)
    axis.set_ylim(0,45)

    ang = np.pi*3.0/4.0
    axis.text(ang, np.abs(40/np.cos(ang)), label)
    
    return
    
# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

def get_corrected_lon(lonData, lonv, iBlock, iAlt):
    lon2d = lonData[lonv][iBlock, 1:-1, 1:-1, iAlt]
    if ((np.min(lon2d) < 45.0) & (np.max(lon2d) > 315.0)):
        if (np.median(lon2d) < 90.0):
            lon2d[lon2d > 315] = lon2d[lon2d > 315] - 360.0
        if (np.median(lon2d) > 270.0):
            lon2d[lon2d < 45] = lon2d[lon2d < 45] + 360.0
    return lon2d
    
# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

def plot_alt_plane(valueData, lonData, latData, altData, var, alt, \
                   ax, 
                   isCubeSphere, \
                   doScatter = False, doPolar = False):

    reallyScatter = doScatter
    
    alts = altData['z'][0,0,0,:]/1000.0
    d = np.abs(alts - alt)
    iAlt = np.argmin(d)
    print('Taking a slice at alt = ',alts[iAlt], ' km')
    sPos = '%4.0f km ' % alts[iAlt]
    sPosFile = 'alt%03d' % iAlt

    mini = np.min(valueData[var][:, 1:-1, 1:-1, iAlt])
    maxi = np.max(np.abs(valueData[var][:, 1:-1, 1:-1, iAlt]))
    if (mini < 0):
        cmap = cm.bwr
        mini = -maxi
    else:
        cmap = cm.plasma

    # First go through the non-polar plot and plot all blocks:
    for iBlock in range(nBlocks):

        # if this is a cubesphere, we don't want to plot the polar cells
        # in the main region as pcolors, since they get messed up
        # but, we we didn't ask for scatter, and we have corners, we
        # want to use the corners

        # first, if we specified scatter, don't use corners:
        if ((doScatter) or (not ('lonc' in lonData))):
            useCorners = False
        else:
            useCorners = True
            # now, only turn off corners if cubesphere and in polar region:
            # First check for CubeSphere:
            if (isCubeSphere):
                latv = 'lat'
                lat2d = latData[latv][iBlock, 1:-1, 1:-1, iAlt]
                # check for polar region:
                if (np.abs(np.mean(lat2d)) > 45.0):
                    reallyScatter = True
                    useCorners = False
                
        if (useCorners):
            lonv = 'lonc'
            latv = 'latc'
        else:
            lonv = 'lon'
            latv = 'lat'
            
        lat2d = latData[latv][iBlock, 1:-1, 1:-1, iAlt]
        lon2d = get_corrected_lon(lonData, lonv, iBlock, iAlt)
        v2d = valueData[var][iBlock, 1:-1, 1:-1, iAlt]

        if (reallyScatter):
            lon2d = (lon2d + 360) % 360
            cax = ax[0].scatter(lon2d, lat2d, c = v2d, \
                             vmin = mini, vmax = maxi, cmap = cmap)
        else:
            cax = ax[0].pcolormesh(lon2d, lat2d, v2d, \
                                vmin = mini, vmax = maxi, cmap = cmap)
            
    if (doPolar):
        for iBlock in range(nBlocks):
            lonv = 'lon'
            latv = 'lat'
            if ((not doScatter) and ('lonc' in lonData)):
                lonv = 'lonc'
                latv = 'latc'
            lat2d = latData[latv][iBlock, 1:-1, 1:-1, iAlt]
            lon2d = get_corrected_lon(lonData, lonv, iBlock, iAlt)
            v2d = valueData[var][iBlock, 1:-1, 1:-1, iAlt]
        
            if (np.max(lat2d) > 50.0):
                t2d = lon2d * np.pi / 180.0 - np.pi/2.0
                r2d = 90.0 - lat2d
                if (doScatter):
                    ax[1].scatter(t2d, r2d, c = v2d, \
                                  cmap = cmap, vmin = mini, vmax = maxi)
                else:
                    ax[1].pcolor(t2d, r2d, v2d, \
                                 cmap = cmap, vmin = mini, vmax = maxi)
        
            if (np.min(lat2d) < -45.0):
                t2d = lon2d * np.pi / 180.0 - np.pi/2.0
                r2d = 90.0 + lat2d
                if (doScatter):
                    ax[2].scatter(t2d, r2d, c = v2d, \
                                  cmap = cmap, vmin = mini, vmax = maxi)
                else:
                    ax[2].pcolor(t2d, r2d, v2d, \
                                 cmap = cmap, vmin = mini, vmax = maxi)
            
    ax[0].set_xlabel('Longitude (deg)')
    ax[0].set_ylabel('Latitude (deg)')
    ax[0].set_ylim([-90.0, 90.0])
    ax[0].set_xlim([0.0, 360.0])
    if (doPolar):
        set_labels_polar(ax[1], 'North')
        set_labels_polar(ax[2], 'South', isSouth = True)

    return cax, sPos, sPosFile


# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------

def plot_lon_plane(valueData, lonData, latData, altData, var, lon, \
                   ax, \
                   doScatter = False,
                   doPlotLog = False):

    # need to cycle through all of the blocks to get the min and max:

    nBlocksPlotted = 0
    mini = 1e32
    maxi = -1e32
    nAlts = len(lonData['lon'][0, 0, 0, :])
    nLats = len(lonData['lon'][0, 0, :, 0])
    
    for iBlock in range(nBlocks):
        lons = lonData['lon'][iBlock, 1:-1, int(nLats/2), int(nAlts/2)]
        if ((lon >= np.min(lons)) & (lon < np.max(lons))):
            d = np.abs(lons - lon)
            iLon = np.argmin(d)
            if (np.min(valueData[var][:, iLon, 1:-1, 1:-1]) < mini):
                mini = np.min(valueData[var][:, iLon, 1:-1, 1:-1])
            if (np.max(np.abs(valueData[var][:, iLon, 1:-1, 1:-1])) > maxi):
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

    if (doPlotLog):
        maxi = np.log10(maxi)
        mini = np.log10(mini)

    nBlocksPlotted = 0
    for iBlock in range(nBlocks):
        lons = lonData['lon'][iBlock, 1:-1, int(nLats/2), int(nAlts/2)]
        print(lons)
        d = np.abs(lons - lon)
        #if ((lon >= np.min(lons)) & (lon < np.max(lons))):
        if (np.min(d) < 20.0):
            d = np.abs(lons - lon)
            iLon = np.argmin(d)
            sPos = '%4.0f deg longitude ' % lons[iLon]
            sPosFile = 'lon%03d' % iLon
            alt2d = altData['z'][iBlock, iLon, 1:-1, 1:-1]/1000.0
            lat2d = latData['lat'][iBlock, iLon, 1:-1, 1:-1]
            v2d = valueData[var][iBlock, iLon, 1:-1, 1:-1]
            if (doPlotLog):
                v2d = np.log10(v2d)
            if (doScatter):
                cax = ax[0].scatter(lat2d, alt2d, c = v2d, \
                                 vmin = mini, vmax = maxi, cmap = cmap)
            else:
                cax = ax[0].pcolormesh(lat2d, alt2d, v2d, \
                                    vmin = mini, vmax = maxi, cmap = cmap)
            
    ax[0].set_xlabel('Altitude (km)')
    ax[0].set_xlabel('Latitude (deg)')
    ax[0].set_xlim([-90.0, 90.0])
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

    # If there is a corner file and the user does not want scatter plots,
    # we should be able to use the corners, if that file exists:

    useCorners = False
    if (not args.scatter):
        cornerFile = glob.glob("3DCOR*")
        if (len(cornerFile) > 0):
            altDataC = read_nc_file(cornerFile[0], 'z')
            lonDataC = read_nc_file(cornerFile[0], 'lon')
            latDataC = read_nc_file(cornerFile[0], 'lat')
            useCorners = True
    altData = read_nc_file(args.filelist[0], 'z')
    lonData = read_nc_file(args.filelist[0], 'lon')
    latData = read_nc_file(args.filelist[0], 'lat')

    if (useCorners):
        altData['zc'] = altDataC['z']
        latData['latc'] = latDataC['lat']
        lonData['lonc'] = lonDataC['lon']
    
    isCube = determine_cubesphere(lonData)
    print("  -> is a cubesphere grid? ", isCube)
    is1D = determine_isoned(lonData)
    print("  -> is a 1d grid? ", is1D)
    
    nBlocks = altData['nblocks']
    nLons = altData['nlons']
    nLats = altData['nlats']

    var = args.var
    doScatter = args.scatter
    varAltered = var
    if (args.log):
        varAltered = 'log(' + varAltered + ')'

    for file in args.filelist:
    
        valueData = read_nc_file(file, var)
        doOutput = False

        if (is1D):
            if (file == args.filelist[0]):
                fig = plt.figure(figsize = (10,8))
                ax = fig.add_axes([0.075, 0.1, 0.90, 0.8])
                
            alts = altData['z'][0, 2, 2, :]/1000.0
            data = valueData[var][0, 2, 2, :]
            ax.plot(data, alts)
            ax.set_xlabel(var)
            ax.set_ylabel('Alts (km)')
            sPosFile = 'oned'

            if (file == args.filelist[-1]):
                doOutput = True
            
        else:
        
            fig = plt.figure(figsize = (10,8))
            ax = []
            doOutput = True
            if (args.polar):
                # bottom full globe
                ax.append(fig.add_axes([0.075, 0.06, 0.95, 0.5]))
                size = 0.37
                # upper left
                ax.append(fig.add_axes([0.06, 0.59, size, size], \
                                       projection='polar'))
                # upper right
                ax.append(fig.add_axes([0.6, 0.59, size, size], \
                                       projection='polar'))
            else:
                ax.append(fig.add_axes([0.075, 0.1, 0.95, 0.8]))
            if (args.cut == 'alt'):
                cax, sPos, sPosFile = plot_alt_plane(valueData, \
                                                     lonData, latData, altData, \
                                                     var, args.alt, ax, \
                                                     isCube, doScatter, args.polar)
            if (args.cut == 'lon'):
                cax, sPos, sPosFile = plot_lon_plane(valueData, \
                                                     lonData, latData, altData, \
                                                     var, args.lon, ax, doScatter, \
                                                     args.log)

                title = varAltered + ' at ' + sPos + ' at\n' + \
                    valueData['time'].strftime('%B %d, %Y; %H:%M:%S UT')
                ax[0].set_title(title)
                cbar = fig.colorbar(cax, ax = ax, shrink = 0.75, pad = 0.02)
                cbar.set_label(varAltered, rotation=90)

        if (doOutput):
            var_name_stripped = var.replace(" ", "")
            sTime = valueData['time'].strftime('%Y%m%d_%H%M%S')
            outfile = var_name_stripped + '_' + sTime + '_' + sPosFile + '.png'

            print('Writing file : ' + outfile)
            plt.savefig(outfile)
            plt.close()
