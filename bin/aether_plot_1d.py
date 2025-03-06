#!/usr/bin/env python3
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md

"""A super-simple Block-based model visualization routine."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from struct import unpack
import argparse
import datetime as dt
from pylab import cm
import os
from netCDF4 import Dataset

iAether_ = 1
iGitm_ = 2

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

#-----------------------------------------------------------------------------
# Read a single GITM file with a variable list
#-----------------------------------------------------------------------------

def read_gitm_one_file(file_to_read, vars_to_read=-1):
    r""" Read list of variables from one GITM file

    Parameters
    ----------
    file_to_read: GITM file to read
    vars_to_read: list of variable NUMBERS to read

    Returns
    -------
    data["time"]: datetime of the file
    data[NUMBER]: data that is read in.
                  NUMBER goes from 0 - number of vars read in (0-3 typical)
    (Also include header information, as described above)
    """

    print("Reading file : "+file_to_read)

    data = {"version": 0, \
            "nLons": 0, \
            "nLats": 0, \
            "nAlts": 0, \
            "nVars": 0, \
            "time": 0, \
            "vars": []}

    f=open(file_to_read, 'rb')

    # This is all reading header stuff:

    endChar='>'
    rawRecLen=f.read(4)
    recLen=(unpack(endChar+'l',rawRecLen))[0]
    if (recLen>10000)or(recLen<0):
        # Ridiculous record length implies wrong endian.
        endChar='<'
        recLen=(unpack(endChar+'l',rawRecLen))[0]

    # Read version; read fortran footer+data.
    data["version"] = unpack(endChar+'d',f.read(recLen))[0]

    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Read grid size information.
    (data["nLons"],data["nLats"],data["nAlts"]) = \
        unpack(endChar+'lll',f.read(recLen))
    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    # Read number of variables.
    data["nVars"]=unpack(endChar+'l',f.read(recLen))[0]
    (oldLen, recLen)=unpack(endChar+'2l',f.read(8))

    if (vars_to_read[0] == -1):
        vars_to_read = np.arange[nVars]

    # Collect variable names.
    for i in range(data["nVars"]):
        var = unpack(endChar+'%is'%(recLen),f.read(recLen))[0]
        (oldLen, recLen)=unpack(endChar+'2l',f.read(8))
        data["vars"].append(var.decode("utf-8"))

    # Extract time. 
    (yy,mm,dd,hh,mn,ss,ms)=unpack(endChar+'lllllll',f.read(recLen))
    data["time"] = dt.datetime(yy,mm,dd,hh,mn,ss,ms*1000)
    #print(data["time"])

    # Header is this length:
    # Version + start/stop byte
    # nLons, nLats, nAlts + start/stop byte
    # nVars + start/stop byte
    # Variable Names + start/stop byte
    # time + start/stop byte

    iHeaderLength = 8 + 4+4 + 3*4 + 4+4 + 4 + 4+4 + \
        data["nVars"]*40 + data["nVars"]*(4+4) + 7*4 + 4+4

    nTotal = data["nLons"]*data["nLats"]*data["nAlts"]
    iDataLength = nTotal*8 + 4+4

    for iVar in vars_to_read:
        f.seek(iHeaderLength+iVar*iDataLength)
        s=unpack(endChar+'l',f.read(4))[0]
        data[iVar] = np.array(unpack(endChar+'%id'%(nTotal),f.read(s)))
        data[iVar] = data[iVar].reshape( 
            (data["nLons"],data["nLats"],data["nAlts"]),order="F")

    f.close()

    return data


#-----------------------------------------------------------------------------
# remap gitm variables
#-----------------------------------------------------------------------------

def remap_variable_names(varsIn):

    mapVars = {
        '[O(!U3!NP)]': '[O] (/m3)',
        '[O!D2!N]': '[O2] (/m3)',
        '[N!D2!N]': '[N2] (/m3)',
        '[N(!U4!NS)]': '[N] (/m3)',
        '[NO]': '[NO] (/m3)',
        '[He]': '[He] (/m3)',
        '[N(!U2!ND)]': '[N_2D] (/m3)',
        '[N(!U2!NP)]': '[N_2P] (/m3)',
        '[H]': '[H] (/m3)',
        '[CO!D2!N]': '[CO2] (/m3)',
        '[O(!U1!ND)]': '[O_1D] (/m3)',
        'Temperature': 'Tn (K)',
        'V!Dn!N(east)': 'Ve (m/s)',
        'V!Dn!N(north)': 'Vn (m/s)',
        'V!Dn!N(up)': 'Vv (m/s)',
        'V!Dn!N(up,O(!U3!NP))': 'Vv_O (m/s)',
        'V!Dn!N(up,O!D2!N)': 'Vv_O2 (m/s)',
        'V!Dn!N(up,N!D2!N)': 'Vv_N2 (m/s)',
        'V!Dn!N(up,N(!U4!NS))': 'Vv_N (m/s)',
        'V!Dn!N(up,NO)': 'Vv_NO (m/s)',
        'V!Dn!N(up,He)': 'Vv_He (m/s)',
        '[O_4SP_!U+!N]': '[O+] (/m3)',
        '[NO!U+!N]': '[NO+] (/m3)',
        '[O!D2!U+!N]': '[O2+] (/m3)',
        '[N!D2!U+!N]': '[N2+] (/m3)',
        '[N!U+!N]': '[N+] (/m3)',
        '[O(!U2!ND)!U+!N]': '[O_2D+] (/m3)',
        '[O(!U2!NP)!U+!N]': '[O_2P+] (/m3)',
        '[H!U+!N]': '[H+] (/m3)',
        '[He!U+!N]': '[He+] (/m3)',
        '[e-]': '[e-] (/m3)',
        'eTemperature': 'Te (K)', 
        'iTemperature': 'Ti (K)',
        'V!Di!N(east)': 'Vie (m/s)',
        'V!Di!N(north)': 'Vin (m/s)',
        'V!Di!N(up)': 'Viv (m/s)'}

    varsOut = []

    for var in varsIn:
        if (var in mapVars):
            varsOut.append(mapVars[var])
        else:
            varsOut.append(var)
    return varsOut

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

    firstfile = args.filelist[0]
    fileType = 0
    if ('.nc' in firstfile):
        fileType = iAether_
    if ('.bin' in firstfile):
        fileType = iGitm_
    if (fileType == 0):
        print('filetype not found... need a .nc or .bin file!')
        exit()
    
    if (args.list):
        if (fileType == iAether_):
            header = read_nc_file(firstfile)
        if (fileType == iGitm_):
            header = read_gitm_one_file(firstfile, vars_to_read=[0])
        for k, v in header.items():
            if ((k != 'vars') and (k != 0)):
                print(k, '-> ', v)
            if (k == 'vars'):
                print('vars : ')
                for i, var in enumerate(v):
                    print(i, var)
        exit()

    if (fileType == iAether_):
        altData = read_nc_file(args.filelist[0], 'z')
        alts = altData['z'][0, 2, 2, :]/1000.0
        
        lonData = read_nc_file(args.filelist[0], 'lon')
        latData = read_nc_file(args.filelist[0], 'lat')

        is1D = determine_isoned(lonData)
        print("  -> is a 1d grid? ", is1D)

        if (not is1D):
            print('This code is for 1d files.....')
            exit()

    if (fileType == iGitm_):
        vars = [0, 1, 2]
        data = read_gitm_one_file(firstfile, vars_to_read = vars)
        alts = data[2][0][0] / 1000.0  # Convert from m to km
        lons = np.degrees(data[0][:, 0, 0])  # Convert from rad to deg
        lats = np.degrees(data[1][0, :, 0])  # Convert from rad to deg
    
    nAlts = len(alts)
    
    var = args.var

    nTimes = len(args.filelist)

    allData = np.zeros((nTimes, nAlts))
    allTimes = []
    
    for iTime, file in enumerate(args.filelist):
        if (fileType == iAether_):
            valueData = read_nc_file(file, var)
            allData[iTime, :] = valueData[var][0, 2, 2, :]
            allTimes.append(valueData['time'])
        if (fileType == iGitm_):
            valueData = read_gitm_one_file(file, [int(var)])
            allData[iTime, :] = valueData[int(var)][0, 0, :]
            allTimes.append(valueData['time'])

    if (fileType == iGitm_):
        var_name_stripped = 'var%03d' % int(var)
        var = valueData['vars'][int(var)]
    else:
        var_name_stripped = var.replace(" ", "")

    varAltered = var
            
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

    outfile = var_name_stripped + '_1d.png'
    print('Writing file : ' + outfile)
    plt.savefig(outfile)
    plt.close()
    
