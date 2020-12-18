# System tools
import os,pickle,inspect,pdb,socket,glob,sys,string

# Utilities
import numpy as np
import pandas as pd
from length import length
from quick_image import *
from FITS_tools.hcongrid import hcongrid
from recursive_glob import recursive_glob
from tqdm import tqdm,trange
from statsmodels.robust.scale import mad
from panstarrs_query import panstarrs_query_sorted
import scipy.optimize as opt
from scipy.special import erfinv
from collections import OrderedDict
from statsmodels.stats.diagnostic import normal_ad
import scipy

# Utilities
from length import length
from select import select

# Astropy stuff
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, Angle, EarthLocation
# depreciated. Use simple_norm
#from astropy.visualization import scale_image
from astropy import units as u
from photutils import CircularAperture, SkyCircularAperture, CircularAnnulus
from astropy.table import Table, Column
from astropy.io.ascii import SExtractor

# Plotting stuff
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from astroplan import Observer
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord


'''
To Do:
------
 - Put FWHM header key in final image from astrometry.net. This info should be available
   in the auxiliary files created in astrometry.net output.

'''



######################################################################
# Thacher Observatory information
######################################################################
######################################################################
import ephem
obs = ephem.Observer()
obs.lat = '34 28 00.5'
obs.lon = '-119 10 38.5'
obs.elevation = 494.7
obs_location = EarthLocation.from_geodetic(np.degrees(obs.lon),np.degrees(obs.lat),obs.elevation)

# Create observatory object
thacher = Observer(location=obs_location, name="Thacher Observatory",
                   timezone="US/Pacific")

######################################################################
# Routines for setting up the pipeline
######################################################################
######################################################################

#----------------------------------------------------------------------#
# get_paths: get appropriate paths for data reduction                  #
#----------------------------------------------------------------------#

def get_paths(obstype=None,targname=None):
    '''
    Description
    -----------
    Take account of the user and computer that is executing this script
    then return the appropriate data and outpath.

    Inputs
    ------
    None

    Outputs
    -------
    dictionary of paths. Keys are:
    data = (string) path to raw data
    output = (string) output path
    archive = (string) path to archive
    execpath = (string) path to directory of this file
    rawpath = (string) path to directory where raw data is stored

    Example
    -------
    paths = get_paths()

    '''

    if obstype is None:
        print("Must specify obstype when establishing paths (for organizational reasons)")
        return None

    paths = {'output':None, 'archive':None, 'execpath':None, 'raw':None, 'dropbox':None}

    # Environment variables
    user = os.environ['USER']
    home = os.environ['HOME']

    # Host not present on all operating systems
    try:
        host = os.environ['HOST']
    except:
        host = socket.gethostname()

# Path from where command was executed
    paths['execpath'] = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/'

    # Data and archive paths
    if host == 'bellerophon':
        paths['archive'] = "/home/administrator/Dropbox (Thacher)/Archive/"
        paths['raw'] = "/home/administrator/Dropbox (Thacher)/Astronomy/ACP_Imaging/"
        paths['dropbox'] = "/home/administrator/Dropbox (Thacher)/"
        if targname is not None:
            paths['output'] = '/home/'+user+'/'+obstype+'/'+targname+'/'
            if obstype=='SNS':
                paths['output'] = '/home/'+user+'/'+obstype+'/'+'SNS_'+targname+'/'
        else:
            paths['output'] = '/home/'+user+'/'+obstype+'/'

    if host == 'mojave' or host == 'munch.local':
        paths['archive'] = '/Users/jonswift/Dropbox (Thacher)/Archive/'
        paths['raw'] = '/Users/jonswift/Dropbox (Thacher)/Astronomy/ACP_Imaging/'
        paths['dropbox'] = '/Users/jonswift/Dropbox (Thacher)/'
        if targname is not None:
            paths['output'] = home+'/Astronomy/ThacherObservatory/'+obstype+'/'+targname+'/'
        else:
            paths['output'] = home+'/Astronomy/ThacherObservatory/'+obstype+'/'

    if host == 'Tobys-MacBook-Pro.local' and user == 'tobyarculli':
        paths['archive'] ='/Users/tobyarculli/Dropbox (Thacher)/Astronomy Archive/'
        paths['dropbox'] ='/Users/tobyarculli/Dropbox (Thacher)/'
        if targname is not None:
            paths['output']  = home + '/Dropbox (Thacher)/Astronomy/'+obstype+'/'+targname+'/'
        else:
            paths['output']  = home + '/Dropbox (Thacher)/Astronomy/'+obstype+'/'

    if host == 'HadrienTangs-MacBook-Pro.local' and user == 'hadrientang':
        paths['archive'] ='/Users/hadrien/Dropbox (Thacher)/Astronomy Archive/'
        paths['dropbox'] ='/Users/hadrien/Dropbox (Thacher)/'
        if targname is not None:
            paths['output']  = home + '/Dropbox (Thacher)/Astronomy/'+obstype+'/'+targname+'/'
        else:
            paths['output']  = home + '/Dropbox (Thacher)/Astronomy/'+obstype+'/'

    # Add your local path here if desired...
    # if host == 'yourhost' and user == 'you':


    # Create output directory if it does not exist already
    if paths['output'] is not None:
        if not os.path.isdir(paths['output']):
            mkdircmd = 'mkdir '+paths['output']
            os.system(mkdircmd)

    return paths


#----------------------------------------------------------------------#
# get_dates: get all dates target has been observed                    #
#----------------------------------------------------------------------#

def get_dates(targname=None,obstype=None):
    '''
    Description
    -----------
    Return the dates for which a target was observed.

    Inputs
    ------
    targname:   (string) Name of the target to search for
    obstype:    (string) Name of observation type

    Output
    ------
    Pandas dataframe with obs_summary information

    Example
    -------
    obs = get_dates('SN2018fgc')


    '''
    if obstype is not None:
        if targname is not None:
            search_str = obstype+'_'+targname
        else:
            search_str = obstype

    elif targname is None:
        print("Must input a targname and/or obstype!")
        return None

    else:
        search_str = targname

    # Obstype is empty string because file beginning has already
    # been determined above
    paths = get_paths(targname=search_str,obstype='')

    dates = []
    for root, dirs, files in os.walk(paths['archive']):
        for file in files:
            if file.startswith(search_str) and file.endswith("solved.fits"):
                dates.append(root.split('/')[-1])
    dates = np.array(dates)
    dates.sort()
    dates = np.unique(dates)

    return dates



#----------------------------------------------------------------------#
# get_files: get all files in archive for a source                     #
#----------------------------------------------------------------------#


def get_files(prefix=None,tag='',date=None,suffix='solved.fits',raw=False,clean=False,datapath=None):

    '''
    Description
    -----------
    Returns list of files with a user defined prefix and suffix withing a
    specified directory.

    Optionally "cleans" the filenames of special characters


    Inputs
    ------


    Output
    ------


    Example
    -------

    '''

    if datapath is None:
        # Get paths
        paths = get_paths(obstype='')

        # Search archive or raw data
        key = 'raw' if raw else 'archive'

        datapath = paths[key]
        if date is not None:
            datapath = datapath+date+'/'

    files = recursive_glob(datapath,prefix+"*"+tag+"*"+suffix)
    fct = len(files)

    # Work around for difficult single quote and inconsistent file naming convention
    # due to filter names
    if clean:
        for file in files:
            inname  = file.replace("'","\\'")
            outname =  file.replace("'","")
            if inname != outname:
                mvcmd = "mv "+inname+" "+outname
                os.system(mvcmd)

        files = [file.replace("'","") for file in files]

        for file in files:
            inname  = file
            outname =  file.replace("p.fts",".fts")
            if inname != outname:
                mvcmd = "mv "+inname+" "+outname
                os.system(mvcmd)

        files = [file.replace("p.fts",".fts") for file in files]

    return files,fct


#----------------------------------------------------------------------#
# make_summary_file:                                                   #
#----------------------------------------------------------------------#

def make_summary_file(obstype=None,targname=None,clobber=False):
    '''
    Description
    -----------
    Make a summary file for a new target

    Inputs
    ------
    targname:   (string) Name of the target
    clobber:   (boolean) Overwrite summary file?

    Output
    ------
    None

    Example
    -------
    make_summary_file(obstype='SNe',targname='AT2018cow')

    '''

    paths = get_paths(obstype=obstype,targname=targname)

    fname = paths['output']+targname+'_summary.csv'
    test = glob.glob(fname)
    if len(test) == 1 and not clobber:
        print('Summary file already exists!')
        return
    else:
        dates = get_dates(targname=targname)
        length = len(dates)
        obs = OrderedDict()
        obs['date'] = dates
        obs['setting'] = np.ones(length).astype('int')
        obs['use'] = ['Y' for i in range(length)]
        obs['phot_corr'] = ['Default' for i in range(length)]
        obs['color_corr'] = ['Y' for i in range(length)]
        obs['time_start'] = ['' for i in range(length)]
        obs['time_end'] = ['' for i in range(length)]
        obs['am_start'] = ['' for i in range(length)]
        obs['am_end'] = ['' for i in range(length)]
        obs['comment'] = ['' for i in range(length)]

        df = pd.DataFrame.from_dict(obs)

        print('Writing summary file in output directory')
        df.to_csv(fname,index=False)

    return


#----------------------------------------------------------------------#
# update_summary_file:                                                 #
#----------------------------------------------------------------------#

def update_summary_file(obstype=None,targname=None):
    """
    Description:
    ------------
    Check dates that targname has been observed in Archive and append
    new line to summary file if not already represented

    Inputs:
    -------
    obstype (string): Observation type
    targname (string): Target name

    Example:
    -------
    update_summary_file(obstype='Monitoring',targname='PKS1510-089)
    """

    paths = get_paths(targname=targname,obstype=obstype)

    fname = paths['output']+targname+'_summary.csv'
    test = glob.glob(fname)
    if len(test) == 0:
        print('Summary file does not exist! Use make_summary_file to generate obs summary.')
        return
    if len(test) == 1:
        data = pd.read_csv(fname).fillna('')
        dates = get_dates(targname).astype('int')
        newdates = [d for d in dates if d not in data['date'].values]

        length = len(newdates)
        if length == 0:
            print('There is nothing to update in summary file')
            return data
        else:
            obs = OrderedDict()
            obs['date'] = newdates
            obs['setting'] = np.ones(length).astype('int')
            obs['use'] = ['Y' for i in range(length)]
            obs['phot_corr'] = ['Default' for i in range(length)]
            obs['color_corr'] = ['Y' for i in range(length)]
            obs['time_start'] = ['' for i in range(length)]
            obs['time_end'] = ['' for i in range(length)]
            obs['am_start'] = ['' for i in range(length)]
            obs['am_end'] = ['' for i in range(length)]
            obs['comment'] = ['' for i in range(length)]
            df = pd.DataFrame.from_dict(obs)
            final = pd.concat([data,df])
            # Need to sort with respect to date before writing out
            print('Updating summary file...')
            final.to_csv(fname,index=False)
            return final


#----------------------------------------------------------------------#
# get_summary: get observing summary file for a target                 #
#----------------------------------------------------------------------#

def get_summary(obstype=None,targname=None):
    '''
    Description
    -----------
    Return the contents of the obs_summary file that lives in the sn repo

    Inputs
    ------
    None

    Output
    ------
    Pandas dataframe with obs_summary information

    Example
    -------
    obs = get_summary()

    '''

    # Path to this file
    paths= get_paths(obstype=obstype,targname=targname)

    # Read obs_summary file in output directory
    fname = paths['output']+targname+'_summary.csv'
    obs = pd.read_csv(fname)

    return obs



#----------------------------------------------------------------------#
# get_obs: parse summary file for a date and target                    #
#----------------------------------------------------------------------#

def get_obs(date,obstype=None,targname=None):
    '''
    Description
    -----------
    Parse information from the obs_summary file

    Inputs
    ------
    date = (string) YYYYMMDD

    Output
    ------
    info dictionary

    Example
    -------

    info = get_obs('20170910')

    '''
    # Check
    if obstype is None or targname is None:
        print('You must supply an obstype and a targname for this function!')
        return None

    paths = get_paths(obstype=obstype,targname=targname)

    obs = get_summary(obstype=obstype,targname=targname)

    if np.sum(obs['date'] == int(date)) == 0:
        print('No photometry for '+date+'!')
        return None

    d = date

    info = {'date':date,
            'setting': obs['setting'][obs['date'] == int(date)].values[0],
            'use': obs['use'][obs['date'] == int(date)].values[0],
            'phot_corr': obs['phot_corr'][obs['date'] == int(date)].values[0],
            'color_corr': obs['color_corr'][obs['date'] == int(date)].values[0],
            'comment': obs['comment'][obs['date'] == int(date)].values[0],
            'flag_time': [obs['time_start'][obs['date'] == int(date)].values[0],
                          obs['time_end'][obs['date'] == int(date)].values[0]],
            'flag_airmass': [obs['am_start'][obs['date'] == int(date)].values[0],
                          obs['am_end'][obs['date'] == int(date)].values[0]]}

    files = glob.glob(paths['output']+date)
    if len(files) == 0:
        mkdircmd = 'mkdir '+paths['output']+date
        os.system(mkdircmd)

    return info



#----------------------------------------------------------------------#
# get_panstarrs_refs: get reference stars from PanSTARRS archive       #
#----------------------------------------------------------------------#


def get_panstarrs_refs(ra=None,dec=None,radius=10.0,maxMag=15,minMag=12,maxErr=0.02,maxStd=0.01,sort='r'):
    '''
    Description
    -----------
    - Input RA and Dec and get back the best reference stars closer than 10' from target
    - Output a csv file called target_info.csv that will be read and updated by later
      routines

    Inputs
    ------
    ra:    (string) Right ascension of the target in hh:mm:ss.ss
    dec:   (string) Declination of the target in dd:mm:ss.ss

    Output
    ------
    info dictionary

    Example
    -------
    '''

    data = panstarrs_query_sorted(ra=ra,dec=dec,verbose=False,radius=radius,maxMag=maxMag,
                                  minMag=minMag,maxErr=maxErr,maxStd=maxStd)
    if len(data) == 0:
        print(' ')
        print('No PanSTARRS sources found! Consider less stringent constraints.')
        print(' ')
        return None

    ra   = np.array([s[0].replace(' ',':') for s in data['raMean'].values])
    dec  = np.array([s[0].replace(' ',':') for s in data['decMean'].values])
    gmag = np.array([mag[0] for mag in data['gMeanApMag'].values])
    gmagerr = np.array([mag[0] for mag in data['gMeanApMagErr'].values])
    gmagstd = np.array([mag[0] for mag in data['gMeanApMagStd'].values])
    rmag = np.array([mag[0] for mag in data['rMeanApMag'].values])
    rmagerr = np.array([mag[0] for mag in data['rMeanApMagErr'].values])
    rmagstd = np.array([mag[0] for mag in data['rMeanApMagStd'].values])
    imag = np.array([mag[0] for mag in data['iMeanApMag'].values])
    imagerr = np.array([mag[0] for mag in data['iMeanApMagErr'].values])
    imagstd = np.array([mag[0] for mag in data['iMeanApMagStd'].values])
    zmag = np.array([mag[0] for mag in data['zMeanApMag'].values])
    zmagerr = np.array([mag[0] for mag in data['zMeanApMagErr'].values])
    zmagstd = np.array([mag[0] for mag in data['zMeanApMagStd'].values])

    radeg  = []
    decdeg = []
    for i in range(len(ra)):
        coords = SkyCoord(ra[i],dec[i],unit=(u.hour,u.degree))
        radeg.append(coords.ra.deg)
        decdeg.append(coords.dec.deg)

    Vmag = Sloan_to_Johnson(gmag,rmag,imag)

    # Sort the magnitudes in the specified band.
    if sort == 'g':
        key = gmag
    if sort == 'r':
        key = rmag
    if sort == 'i':
        key == imag
    if sort == 'z':
        key = zmag
    if sort == 'V':
        key = Vmag
        
    sort_inds = np.argsort(key)

    if Vmag is not None:
        print('WARNING: Vmag errors and standard deviations not calculated correctly!')
        Vmagerr = 0.02*np.ones(len(Vmag))
        Vmagstd = 0.02*np.ones(len(Vmag))
        if len(Vmag) == len(rmag):
            refs = {'RAdeg':np.array(radeg)[sort_inds], 'DECdeg':np.array(decdeg)[sort_inds],
                    'gmag':gmag[sort_inds], 'gmagerr':gmagerr[sort_inds], 'gmagstd':gmagstd[sort_inds],
                    'rmag':rmag[sort_inds], 'rmagerr':rmagerr[sort_inds], 'rmagstd':rmagstd[sort_inds],
                    'Vmag':Vmag[sort_inds], 'Vmagerr':Vmagerr[sort_inds], 'Vmagstd':Vmagstd[sort_inds],
                    'imag':imag[sort_inds], 'imagerr':imagerr[sort_inds], 'imagstd':imagstd[sort_inds],
                    'zmag':zmag[sort_inds], 'zmagerr':zmagerr[sort_inds], 'zmagstd':zmagstd[sort_inds]}
    else:
        print('V magnitudes not reliable!')
        refs = {'RAdeg':np.array(radeg)[sort_inds], 'DECdeg':np.array(decdeg)[sort_inds],
                'gmag':gmag[sort_inds], 'gmagerr':gmagerr[sort_inds],'gmagstd':gmagstd[sort_inds],
                'rmag':rmag[sort_inds], 'rmagerr':rmagerr[sort_inds],'rmagstd':rmagstd[sort_inds],
                'imag':imag[sort_inds], 'imagerr':imagerr[sort_inds],'imagstd':imagstd[sort_inds],
                'zmag':zmag[sort_inds], 'zmagerr':zmagerr[sort_inds],'zmagstd':zmagstd[sort_inds]}
    return refs                                                      


def Sloan_to_Johnson(g,r,i):
    '''
    Use Table 1 in Jester et al. 2005 to convert g and r into Johnson V
    Use conversion for All stars with R-I < 1.15 since it does not depend on u band
    https://arxiv.org/pdf/astro-ph/0506022.pdf
    RMS residual to fits for V band  = 0.02 mags
    '''
    # Check to make sure the transformation criterion is satisfied
    RmI =  1.00*(r - i) + 0.21
    inds, = np.where(RmI >=1.15)
    if len(inds) >= 1:
        print('Not all stars can be converted safely into Johnson V band!')
        print(inds)
        return None
    V = g - 0.59*(g - r) - 0.01
    
    return V

#----------------------------------------------------------------------
# Get image background
#----------------------------------------------------------------------
def get_image_background(image,sigma_clip=3.0,boxsz=50,filtersz=3):
    '''
    Get background for an input image
    '''

    from astropy.stats import SigmaClip
    from photutils import Background2D, MedianBackground

    sc= SigmaClip(sigma=sigma_clip)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (boxsz, boxsz), filter_size=(filtersz,filtersz),
                       sigma_clip=sc, bkg_estimator=bkg_estimator)

    return bkg.background

#----------------------------------------------------------------------#
# night_stack: stack images from a night in a given band               #
#----------------------------------------------------------------------#

def night_stack(date,obstype=None,targname=None,band=None):

    '''
    Description
    -----------
    Stack all the images from a night of observing so that max aperture size and sky radii
    can be checked.

    Inputs
    ------
    date:     (string) YYYYMMDD
    targname: (string) Target name
    band:     (string) Band name (g, r, i, z, V, etc)

    Output
    ------
    stacked image and header

    Example
    -------
    im,h = night_stack('20180621','AT2018cow',band='g')

    '''

    paths = get_paths(obstype=obstype,targname=targname)

    files = glob.glob(paths['output']+date)
    if len(files) == 0:
        mkdircmd = 'mkdir '+paths['output']+date
        os.system(mkdircmd)

    cals = get_cal_frames(date,band=band,targname=targname,obstype=obstype)
    if cals['bias'] is None:
        print('No calibration frames found!!!')
        return

    # Get all files in specified band on specified date
    sfiles,sct = get_files(date=date,prefix=targname,tag=band,suffix='solved.fits')

    # Choose a reference file
    ref = sfiles[sct/2]
    imref,href = read_image(ref,plot=False)

    # Get x and y dimensions of the image
    xsz,ysz = np.shape(imref)

    # Create an empty stack of images
    stack = np.zeros((xsz,ysz,sct))

    # Stack images
    for i in trange(sct,desc = 'Stacking images', unit = ' image'):
        # Read image in stack
        im,header = read_image(sfiles[i])

        image = calibrate_image(im,header,cals,rotated_flat=True)

        # Align image with astrometry from the refrerence image
        newim = hcongrid(image,header,href)

        # Put it in the stack
        stack[:,:,i] = newim

    # ...OR do a median filter
    final_med = np.nanmedian(stack,axis=2)
    inds = ~np.isfinite(final_med)
    final_med[inds] = 0

    return final_med,href



#----------------------------------------------------------------------#
# make_targ_info: make target information dictionary                   #
#----------------------------------------------------------------------#


def make_targ_info(refs,ra=None,dec=None):
    '''
    Take refs dictionary and make the target the first entry. Call this dictionary
    targ_info
    '''

    coords = SkyCoord(ra,dec,unit=(u.hour,u.degree))

    ras  = np.append(coords.ra.deg,refs['RAdeg'])
    decs = np.append(coords.dec.deg,refs['DECdeg'])

    targ_info = refs.copy()
    targ_info['RAdeg'] = ras
    targ_info['DECdeg'] = decs
    for tag in ['gmag','rmag','imag','zmag','Vmag',
                'gmagerr','rmagerr','imagerr','zmagerr','Vmagerr']:
        targ_info[tag] = np.append(np.nan,refs[tag])

    return targ_info


#----------------------------------------------------------------------#
# choose_refs:                                                         #
#    routine to select the reference stars to be used
#----------------------------------------------------------------------#

def choose_refs(fname,target_ra,target_dec,bias=None,dark=None,flat=None,origin='lower',
                figsize=8,outdir='./',outfile='coordinates.txt',clobber=False):

    '''
    target_ra in hours
    target_dec in degrees
    '''

    # Don't redo choose_refs unless clobber keyword set
    if len(glob.glob(outdir+outfile)) == 1 and not clobber:
        print("Reference position file: "+outfile+" already exists!")
        print("... reading saved positions")
        coords = np.loadtxt(outdir+outfile)
        ras = coords[:,0]
        decs = coords[:,1]
        return ras,decs

    # Convert ra and dec strings to decimal degree floats
    coords = SkyCoord(target_ra,target_dec,unit=(u.hour,u.deg))
    ra0  = coords.ra.deg
    dec0 = coords.dec.deg

    # Read image
    image,header = fits.getdata(fname, 0, header=True)
    image = image.astype('float32')
    ysz,xsz = image.shape
    if length(bias) > 1:
        image -= bias
    if length(dark) > 1:
        image -= (dark*header['EXPTIME'])
    if length(flat) > 1:
        image /= flat

# Read header
    hdulist = fits.open(fname)

# Convert RA and Dec to pixel coordinates
    w = wcs.WCS(hdulist[0].header)
    world0 = np.array([[ra0, dec0]])
    pix0 = w.wcs_world2pix(world0,1) # Pixel coordinates of (RA, DEC)
    x0 = pix0[0,0]
    y0 = pix0[0,1]

    if (x0 > xsz) |(x0 < 0) | (y0 > ysz) | (y0 < 0):
        print(x0,y0)
        print("Target star position is out of range!")
        return None, None

# Get image information
    sig = mad(image.flatten())
    #sig = rb.std(image)
    med = np.median(image)
    vmin = med - 5*sig
    vmax = med + 15*sig

# Plot image
    plt.ion()
    fig = plt.figure(99,figsize=(figsize,figsize))
    ax = fig.add_subplot(111)
    ax.imshow(image,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest',origin=origin)
    ax.scatter(x0,y0,marker='o',s=100,facecolor='none',edgecolor='green',linewidth=1.5)
    ax.scatter(x0,y0,marker='+',s=100,facecolor='none',edgecolor='green',linewidth=1.5)
    plt.draw()

# Click image and receive x and y values
    refx=[]
    refy=[]
    def onclick(event):
        newx = event.xdata
        newy = event.ydata
        if newx < xsz and newy < ysz and newx > 0 and newy > 0:
            refx.append(newx)
            refy.append(newy)
            print("------------------------------")
            print("refx = " , newx)
            print("refy = " , newy)
            ax.scatter(refx,refy,marker='o',s=100,facecolor='none',edgecolor='yellow',linewidth=1.5)
            plt.draw()

# Engage "onclick"
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Stall here such that canvas can disconnect before further calculations
    print("Click positions of reference stars")
    print(raw_input("Press return when finished selecting sources \n"))

# Disengage "onclick"
    fig.canvas.mpl_disconnect(cid)

# Convert xs and ys to RAs and Decs
    raval,decval = w.wcs_pix2world(refx,refy,1)


# Put the target star at the beginning of the lists
    ras = np.append(ra0,raval)
    decs = np.append(dec0,decval)

# Write out coordinate file
    coords = np.zeros((len(ras),2))
    coords[:,0] = ras
    coords[:,1] = decs
    np.savetxt(outdir+outfile,coords)

    return ras,decs


#----------------------------------------------------------------------#
# display_targets: visualize targets                                   #
#----------------------------------------------------------------------#


def display_targets(image,header,targ_info,targname=None,obstype=None,write=False,outfile=None,fignum=1,
                    siglo=5,sighi=50):

    # Get paths first...
    paths= get_paths(obstype=obstype,targname=targname)

    # Display image
    display_image(image,fignum=fignum,figsize=(8,6),siglo=siglo,sighi=sighi)

    # Get the World Coordinate System information from the header
    w = wcs.WCS(header)

    # Append reference coordinates
    ras  = targ_info['RAdeg']
    decs = targ_info['DECdeg']

    N = len(ras)
    cmap = plt.get_cmap('rainbow',N)

    # Plot up the locations of the chosen stars
    for i in range(N):
        c = SkyCoord(ras[i],decs[i], unit=(u.deg, u.deg))
        x, y = wcs.utils.skycoord_to_pixel(c, w)
        try:
            bkg_aperture = CircularAnnulus((x,y),r_in=targ_info['skyrad'][i][0],
                                           r_out=targ_info['skyrad'][i][1])
            aperture = CircularAperture((x,y), r=targ_info['ap_max'][i])
            do_aperture = True
        except:
            do_aperture = False

        if i == 0:
            label = targname
        else:
            label = 'Ref '+str(i)
        if do_aperture:
            color= cmap(float(i)/float(N))
            aperture.plot(color=color,label=label)
            bkg_aperture.plot(color=color, fill=False)
        else:
            plt.plot(x,y,'o',ms=15,markerfacecolor='none',
                     markeredgewidth=2,label=label)
    plt.legend(loc=2,bbox_to_anchor=(1.01,1.01))
    if outfile is None:
        outfile = targname+'_stack_refs.png'
    if write:
        if paths['output'] is None:
            plt.savefig('./'+outfile,dpi=300)
        else:
            plt.savefig(paths['output']+outfile,dpi=300)

    return


#----------------------------------------------------------------------
# check_ast: is there astrometry in the header?
#----------------------------------------------------------------------
def check_ast(fname):
    header = fits.getheader(fname)

    if 'CD1_1' in header and 'CD1_2' in header and 'CD2_1' in header \
       and 'CD2_2' in header:
        status = 1
    else:
        status = 0
    return status

#----------------------------------------------------------------------#
# check_skyrad:                                                        #
#----------------------------------------------------------------------#

def check_skyrad(file=None,image=None,header=None,ra=None,dec=None,bias=None,dark=None,flat=None,
                 skyrad=[20,25],aperture=None,figsize=(8,8),siglo=3.0,sighi=5.0,recenter=True):

    if file is not None:
        status = check_ast(file)
        if status == 0:
            print('No astrometry in header')
            return

        image, header = fits.getdata(file, 0, header=True)
        image = np.float32(image)
    elif image is not None:
        image = np.float32(image)
    else:
        print('File or image must be supplied')
        return

    if np.shape(bias) == np.shape(image):
        image -= bias
    if np.shape(dark) == np.shape(image):
        image -= (dark*header['EXPTIME'])
    if np.shape(flat) == np.shape(image):
        image /= flat

    op_dict = optimal_aperture(image,header,ra=ra,dec=dec,skyrad=skyrad,plot=False,recenter=recenter)
    x = op_dict['xcen']
    y = op_dict['ycen']
    if not aperture:
        aperture = op_dict['optimal_aperture']

    sz = int(round(max(200,np.max(skyrad)*2.25)))

    # Check this code !!!
    if sz % 2 == 0:
        sz += 1

    yround = int(round(y))
    xround = int(round(x))
    patch = image[yround-sz/2:yround+sz/2+1,xround-sz/2:xround+sz/2+1]

    position = (x-np.round(x)+sz/2,y-np.round(y)+sz/2)

    bkg_aperture = CircularAnnulus(position, r_in=skyrad[0], r_out=skyrad[1])
    aperture = CircularAperture(position, r=aperture)

    sig = mad(patch.flatten())
    #sig = rb.std(patch)
    med = np.median(patch)
    vmin = med - siglo*sig
    vmax = med + sighi*sig
    plt.ion()
    plt.figure(20,figsize=figsize)
    plt.clf()
    plt.imshow(patch,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest',origin='lower')
    plt.scatter(sz/2,sz/2,marker='+',s=200,color='yellow',linewidth=1.5,label='Original position')
    plt.title('zoom in of target')
    aperture.plot(color='cyan')
    bkg_aperture.plot(color='cyan', hatch='//', alpha=0.8)

    # Need to see if star is saturated!!
    # This may be broken
    #xpos = np.arange(-(sz/2),(sz/2)+1,1)
    #ypos = np.arange(-(sz/2),(sz/2)+1,1)
    xpos = np.arange(-(20),(20)+1,1)
    ypos = np.arange(-(20),(20)+1,1)
    X,Y = np.meshgrid(xpos,ypos)
    fig = plt.figure(21,figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.set_title('Point Spread Function')
    zoom = patch[sz/2-20:sz/2+21,sz/2-20:sz/2+21]
    surf = ax.plot_surface(X, Y, zoom, rcount=100, ccount=100,
                           cmap=cm.gist_earth_r)
#    ax.set_xlim(-20,20)
#    ax.set_ylim(-20,20)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    return



def check_apertures(image,header,targ_info,index=0,aperture=12,skyrad=[25,35],recenter=True):
    '''
    - Input stacked image, header, and target info
    - Loop through each target and check aperture settings
    - When you are happy with the aperture it will update the target_info dictionary
    '''

    # Get the World Coordinate System information from the header
    w = wcs.WCS(header)

    # Create new keys for targ_info dictionary if not already there
    if 'ap_max' not in targ_info:
        length = len(targ_info['RAdeg'])
        nans = np.ones(length)*np.nan
        targ_info['ap_max'] = nans
        # needs to be a list of lists
        targ_info['skyrad'] = np.array([[np.nan,np.nan] for i in range(length)])

    ras = targ_info['RAdeg']
    decs = targ_info['DECdeg']

    if index > len(ras):
        print('Target index out of range!')
        return targ_info
    else:
        radeg = ras[index]
        decdeg = decs[index]

    check_skyrad(image=image,header=header,ra=radeg,dec=decdeg,aperture=aperture,skyrad=skyrad,
                 recenter=recenter)

    plt.show()
    keep = raw_input('Are the aperture and sky radii adequate? (y/n) ')
    if keep == 'y':
        targ_info['ap_max'][index] = aperture
        targ_info['skyrad'][index] = skyrad
    else:
        eject = raw_input('Do you want to delete this target from the list? (y/n) ')
        if eject == 'y':
            for key in targ_info.keys():
                targ_info[key] = np.delete(targ_info[key],index,0)
    return targ_info


def write_targ_info(targ_info,obstype=None,targname=None,clobber=False):
    '''
    Write out pickle file once all values of targ_info have been defined and checked
    '''
    paths= get_paths(targname=targname,obstype=obstype)
    fname = paths['output']+targname+'_target_info.pck'
    test = glob.glob(fname)
    if len(test) == 1 and not clobber:
        print('Pickle file already exists!')
        return
    else:
        outfile = open(fname,"w")
        pickle.dump(targ_info,outfile)

    return


def read_targ_info(obstype=None,targname=None):
    '''
    Description
    -----------
    Return dictionary of information about all the targets in the field
    of Tabby's star

    Inputs
    ------

    Outputs
    -------
    Dictionary with target information. Dictionary keys are as follows:
    RAdeg = (float) RA in degrees
    DECdeg = (float) DEC in degrees
    skyrad = list of list pairs. each list pair signifies the inner and outer
             sky radius
    ap_max = (float) the maximum aperture allowable as to not overlap with
             background sources
    Vmag   = (float) Johnson V band magnitudes
    gmag   = (float) Sloan 2nd gen. g magnitudes
    rmag   = (float) Sloan 2nd gen. r magnitudes
    imag   = (float) Sloan 2nd gen. i magnitudes
    zmag   = (float) Sloan 2nd gen. z magnitudes

    Example
    -------
    targ_info = read_targ_info()

    '''


    paths = get_paths(obstype=obstype,targname=targname)
    fname = paths['output']+targname+'_target_info.pck'
    test = glob.glob(fname)
    if len(test) == 1:
        dict = pickle.load( open( fname, "rb" ) )
        return dict
    else:
        print('No target info file found for '+targname)
        return None


#----------------------------------------------------------------------#
# all_stack: stack images subject to given constraints
#----------------------------------------------------------------------#

def all_stack(files,ra=None,dec=None,skyrad=None,apmax=None,fwhm_max=4.0,
              outname='stack',outpath='./',dpix_max=50,dtheta_max=5,write=True,
              calwrite=False,calarchive=False,obstype=None,targname=None,
              clobber=False,skymax=2000.0):

    '''
    Description
    -----------
    Stack all the images from a list of files

    Inputs
    ------
    files:     (list) File names
    fwhm_max:  (float) Maximum FWHM to be added to stack (arcseconds)
    outpath:   (string) Path where output files will be written

    Output
    ------
    stacked image and header

    Example
    -------
    files,fct = get_files('SNS_NGC5055')
    im,h = all_stack(files,fwhm=5.0

    '''

    # See if output files exist, if so either load them or delete them
    testfile = glob.glob(outpath+outname+'.fits')
    if len(testfile) != 0:
        if not clobber:
            stack,stackh = read_image(outpath+outname+'.fits')
            return stack,stackh
        else:
            rmcmd = 'rm '+outpath+outname+'.fits'
            os.system(rmcmd)

    testfile = glob.glob(outpath+outname+'.png')
    if len(testfile) != 0:
        rmcmd = 'rm '+outpath+outname+'.png'
        os.system(rmcmd)

    # Check for which header to use as reference
    xpos = [] ; ypos = []
    pa = []
    pbar = tqdm(desc = 'Checking image headers', total = len(files), unit = 'files')
    for fname in files:
        try:
            header = fits.getheader(fname)
            if ra is None and dec is None:
                rastr  = header['OBJCTRA']
                decstr = header['OBJCTDEC']
                coords = SkyCoord(rastr,decstr,unit=(u.hour,u.deg))
                ra  = coords.ra.deg
                dec = coords.dec.deg
            x,y = radec_to_xy(ra,dec,header)
            xpos = np.append(xpos,x)
            ypos = np.append(ypos,y)
            pa = np.append(pa,posang(header))
        except:
            print('Cannot retrieve astrometry info from image header...')
        pbar.update(1)
    pbar.close()

    # Keep on the files whose distances are less than dpix_max from median position of target
    xmean = np.median(xpos)
    ymean = np.median(ypos)
    dists = np.sqrt((xpos-xmean)**2+(ypos-ymean)**2)
    dargs, = np.where(dists < dpix_max)
    print('Vetting '+str(len(dists)-len(dargs))+' targets based on distance requirement')
    refhead  = fits.getheader(files[np.argmin(dists)])
    patest = np.array([min((p-180)%180,180-p%180) for p in pa])
    pargs, = np.where(patest < dtheta_max)
    print('Vetting '+str(len(patest)-len(pargs))+' targets based on distance requirement')
    gargs = np.intersect1d(pargs,dargs)
    keepers = [files[arg] for arg in gargs]

    if len(keepers) == 0:
        print('Criteria too stringent! No files left')
        return None,None

    stack = None
    pbar = tqdm(desc = 'Stacking images', total = len(keepers), unit = 'files')
    for fname in keepers:
        temp,temph = read_image(fname)
        date = fname.split('/')[-2]
        band = temph['filter']
        cals = get_cal_frames(date,band=band,write=calwrite,archive=calarchive,
                              obstype=obstype,targname=targname)
        if cals['bias'] is None:
            print('No calibration frames found for '+fname.split('/')[-1]+'!!!')
        else:
            cal = calibrate_image(temp,temph,cals,rotated_flat=False)
            iminfo = do_sextractor(cal,temph,detect_thresh=5.0,analysis_thresh=5.0,
                                   minpix=3,outdir=outpath)
            fwhm = np.median(iminfo['FWHM_WORLD']*3600.0)
            skylev = np.median(iminfo['BACKGROUND'])

            if fwhm < fwhm_max and skylev < skymax:
                w = wcs.WCS(temph)
                # Align image with astrometry from the refrerence image
                xsz,ysz = np.shape(cal)
                newim = np.reshape(hcongrid(cal,temph,refhead),(xsz,ysz,1))
                if stack is None:
                    stack = np.copy(newim)
                else:
                    stack = np.append(stack,newim,axis=2)
            else:
                if fwhm > fwhm_max:
                    print('Rejecting image based on FWHM constraint: %.2f arcsec'%fwhm)
                if skylev > skymax:
                    print('Rejecting image based on sky level constraint: %.0f'%skylev)

        pbar.update(1)
    pbar.close()

    try:
        xsz,ysz,zsz = np.shape(stack)
    except:
        return None,None
    if zsz == 1:
        final_med = stack[:,:,0]
    if zsz == 0:
        return None,None
    if zsz > 1:
        # Median filter
        final_med = np.nanmedian(stack,axis=2)
    #inds = ~np.isfinite(final_med)
    #final_med[inds] = 0

    if write:
        print('Writing out '+outpath+outname+'.fits')
        fits.writeto(outpath+outname+'.fits', np.float32(final_med), refhead)
        plt.ion()
        display_image(final_med,siglo=3,sighi=30,fignum=99)
        plt.figure(99)
        plt.xticks([])
        plt.yticks([])
        axvals1 = plt.axis('off')
        plt.savefig(outpath+outname+'.png',dpi=300)

    return final_med,refhead



######################################################################
# Routines for the calibration of images
######################################################################
######################################################################

#----------------------------------------------------------------------#
# do_astrometry:                                                       #
#----------------------------------------------------------------------#

def do_astrometry(files,clobber=False,pixlo=0.1,pixhi=1.5,ra=None,dec=None,object=None,field=0.5,
                  numstars=100,downsample=4,keep_orig=False):

    """
    Overview:
    ---------
    Input list of FITS files and return solved FITS files to same
    directory with suffix "_solved.fits"

    Requirements:
    -------------
    Astrometry.net routines with calibration tiles installed locally
    will need to manually change astrometrydotnet_dir in routine if
    not installed in default location

    Calling sequence:
    -----------------
    do_astrometry(files,clobber=True)

    """

    # Get host... must be on bellerophon
    try:
        host = os.environ['HOST']
    except:
        host = socket.gethostname()

    if host != 'bellerophon':
        print('You must be on bellerophon for this to work!')
        return

    # Default directory for astrometry.net
    astrometrydotnet_dir = "/usr/local/astrometry"
    if not os.path.isdir(astrometrydotnet_dir):
        print(astrometrydotnet_dir+' path does not exist')
        return

    for targfile in files:
        # Allow for graceful opt out...
        # or not...
        '''
        print("Press any key to quit, continuing in 1 second...")
        timeout=1
        rlist, wlist, xlist = select([sys.stdin], [], [], timeout)
        if rlist:
            break
        '''
        # Get image and header
        print('reading '+targfile)
        data, header = fits.getdata(targfile, 0, header=True)
        if 'CD1_1' in header and 'CD1_2' in header and 'CD2_1' in header \
           and 'CD2_2' in header and not clobber:
            if targfile[-12:] != '_solved.fits':
                prepend = ['.'.join(targfile.split('.')[0:-1])]
                prepend.append('_solved.fits')
                outfile = ''.join(prepend)
                outfile = outfile.replace(' ','\ ').replace('(','\(').replace(')','\)')
                targfile = targfile.replace(' ','\ ').replace('(','\(').replace(')','\)')
                print('Moving solved file '+targfile+' ...')
                mvcmd = "mv "+targfile+" "+outfile
                os.system(mvcmd)

        else:
            # Get telescope RA and Dec as starting point

            # Test if RA and Dec is in header
            guess = False

            if 'OBJCTRA' in header and 'OBJCTDEC' in header:
                rastr  = header['OBJCTRA']
                decstr = header['OBJCTDEC']
                coords = SkyCoord(rastr,decstr,unit=(u.hour,u.deg))
                RAdeg  = coords.ra.deg
                DECdeg = coords.dec.deg
                guess = True

            if object != None:
                targ = SkyCoord.from_name(object)
                RAdeg = targ.ra.degree
                DECdeg = targ.dec.degree
                guess = True

            if ra != None and dec != None:
                RAdeg = ra
                DECdeg = dec
                guess = True

            # Do some string handling
            fname = targfile.split('/')[-1]
            outdir = targfile.split(fname)[0]+'astrometry'
            datadir = targfile.split(fname)[0]
            fname_sp = fname.split('.')
            if fname_sp[-2][-7:] == '_solved':
                stag = '.fits'
            else:
                stag = '_solved.fits'
            if len(fname_sp) > 2:
                ffinal = '.'.join(fname_sp[:-1])+stag
            elif len(fname_sp) == 2:
                ffinal = fname.split('.')[0]+stag
            else:
                print('File name '+fname+' has unconventional format')
                return None

            if fname == ffinal:
                rmfile = False
                print('Final file will have the same name as the original!')
            else:
                rmfile = True
            # Don't redo astrometry unless clobber keyword set
            if len(glob.glob(datadir+ffinal)) == 1 and not clobber:
                print("Astrometry solution for "+fname+" already exists!")
                print("Skipping...")
            else:

                # Construct the command string
                if guess:
                    command=string.join(
                        [astrometrydotnet_dir+"/bin/solve-field",
                         targfile.rstrip().replace(' ','\ ').replace('(','\(').replace(')','\)'),
                         "--scale-units arcsecperpix --scale-low "+str(pixlo)+" --scale-high "+str(pixhi),
                         "--ra ",str(RAdeg)," --dec ", str(DECdeg),
                         "--radius "+str(field),
                         "--downsample "+str(downsample),
                         "--no-plots",#" --no-fits2fits ",
                         "--skip-solved",
                         "--objs "+str(numstars),
                         "--odds-to-tune-up 1e4",
                         "--no-tweak",
                         "--dir",outdir.replace(' ','\ ').replace('(','\(').replace(')','\)'),"--overwrite"])
                else:
                    command=string.join(
                        [astrometrydotnet_dir+"/bin/solve-field",
                         targfile.rstrip().replace(' ','\ ').replace('(','\(').replace(')','\)'),
                         "--scale-units arcsecperpix --scale-low "+str(pixlo)+" --scale-high "+str(pixhi),
                         "--radius "+str(field),
                         "--downsample "+str(downsample),
                         "--no-plots",#" --no-fits2fits ",
                         "--skip-solved",
                         "--objs "+str(numstars),
                         "--odds-to-tune-up 1e4",
                         "--no-tweak",
                         "--dir",outdir.replace(' ','\ ').replace('(','\(').replace(')','\)'),"--overwrite"])


                rmcmd = "rm -rf "+outdir.replace(' ','\ ').replace('(','\(').replace(')','\)')
                os.system(rmcmd)
                mkdircmd = 'mkdir '+outdir.replace(' ','\ ').replace('(','\(').replace(')','\)')
                os.system(mkdircmd)

                # Execute solve-field
                os.system(command)

                if len(fname_sp) > 2:
                    outname = '.'.join(fname_sp[:-1])+'.new'
                elif len(fname_sp) == 2:
                    outname = fname.split('.')[0]+'.new'

                mvcmd = "mv "+outdir.replace(' ','\ ').replace('(','\(').replace(')','\)')+"/"+outname+" "+ \
                        datadir.replace(' ','\ ').replace('(','\(').replace(')','\)')+ffinal

                os.system(mvcmd)

                rmcmd = "rm -rf "+outdir.replace(' ','\ ').replace('(','\(').replace(')','\)')
                rmcmd = "rm -rf "+outdir.replace(' ','\ ').replace('(','\(').replace(')','\)')
                os.system(rmcmd)

                if not keep_orig:
                    # checksum
                    try:
                        newim,newh = fits.getdata(datadir+ffinal, 0, header=True)
                        check = np.sum(data-newim)
                        if check == 0 and rmfile:
                            rmcmd = "rm -rf "+targfile.replace(' ','\ ').replace('(','\(').replace(')','\)')
                            os.system(rmcmd)
                        else:
                            print('Original image not removed!')
                    except:
                        pass
    return



def color_correction():
    # g_r_targ is preliminary g-r color of target.
    # g_r_ref is the g-r color of reference.
    # snr_t is the signal to noise ratio of the target star
    # snr_r is the signal to noise ratio of the reference star
    # ref_err is reference star error (error on brightness of star)
    # ref_std is reference star std (variance of star brightness over many measurements)

    Dg_r = g_r_targ - g_r_ref
    gout = -0.035*Dg_r + gin
    gout_err = 1.0

    return
    

#----------------------------------------------------------------------#
# get_cal_frames: get all calibration frames for a night of observing  #
#----------------------------------------------------------------------#

def get_cal_frames(date,setting=1,readnoise=False,band='V',targname=None,obstype=None,
                   write=True,archive=False,verbose=False):

    '''
    Description
    -----------
    Get the calibration frames for date supplied. Priority is for calibration frames
    from the given night of observing to be used. Else, master cal frames are used
    from the data archive

    Inputs
    ------
    date = (string) date of observation
    setting = (int) camera setting
    readnoise = (boolean) compute readnoise from bias frames?
    band = (string) photometric band

    Outputs
    -------
    dictionary of calibration frames. Keys are:
    bias = (float array) bias frame (counts)
    dark = (float array) dark frame (counts/sec)
    flat = (float array) flat field (relative sensitivity)
    NFF  = (float array) Near field flat (if available)
    FFF  = (float array) Far field flat (if available)

    Example
    -------
    cals = get_cal_frames('20171108',setting=1,readnoise=True,band='V')

    '''


    cals = {'bias':None, 'dark':None, 'flat':None, 'FFF':None, 'NFF':None, 'info':None}

    # Get paths
    paths = get_paths(obstype=obstype,targname=targname)

    if paths['output'] is None:
        paths['output'] = './'
    if archive:
        outdir=paths['archive']+date+'/'
    else:
        outdir = paths['output']+date+'/'
        if not os.path.isdir(outdir):
            mkdircmd = 'mkdir '+outdir
            os.system(mkdircmd)

    # Get calibration frames from the archive if they are there
    fname = paths['archive']+date+'/master_bias_'+date+'.fits'
    fname1 = paths['archive']+date+'/master_bias.fits'
    if len(glob.glob(fname)) == 1:
        bias, bh = read_image(fname)
        cals['bias'] = bias
    elif len(glob.glob(fname1)) == 1:
        bias, bh = read_image(fname1)
        cals['bias'] = bias
    else:
        # Make master bias from nightly calibrations, else use master in archive
        biasfiles,bct = get_files(date=date,prefix='Bias',tag='1X1',suffix='fts')
        if bct > 0:
            bias = master_bias(biasfiles,readnoise=readnoise,tag='_'+date,outdir=outdir,write=write)
            cals['bias'] = bias

        if bct == 0 and setting == 1:
            bias = None
            if targname is not None and obstype is not None:
                try:
                    bias,bh = fits.getdata(paths['output']+date+'/master_bias.fits', 0, header=True)
                    cals['bias'] = bias
                    if verbose:
                        print('Using master bias from '+paths['output']+date+'/')
                except:
                    pass
            if bias is None:
                try:
                    bias,bh = fits.getdata(paths['archive']+'calfiles/master_bias.fits', 0,
                                           header=True)
                    if verbose:
                        print('Using master biases')
                    cals['bias'] = bias
                except:
                    if verbose:
                        print('No bias frame!')

    # Get calibration frames from the archive if they are there
    fname = paths['archive']+date+'/master_dark_'+date+'.fits'
    fname1 = paths['archive']+date+'/master_dark.fits'
    if len(glob.glob(fname)) == 1:
        dark, dh = read_image(fname)
        cals['dark'] = dark
    elif len(glob.glob(fname1)) == 1:
        dark, dh = read_image(fname1)
        cals['dark'] = dark
    else:
        # Make master dark from nightly calibrations, else use master in archive
        darkfiles,dct = get_files(date=date,prefix='Dark',tag='1X1',suffix='fts')
        if dct > 0:
            if bias is None:
                print('')
                print('DATE: '+date)
                print('WARNING: creating dark frame with no bias!!!')
                pdb.set_trace()
            dark = master_dark(darkfiles,bias=bias,tag='_'+date,outdir=outdir,write=write)
            cals['dark'] = dark


        if dct == 0 and setting == 1:
            if targname is not None and obstype is not None:
                try:
                    dark,dh = fits.getdata(paths['output']+date+'/master_dark.fits', 0,
                                           header=True)
                    cals['dark'] = dark
                    if verbose:
                        print('Using master dark from '+paths['output']+date+'/')
                except:
                    pass
            if cals['dark'] is None:
                try:
                    dark,dh = fits.getdata(paths['archive']+'calfiles/master_dark.fits', 0,
                                           header=True)
                    if verbose:
                        print('Using master dark')
                    cals['dark'] = dark
                except:
                    if verbose:
                        print('No dark frame!')
    info = {}
    # Look for correct directory for master flats
    if int(date) <= 20171202:
        fdir = 'Pre-20171202'
        info['flipped'] = True
        cals['info'] = info
    if int(date) >= 20171203 and int(date) <= 20180211:
        fdir = '20171203-20180211'
        info['flipped'] = True
        cals['info'] = info
    if int(date) >= 20180212 and int(date) <= 20180421:
        fdir = '20180212-20180421'
        info['flipped'] = True
        cals['info'] = info
    if int(date) >= 20180422 and int(date) <= 20181001:
        fdir = '20180422-20181001'
        info['offset'] = 181.0
        info['xsz'] = 2048 ; info['ysz'] = 2048
        info['xcen'] = 1052 ; info['ycen'] = 958
        info['xshift'] = info['xcen'] - info['xsz']/2
        info['yshift'] = info['ycen'] - info['ysz']/2
        cals['info'] = info
        info['flipped'] = True
    if int(date) >= 20181002 and int(date) <= 20181220:
        fdir = '20181002-20181220'
        info['offset'] = 181.0
        info['xsz'] = 2048 ; info['ysz'] = 2048
        info['xcen'] = 1040 ; info['ycen'] = 977
        info['xshift'] = info['xcen'] - info['xsz']/2
        info['yshift'] = info['ycen'] - info['ysz']/2
        cals['info'] = info
        info['flipped'] = False
    if int(date) >= 20181221 and int(date) <= 20190930:
        fdir = '20181221-20190930'
        info['offset'] = 181.0
        info['xsz'] = 2048 ; info['ysz'] = 2048
        info['xcen'] = 1038 ; info['ycen'] = 975
        info['xshift'] = info['xcen'] - info['xsz']/2
        info['yshift'] = info['ycen'] - info['ysz']/2
        cals['info'] = info
        info['flipped'] = False
    if int(date) >= 20191001 and int(date) <= 20200302:
        fdir = '20191001-20200302'
        info['offset'] = 182.08
        info['xsz'] = 2048 ; info['ysz'] = 2048
        info['xcen'] = 1032; info['ycen'] = 966
        info['xshift'] = info['xcen'] - info['xsz']/2
        info['yshift'] = info['ycen'] - info['ysz']/2
        cals['info'] = info
        info['flipped'] = False
    if int(date) >= 20200303:
        fdir = '20200303'
        info['offset'] = 182.08
        info['xsz'] = 2048 ; info['ysz'] = 2048
        info['xcen'] = 1032; info['ycen'] = 966
        info['xshift'] = info['xcen'] - info['xsz']/2
        info['yshift'] = info['ycen'] - info['ysz']/2
        cals['info'] = info
        info['flipped'] = False

    flatfile = 'master_flat_'+band+'.fits'

    try:
        fff,fffh = fits.getdata(paths['archive']+'calfiles/'+fdir+'/'+'FFF_'+band+'.fits', 0, header=True)
        nff,nffh = fits.getdata(paths['archive']+'calfiles/'+fdir+'/'+'NFF_'+band+'.fits', 0, header=True)
        cals['FFF'] = fff
        cals['NFF'] = nff
    except:
        if verbose:
            print('No rotated flats available')
        else:
            pass

    try:
        flat,fh = fits.getdata(paths['archive']+'calfiles/'+fdir+'/'+flatfile, 0, header=True)
        cals['flat'] = flat
    except:
        try:
            cals['flat'] = nff
        except:
            if verbose:
                print('No flat frame!')

    return cals



#----------------------------------------------------------------------#
# master_bias: create a master bias frame from a list of files         #
#----------------------------------------------------------------------#

def master_bias(files,write=True,outdir='./',readnoise=False,clobber=False,verbose=True,
                float32=True,tag='',median=False):

    """
    Overview:
    ---------
    Create master bias frame from series of biases (median filter).
    Returns a master_bias frame and writes FITS file to disk in specified
    directory.

    Optionally, the read noise is calculated from the variance of each
    pixel in the bias stack. This is *very* slow. So only use this option
    if you really need to. The readnoise image is also written to disk.

    Inputs:
    -------
    files       : List of flat field files from which a master bias will be created.
                  Must be provided, no default.

    Keyword arguments:
    ------------------
    write       : Toggle to write files to disk (default True)
    outdir      : Directory to which output files are written (default pwd)
    clobber     : Toggle to overwrite files if they already exist in outdir
                  (default False)
    readnoise   : Do readnoise calculation (very slow! default False)
    verbose     : Print out progress (default True)

    Calling sequence:
    -----------------
    master_bias = master_bias(biasfiles,write=True,readnoise=False,
                              outdir='/home/users/bob/stuff/')

    """

# Don't redo master_bias unless clobber keyword set
    name  = outdir+'master_bias'+tag+'.fits'
    if len(glob.glob(name)) == 1 and not clobber:
        print("Master bias already exists!")
        master_bias = fits.getdata(name,0,header=False)
        return master_bias

# Get information from inputs and create stack array
    fct = len(files)
    image, header = fits.getdata(files[0], 0, header=True)
    ysz,xsz = image.shape
    stack = np.zeros((fct,ysz,xsz))
    temps = []

# Load stack array and get CCD temperatures
    for i in np.arange(fct):
        output = 'Reading {}: frame {} of {} \r'.format(files[i].split('/')[-1],\
                                                          str(i+1),str(fct))
        sys.stdout.write(output)
        sys.stdout.flush()
        image, header = fits.getdata(files[i], 0, header=True)
        temps.append(header["CCD-TEMP"])
        stack[i,:,:] = image

# Calculate read noise directly from bias frames if prompted
    if readnoise:
        rn = np.zeros((ysz,xsz))
        print("Starting readnoise calculation")
        pbar = tqdm(desc = 'Calculating readnoise', total = ysz, unit = 'rows')
        for i in np.arange(ysz):
            for j in np.arange(xsz):
                rn[i,j] = mad(stack[:,i,j].flatten())
                #rn[i,j] = np.std(stack[:,i,j].flatten(),ddof=1)
            pbar.update(1)

# Make a nice plot (after all that hard work)
        aspect = np.float(xsz)/np.float(ysz)
        plt.figure(39,figsize=(5*aspect*1.2,5))
        plt.clf()
        sig = mad(rn.flatten())
        med = np.median(rn)
        mean = np.mean(rn)
        vmin = med - 2*sig
        vmax = med + 2*sig
        plt.imshow(rn,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest',origin='lower')
        plt.colorbar()
        plt.annotate(r'$\bar{\sigma}$ = %.2f cts' % mean, [0.95,0.87],horizontalalignment='right',
                     xycoords='axes fraction',fontsize='large')
#                    path_effects=[PathEffects.SimpleLineShadow(linewidth=3,foreground="w")])
        plt.annotate(r'med($\sigma$) = %.2f cts' % med, [0.95,0.8],horizontalalignment='right',
                     xycoords='axes fraction',fontsize='large')
#                    path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
        plt.annotate(r'$\sigma_\sigma$ = %.2f cts' % sig,
                     [0.95,0.73],horizontalalignment='right',
                     xycoords='axes fraction',fontsize='large')
#                    path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
        plt.title("Read Noise")
        plt.xlabel("pixel number")
        plt.ylabel("pixel number")

        if write:
            plt.savefig(outdir+'readnoise'+tag+'.png',dpi=300)

# Calculate master bias frame by median filter
    print('Calculating median of stacked frames...')
    if median:
        master_bias = np.median(stack,axis=0)
    else:
        master_bias = np.mean(stack,axis=0)

    # Make a plot
    aspect = np.float(xsz)/np.float(ysz)
    plt.figure(38,figsize=(5*aspect*1.2,5))
    plt.clf()
    sig = mad(master_bias.flatten())
    med = np.median(master_bias)
    vmin = med - 2*sig
    vmax = med + 2*sig
    plt.imshow(master_bias,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest',origin='lower')
    plt.colorbar()
    plt.annotate('Bias Level = %.2f cts' % med, [0.95,0.87],horizontalalignment='right',
                 xycoords='axes fraction',fontsize='large',color='k')
    plt.annotate(r'$\sigma$ = %.2f cts' % sig, [0.95,0.8],horizontalalignment='right',
                 xycoords='axes fraction',fontsize='large')
    plt.annotate(r'$\langle T_{\rm CCD} \rangle$ = %.2f C' % np.median(temps),
                 [0.95,0.73],horizontalalignment='right',
                 xycoords='axes fraction',fontsize='large')
    plt.title("Master Bias")
    plt.xlabel("pixel number")
    plt.ylabel("pixel number")

# Write out bias, readnoise and plot
    if write:
        name  = outdir+'master_bias'+tag
        plt.savefig(name+'.png',dpi=300)

        hout = fits.Header()
        hout['CCDTEMP'] = (np.median(temps), "Median CCD temperature")
        hout["TEMPSIG"] = (np.std(temps), "CCD temperature RMS")
        hout["BIAS"] = (med, "Median bias level (cts)")
        hout["BIASSIG"] = (sig, "Bias RMS (cts)")
        if len(glob.glob(name+'.fits')) == 1:
            os.system('rm '+name+'.fits')
        if float32:
            fits.writeto(name+'.fits', np.float32(master_bias), hout)
        else:
            fits.writeto(name+'.fits', master_bias, hout)

        if readnoise:
            name  = outdir+'readnoise'+tag
            if len(glob.glob(name+'.fits')) == 1:
                os.system('rm '+name+'.fits')
            if float32:
                fits.writeto(name+'.fits', np.float32(rn), hout)
            else:
                fits.writeto(name+'.fits', rn, hout)

    return master_bias




#----------------------------------------------------------------------#
# master_dark:
#----------------------------------------------------------------------#

def master_dark(files,bias=None,write=True,outdir='./',clobber=False,float32=True,tag='',
                median=False):
    """
    Overview:
    ---------
    Create master dark frame from series of darks (median filter).
    Returns a master dark frame. If write is specified, a FITS file
    will be written to "outdir" (default is pwd).

    Inputs:
    -------
    files       : List of flat field files from which a master dark will be created.
                  Must be provided, no default.

    Keyword arguments:
    ------------------
    bias        : Master bias frame (default None)
    write       : Toggle to write files to disk (default True)
    outdir      : Directory to which output files are written (default pwd)
    clobber     : Toggle to overwrite files if they already exist in outdir
                  (default False)

    Calling sequence:
    -----------------
    master_dark = master_dark(darkfiles,bias=master_bias,write=True,
                              outdir='/home/users/bob/stuff/')

    """

# Don't redo master_dark unless clobber keyword set
    name  = outdir+'master_dark'+tag+'.fits'
    if len(glob.glob(name)) == 1 and not clobber:
        print("Master dark already exists!")
        master_dark = fits.getdata(name,0,header=False)
        return master_dark

 # Get information from inputs and create stack array
    fct = len(files)
    image, header = fits.getdata(files[0], 0, header=True)
    ysz,xsz = image.shape
    stack = np.zeros((fct,ysz,xsz))
    temps = []
    exps = []

# Load stack array and get CCD temperatures
    for i in np.arange(fct):
        output = 'Reading {}: frame {} of {} \r'.format(files[i].split('/')[-1],\
                                                          str(i+1),str(fct))
        sys.stdout.write(output)
        sys.stdout.flush()
        image, header = fits.getdata(files[i], 0, header=True)
        exp = header["EXPOSURE"]
        exps.append(exp)
        temps.append(header["CCD-TEMP"])
        if length(bias) == 1:
            image = np.float(image)/exp
        else:
            image = (image-bias)/exp
        stack[i,:,:] = image

# Obtain statistics for the master dark image header
    # Temperature
    tmax = np.max(temps)
    tmin = np.min(temps)
    tmean = np.mean(temps)
    tmed = np.median(temps)
    tsig = np.std(temps)
    # Exposure times
    expmax = np.max(exps)
    expmin = np.min(exps)
    print('')
    print("Minimum CCD Temp. %.2f C" % tmin)
    print("Maximum CCD Temp. %.2f C" % tmax)
    print("CCD Temp. rms: %.3f C" % tsig)
    print("CCD Temp. mean: %.2f C" % tmean)
    print("CCD Temp. median: %.2f C" % tmed)

# Create master dark by median filter or mean
    if median:
        master_dark = np.median(stack,axis=0)
    else:
        master_dark = np.mean(stack,axis=0)

# Make a plot
    sig = mad(master_dark.flatten())
    med = np.median(master_dark)
    vmin = med - 2*sig
    vmax = med + 2*sig
    aspect = np.float(xsz)/np.float(ysz)
    plt.figure(37,figsize=(5*aspect*1.2,5))
    plt.clf()
    plt.imshow(master_dark,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest',origin='lower')
    plt.colorbar()
    plt.annotate('Dark Current = %.2f cts/sec' % med, [0.72,0.8],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
#                 path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
    plt.annotate(r'$\sigma$ = %.2f cts/sec' % sig, [0.72,0.75],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
#                 path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
    plt.annotate(r'$\langle T_{\rm CCD} \rangle$ = %.2f C' % np.median(temps),
                 [0.72,0.7],horizontalalignment='right',
                 xycoords='figure fraction',fontsize='large')
#                 path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
    plt.title("Master Dark")
    plt.xlabel("pixel number")
    plt.ylabel("pixel number")

# Write out plot and master dark array
    if write:
        name = outdir+'master_dark'+tag
        plt.savefig(name+'.png',dpi=300)

        hout = fits.Header()
        hout["TEMPMAX"] = (tmax, "Maximum CCD temperature")
        hout["TEMPMIN"] = (tmin, "Minimum CCD temperature")
        hout["TEMPMED"] = (tmed, "Median CCD temperature")
        hout["TEMPMN"] = (tmean, "Mean CCD temperature")
        hout["TEMPSIG"] = (tsig, "CCD temperature RMS")
        hout["EXPMAX"] = (expmax,"Maximum exposure time")
        hout["EXPMIN"] = (expmin, "Minimum exposure time")
        hout["EXPTIME"] = (1.0, "Effective exposure time for master dark")
        hout["DARKCNT"] = (med, "Median dark current (cts/sec)")
        hout["DARKSIG"] = (sig, "Dark current RMS (cts/sec)")
        if len(glob.glob(name+'.fits')) == 1:
            os.system('rm '+name+'.fits')
        if float32:
            fits.writeto(name+'.fits', np.float32(master_dark), hout)
        else:
            fits.writeto(name+'.fits', master_dark, hout)

    return master_dark




#----------------------------------------------------------------------#
# master flat field                                                    #
#----------------------------------------------------------------------#

def master_flat(files,bias=None,dark=None,write=True,outdir='./',
                tag='',clobber=False,stretch=3,float32=True,median=False):

    """
    Overview:
    ---------
    Create a master flat using (optionally) a provided bias and dark frame. Output
    is written to "outdir" in FITS format.

    Inputs:
    -------
    files       : List of flat field files from which a master flat will be created.
                  Must be provided, no default.

    Keyword arguments:
    ------------------
    bias        : Master bias frame (default None)
    dark        : Master dark frame calibrated in ADU/sec (default None)
    write       : Toggle to write files to disk (default True)
    outdir      : Directory to which output files are written (default pwd)
    clobber     : Toggle to overwrite files if they already exist in outdir
                  (default False)
    stretch     : Multiple of the noise RMS to stretch image (default 3)


    Calling sequence:
    -----------------
    master_flat = master_flat(flatfiles,bias=master_bias,dark=master_dark,write=True,
                              outdir='/home/users/bob/stuff/')

    """

# Don't redo master_dark unless clobber keyword set
    name = outdir+'master_flat'+tag+'.fits'
    if len(glob.glob(name)) == 1 and not clobber:
        print("Master flat already exists!")
        master_flat = fits.getdata(name,0, header=False)
        return master_flat

 # Get information from inputs and create stack array
    fct = len(files)
    image, header = fits.getdata(files[0], 0, header=True)
    filter = header["filter"]

    ysz,xsz = image.shape
    stack = np.zeros((fct,ysz,xsz))

# Load stack array and get CCD temperatures
    meds = []
    for i in np.arange(fct):
        output = 'Reading {}: frame {} of {} \r'.format(files[i].split('/')[-1],\
                                                          str(i+1),str(fct))
        sys.stdout.write(output)
        sys.stdout.flush()
        image, header = fits.getdata(files[i], 0, header=True)
        image = np.float32(image)
        if header["filter"] != filter:
            sys.exit("Filters do not match!")
        if length(bias) > 1:
            image -= bias
        if length(dark) > 1:
            exptime = header['EXPTIME']
            image -= dark*exptime
        meds.append(np.median(image))
        stack[i,:,:] = image/np.median(image)

# Obtain statistics for the master dark image header
    med = np.median(meds)
    sig = np.std(meds)

# Create master flat by median filter
    if median:
        master_flat = np.median(stack,axis=0)
    else:
        master_flat = np.mean(stack,axis=0)

# Make a plot
    sig = mad(master_flat.flatten())
    #sig = rb.std(master_flat)
    med = np.median(master_flat)
    vmin = med - stretch*sig
    vmax = med + stretch*sig
    aspect = np.float(xsz)/np.float(ysz)
    plt.figure(40,figsize=(5*aspect*1.2,5))
    plt.clf()
    plt.imshow(master_flat,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest',origin='lower')
    plt.colorbar()
    plt.title("Master Flat")
    plt.xlabel("pixel number")
    plt.ylabel("pixel number")


# Write out plot and master flat array
    if write:
        plt.savefig(outdir+'master_flat'+tag+'.png',dpi=300)
        hout = fits.Header()
        hout["FILTER"] = (filter, "Filter used when taking image")
        hout["MEDCTS"] = (med, "Median counts in individual flat frames")
        hout["MEDSIG"] = (sig, "Median count RMS in individual flat frames")
        if length(bias) > 1:
            hout.add_comment("Bias subtracted")
        if length(dark) > 1:
            hout.add_comment("Dark subtracted")

        if len(glob.glob(outdir+'master_flat'+tag+'.fits')) == 1:
            os.system('rm '+outdir+'master_flat'+tag+'.fits')
        if float32:
            fits.writeto(outdir+'master_flat'+tag+'.fits', np.float32(master_flat), hout)
        else:
            fits.writeto(outdir+'master_flat'+tag+'.fits', master_flat, hout)

    return master_flat


#----------------------------------------------------------------------#
# calibrate_image:                                                     #
#----------------------------------------------------------------------#

def calibrate_image(image,header,cals,rotated_flat=True, skipprint = False,
                    saturation=30000.0,masklev = 0.98,domask=False, skipflip=False):
    '''
    flipped refers to whether or not the science image is flipped
    doflip: refers to whether or not the science image is flipped with respect
            to the cal frames.
    calflipped: refers to wheter or not the cal frame images are flipped
    '''

    calim = np.copy(image).astype('float32')

    # Make a mask for saturated pixels
    mask = calim < saturation

    # Check for image flip!
    if skipflip is True:
        doflip = False
    else:
        flipped = posang(header,flipcheck=True)
        calflipped = cals['info']['flipped']
        doflip = not (flipped == calflipped)

    if doflip:
        print('WARNING: Image appears to be flipped with respect to the cal frames!')
        print('WARNING: Will attempt to account for this')

    exptime = header['EXPTIME']
    # Apply bias subtraction if there is a bias...
    if cals['bias'] is not None:
        if doflip:
            calim -= np.fliplr(cals['bias'])
        else:
            calim -= cals['bias']
    if cals['dark'] is not None:
        if doflip:
            calim -= (np.fliplr(cals['dark'])*exptime)
        else:
            calim -= (cals['dark']*exptime)

    if rotated_flat and cals['FFF'] is not None:

        # First divide out the near field...
        if doflip:
            calim /= np.fliplr(cals['NFF'])
        else:
            calim /= cals['NFF']
        print('Applying rotated flat field...')

        # parameters of rotated flat
        xsz = cals['info']['xsz'] ; ysz = cals['info']['ysz']
        xcen = cals['info']['xcen'] ; ycen = cals['info']['ycen']
        xshift = cals['info']['xshift'] ; yshift = cals['info']['yshift']

        # need image position angle
        posangle = None
        try:
            posangle = header['PA']
        except:
            print(' - no PA keyword in header')
        if posangle is None:
            try:
                posangle = header['ROT_PA']
            except:
                print(' - no ROT_PA keyword in header')
        if posangle is None:
            print('WARNING: getting position angle from CD matrix')
            posangle = posang(header,verbose=False)

        # need hour angle and deccenter
        try:
            ha = Angle(header['HA'],unit=u.hour).hour
        except:
            ha = calculate_ha(header)
        deccenter = xy_to_radec(1024,1024,header)[1]
        offset = cals['info']['offset']
        mech_angle = mech_pos(ha=ha,dec=deccenter,posangle=posangle,offset=offset)

        if flipped:
            rotang = mech_angle
        else:
            rotang = -1*mech_angle

        # shift to boresight center, then rotate
        rotshift = scipy.ndimage.rotate(scipy.ndimage.shift(cals['FFF'],(yshift,xshift)),
                                        rotang,reshape=False)
        # then shift back
        fff = scipy.ndimage.shift(rotshift,(-yshift,-xshift))

        shape = np.shape(fff)
        fff_flat = fff.flatten()
        inds, = np.where(fff_flat <= masklev)
        fff_flat[inds] = np.nan
        fff = np.reshape(fff_flat,shape)

        calim /= fff
        calim[~np.isfinite(calim)] = 0

    else:
        if not skipprint:
            print("No Far Field Flat, using normal flat")
        if cals['flat'] is not None:
            #!!!
            # Need logic to check if flat should be flipped
            if doflip:
                calim /= np.fliplr(cals['flat'])
            else:
                calim /= cals['flat']
    if domask:
        return calim*mask
    else:
        return calim







######################################################################
# Photometry routines
######################################################################
######################################################################

#----------------------------------------------------------------------#
# optimal_aperture:                                                    #
#            calculate optimal aperture for a source at "x" and "y"    #
#            coords in "image" using defined skyradii = [in,out]       #
#----------------------------------------------------------------------#

def optimal_aperture(image,header,x=None,y=None,ra=None,dec=None,
                     apmax=None,aperture=None,skyrad=[20,25],plot=False,
                     recenter=True):
    '''
    Description
    -----------
    optimal_aperture takes an image, a header, and skyrad [in,out]
        either x and y in pixel values, or ra and dec in degrees
    returns dictionary at end of function


    Inputs
    ------


    Output
    ------


    Example
    -------

    '''

    # Get ra, dec, x and y
    if ra == None and dec == None:
        ra,dec = xy_to_radec(x,y,header)
    elif x == None and y == None:
        x,y = radec_to_xy(ra,dec,header)


    # Want midpoint time of the exposure
    jd = header["jd"] + (header["exptime"]/2.0)/(24.0*3600.0)


    # Get airmass
    try:
        secz1 = header['airmass']
    except:
        print('No airmass in header')
        #print('Airmass in header: %.4f'%secz1)

    #print(' - computing airmass from header info')
    obs.date = ephem.date(jd-2415020.0) # pyephem uses Dublin Julian Day (why!!??!?)
    star = ephem.FixedBody()
    star._ra = np.radians(ra)
    star._dec = np.radians(dec)
    star.compute(obs)
    secz2 = calculate_airmass(np.degrees(star.alt))
    #print('Calculated airmass: %.4f'%secz2)

    # Change this to secz2 after testing
    secz = secz2
    try:
        dair = secz2-secz1
        #print('Airmass difference (calc-header): %.4f'%dair)
    except:
        pass

    if recenter:
        try:
            center_info = center_target(image,x,y,plot=plot)
            aspect = center_info['aspect']
            fwhm = center_info['fwhm']
            peak = center_info['peak']
            fit = center_info['fit']
            level = center_info['level']
            xpeak = center_info['xpeak']
            ypeak = center_info['ypeak']
            chisq = center_info['chisq']
        except:
            print('...Gaussian2D fit failed')
            aspect = np.nan ; fwhm = np.nan ; norm = np.nan ; peak = np.nan
            fit = np.nan ; level = np.nan ; chisq = np.nan
            xpeak = x ; ypeak = y
    else:
        print('No recentering on target')
        aspect = np.nan ; fwhm = np.nan ; norm = np.nan ; peak = np.nan
        fit = np.nan ; level = np.nan ; chisq = np.nan
        xpeak = x ; ypeak = y


    # Create vector of apertures
    if not apmax:
        apmax = np.min(skyrad)-1
    ap = np.linspace(1,apmax,100)

    # Do aperture photometry
    phot = do_aperture_photometry(image,header,xpeak,ypeak,ap,skyrad,recenter=False)

    # Optimize based on signal to noise from counts and background RMS
    counts = phot['counts']
    counterr = phot['counterr']
    flux = phot['flux']
    fluxerr = phot['fluxerr']
    snr = counts/counterr
    snrmax = np.max(snr)
    maxi = np.argmax(counts)
    totcounts = counts[maxi]
    totflux = counts[maxi]/header['exptime']
    totcounterr = counterr[maxi]
    totfluxerr = counterr[maxi]/header['exptime']
    totcountap = ap[np.argmax(counts)]
    totfluxap = ap[np.argmax(counts)]/header['exptime']

    # Plot optimization
    cog = counts/totcounts
    if plot:
        plt.ion()
        plt.figure(3)
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(ap,cog)
        #plt.xlabel("aperture radius (pixels)")
        plt.ylabel("normalized counts")
        plt.annotate('Total counts = %.f' % np.max(counts), [0.86,0.57],horizontalalignment='right',
                     xycoords='figure fraction',fontsize='large')
        plt.ylim([0,1.1])
        plt.axhline(y=1.0,linestyle='--',color='green')
        plt.subplot(2,1,2)
        plt.plot(ap,snr)
        plt.xlabel("aperture radius (pixels)")
        plt.ylabel("SNR")

# Optimal aperture
    op_ap = ap[np.argmax(snr)]
    op_ap_flux = flux[np.argmax(snr)]
    op_ap_fluxerr = fluxerr[np.argmax(snr)]
    op_ap_counts = counts[np.argmax(snr)]
    op_ap_counterr = counterr[np.argmax(snr)]

    if plot:
        plt.axvline(x=op_ap,linestyle='--',color='red')
        plt.annotate('SNR maximum = %.f' % np.max(snr), [0.8,0.17],horizontalalignment='right',
                     xycoords='figure fraction',fontsize='large')
        plt.annotate('Optimal aperture = %.2f' % op_ap, [0.8,0.12],horizontalalignment='right',
                     xycoords='figure fraction',fontsize='large')
        if aperture != None:
            plt.axvline(x=aperture,linestyle='--',color='black')

        plt.draw()

    out = {'optimal_aperture':op_ap, 'optimal_flux':op_ap_flux,
            'optimal_fluxerr':op_ap_fluxerr, 'optimal_counts':op_ap_counts,
            'optimal_counterr': op_ap_counterr, 'xcen':phot['xcen'], 'ycen':phot['ycen'],
            'fwhm':fwhm, 'aspect':aspect,'snrmax':snrmax,'totflux':totflux,
            'totfluxerr':totfluxerr, 'totcounts':totcounts, 'totcounterr':totcounterr,
            'tot_aperture':totfluxap,'chisq':chisq,
            'curve_of_growth':[ap,cog],'secz':secz,'jd':jd, 'exptime':header['exptime']}

    return out



#----------------------------------------------------------------------#
# do_aperture_photometry: do photometry using standard circular apertures
#----------------------------------------------------------------------#

def do_aperture_photometry(image,header,xpos,ypos,aperture,skyrad,recenter=True,skymed=False):
    '''
    - Can either give multiple positions, or multiple aperture sizes, not both!
    - Can give customized sky radii for each position or a single set of sky radii
      for all sources
    - Clunky structure. Reorganize this code!!!

    '''

    # Checks input parameters
    if length(aperture) > 1 and length(xpos) > 1:
        print(' - cannot give multiple apertures for multiple sources!')
        return None
    if length(xpos) > 1:
        if length(ypos) != length(xpos):
            print(' - coordinates do not have same length!')
            return None
        if length(np.array(skyrad).flatten()) != 2*length(xpos) and length(np.array(skyrad).flatten()) != 2:
            print(' - can either specify one set of sky radii, or the same number as positions!')
            return None

    # Output dictionary
    outputs = {'counts':None, 'counterr':None,
               'flux':None, 'fluxerr':None,
               'xval':None, 'yval':None,
               'xcen':None, 'ycen':None,
               'skycounts':None,'skycounterr':None,
               'skyflux':None, 'skyfluxerr':None,
               'normal_pval':None}

    # Need exptime for flux
    try:
        exptime = header['exptime']
    except:
        print(' - no exposure time info! Using exptime = 1s.')
        exptime = 1.0

    # Start loop for multiple targets in the same image
    if length(xpos) > 1:
        # Create empty vectors
        counts = [] ; counterr = []
        flux = [] ; fluxerr = []
        xval = [] ; yval = []
        xcen = [] ; yval = []
        skycounts = [] ; skycounterr = []
        skyflux = [] ; skyfluxerr = []
        maxcounts = []; normal_pval = []
        skycountrms = [] ; skyfluxrms = []

        num = length(xpos)
        for i in range(num):
            # Get center postion for aperture photometry
            if recenter:
                try:
                    center_info = center_target(image,xpos[i],ypos[i],plot=plot)
                    xcen = np.append(xcen,center_info['xpeak'])
                    ycen = np.append(ycen,center_info['ypeak'])
                except:
                    print(' - recentering failed!')
                    xcen = np.append(xcen,xpos[i]) ; ycen = np.append(ycen,ypos[i])
            else:
                xcen = xpos ; ycen = ypos
            position = [(xcen[-1],ycen[-1])]

            # Get sky radii
            if length(np.array(skyrad).flatten()) == 2:
                r_in = skyrad[0] ; r_out = skyrad[1]
            else:
                r_in = skyrad[i][0] ; r_in = skyrad[i][1]

            try:
                # Create apertures
                bkg_apertures = CircularAnnulus(position, r_in=r_in, r_out=r_out)
                circ_ap = CircularAperture(position, r=aperture)
                # Background calculation
                bkg_mask = bkg_apertures.to_mask(method='center')[0]
                # Cutout of image
                bkg_cutout = bkg_mask.cutout(image)
                # Image multiplied by mask
                image_bkg_mask = bkg_mask.multiply(image)
                # Pixel weighting map
                fracs_bkg_mask = image_bkg_mask/bkg_cutout
                # Only consider annulus
                xinds,yinds = np.where(fracs_bkg_mask > 0.0)
                sky_cutout = bkg_cutout[xinds,yinds]

                # Sky values and rms
                if skymed:
                    skycounts = np.append(skycounts,np.median(sky_cutout))
                else:
                    skycounts = np.append(skycounts,np.mean(sigmaRejection(sky_cutout)))
                skycounterr = np.append(skycounterr,mad(sky_cutout.flatten())/np.sqrt(len(xinds)))
                skycountrms = np.append(skycountrms,mad(sky_cutout.flatten()))

                skyflux = np.append(skyflux,skycounts[-1]/exptime)
                skyfluxerr = np.append(skyfluxerr,skycounterr[-1]/exptime)
                skyfluxrms = np.append(skyfluxrms,skycountrms[-1]/exptime)

                # Aperture photometry
                mask = circ_ap.to_mask(method='exact')[0]
                cutout = mask.cutout(image)
                image_mask = mask.multiply(image)
                fracs_mask = image_mask/cutout
                pixerr = np.sqrt(skycounts[-1]*fracs_mask + (cutout-skycounts[-1])*fracs_mask)

                counts = np.append(counts,np.sum(mask.multiply(image-skycounts[-1])))
                counterr = np.append(counterr,np.sqrt(np.sum(pixerr**2)))

                flux = np.append(flux,counts[-1]/exptime)
                fluxerr = np.append(fluxerr,counterr[-1]/exptime)

                AD,p_pval = normal_ad(sky_cutout)
                normal_pval = np.append(normal_pval,p)

                maxcounts = np.append(maxcounts,np.max(cutout))
            except:
                skycounts = np.append(skycounts,np.nan)
                skycounterr = np.append(skycounterr,np.nan)
                skycountrms = np.append(skycountrms,np.nan)
                skyflux = np.append(skyflux,np.nan)
                skyfluxerr = np.append(skyfluxerr,np.nan)
                skyfluxrms = np.append(skyfluxrms,np.nan)
                counts = np.append(counts,np.nan)
                counterr = np.append(counterr,np.nan)
                flux = np.append(flux,np.nan)
                fluxerr = np.append(fluxerr,np.nan)
                normal_pval = np.append(normal_pval,np.nan)
                maxcounts = np.append(maxcounts,np.nan)

    elif length(aperture) > 1:
        num = length(aperture)
        counts = [] ; counterr = []
        flux = [] ; fluxerr = []
        xval = [] ; yval = []
        xcen = [] ; yval = []
        skycounts = [] ; skycounterr = []
        skyflux = [] ; skyfluxerr = []
        maxcounts = []; normal_pval = []
        skycountrms = [] ; skyfluxrms = []

        for i in range(num):
            if recenter:
                try:
                    center_info = center_target(image,xpos,ypos,plot=plot)
                    xcen = center_info['xpeak']
                    ycen = center_info['ypeak']
                except:
                    print(' - recentering failed!')
                    xcen = xpos ; ycen = ypos
            else:
                xcen = xpos ; ycen = ypos
            position = [(xcen,ycen)]
            r_in = skyrad[0] ; r_out = skyrad[1]

            # Make apertures
            bkg_apertures = CircularAnnulus(position, r_in=r_in, r_out=r_out)
            circ_ap = CircularAperture(position, r=aperture[i])
            try:
                # Background calculation
                bkg_mask = bkg_apertures.to_mask(method='center')[0]
                # Cutout of image
                bkg_cutout = bkg_mask.cutout(image)
                # Image multiplied by mask
                image_bkg_mask = bkg_mask.multiply(image)
                # Pixel weighting map
                fracs_bkg_mask = image_bkg_mask/bkg_cutout
                # Only consider annulus
                xinds,yinds = np.where(fracs_bkg_mask > 0.0)
                sky_cutout = bkg_cutout[xinds,yinds]

                # Sky values and rms
                if skymed:
                    skycounts = np.median(sky_cutout)
                else:
                    skycounts = np.mean(sigmaRejection(sky_cutout))

                skycounterr = mad(sky_cutout.flatten())/np.sqrt(len(xinds))
                skycountrms = mad(sky_cutout.flatten())
                skyflux = skycounts/exptime
                skyfluxerr = skycounterr/exptime
                skyfluxrms = skycountrms/exptime

                # Aperture
                mask = circ_ap.to_mask(method='exact')[0]
                cutout = mask.cutout(image)
                image_mask = mask.multiply(image)
                fracs_mask = image_mask/cutout
                pixerr = np.sqrt(skycounts*fracs_mask + (cutout-skycounts)*fracs_mask)

                counts = np.append(counts,np.sum(mask.multiply(image-skycounts)))
                counterr = np.append(counterr,np.sqrt(np.sum(pixerr**2)))
                flux = np.append(flux,counts[-1]/exptime)
                fluxerr = np.append(fluxerr,counterr[-1]/exptime)

                AD,p = normal_ad(sky_cutout)
                normal_pval = np.append(normal_pval,p)

                maxcounts = np.append(maxcounts,np.max(cutout))
            except:
                skycounts = np.append(skycounts,np.nan)
                skycounterr = np.append(skycounterr,np.nan)
                skycountrms = np.append(skycountrms,np.nan)
                skyflux = np.append(skyflux,np.nan)
                skyfluxerr = np.append(skyfluxerr,np.nan)
                skyfluxrms = np.append(skyfluxrms,np.nan)
                counts = np.append(counts,np.nan)
                counterr = np.append(counterr,np.nan)
                flux = np.append(flux,np.nan)
                fluxerr = np.append(fluxerr,np.nan)
                normal_pval = np.append(normal_pval,np.nan)
                maxcounts = np.append(maxcounts,np.nan)


    else:
        if recenter:
            try:
                center_info = center_target(image,xpos,ypos,plot=plot)
                xcen = center_info['xpeak']
                ycen = center_info['ypeak']
            except:
                print(' - recentering failed!')
                xcen = xpos ; ycen = ypos
        else:
            xcen = xpos ; ycen = ypos
        position = [(xcen,ycen)]
        r_in = skyrad[0] ; r_out = skyrad[1]

        # Make apertures
        bkg_apertures = CircularAnnulus(position, r_in=r_in, r_out=r_out)
        circ_ap = CircularAperture(position, r=aperture)

        try:
            # Background calculation
            bkg_mask = bkg_apertures.to_mask(method='center')[0]
            # Cutout of image
            bkg_cutout = bkg_mask.cutout(image)
            # Image multiplied by mask
            image_bkg_mask = bkg_mask.multiply(image)
            # Pixel weighting map
            fracs_bkg_mask = image_bkg_mask/bkg_cutout
            # Only consider annulus
            xinds,yinds = np.where(fracs_bkg_mask > 0.0)
            sky_cutout = bkg_cutout[xinds,yinds]

            # Sky values and rms
            if skymed:
                skycounts = np.median(sky_cutout)
            else:
                skycounts = np.mean(sigmaRejection(sky_cutout))
            skycounterr = mad(sky_cutout.flatten())/np.sqrt(len(xinds))
            skycountrms = mad(sky_cutout.flatten())
            skyflux = skycounts/exptime
            skyfluxerr = skycounterr/exptime
            skyfluxrms = skycountrms/exptime

            # Aperture
            mask = circ_ap.to_mask(method='exact')[0]
            cutout = mask.cutout(image)
            image_mask = mask.multiply(image)
            fracs_mask = image_mask/cutout
            pixerr = np.sqrt(skycounts*fracs_mask + (cutout-skycounts)*fracs_mask)

            counts = np.sum(mask.multiply(image-skycounts))
            counterr = np.sqrt(np.sum(pixerr**2))

            flux = counts/exptime
            fluxerr = counterr/exptime

            AD,normal_pval = normal_ad(sky_cutout)

            maxcounts = np.max(cutout)
        except:
            skycounts = np.nan
            skycounterr = np.nan
            skycountrms = np.nan
            skyflux = np.nan
            skyfluxerr = np.nan
            skyfluxrms = np.nan
            counts = np.nan
            counterr = np.nan
            flux = np.nan
            fluxerr = np.nan
            normal_pval = np.nan
            maxcounts = np.nan

    outputs['xval'] = xpos
    outputs['yval'] = ypos

    outputs['xcen'] = xcen
    outputs['ycen'] = ycen

    outputs['counts'] = counts
    outputs['flux'] = counts/exptime

    outputs['counterr'] = counterr
    outputs['fluxerr'] = counterr/exptime

    outputs['skycounts'] = skycounts
    outputs['skyflux'] = skycounts/exptime
    outputs['skycounterr'] = skycounterr
    outputs['skyfluxerr'] = skycounterr/exptime
    outputs['skycountrms'] = skycountrms
    outputs['skyfluxrms'] = skycountrms/exptime
    outputs['normal_pval'] = normal_pval
    outputs['maxcounts'] = maxcounts

    return outputs



def aperture_photometry(cal,header,targ_info=None,targname=None,fwhm_flag=8.0,
                        aspect_flag=0.7,north_flag=10.0,snr_flag=5.0,
                        airmass_flag=2.0,bkg_level=15.0,saturation=30000.0,
                        normal_pflag = 0.01,recenter=False):

    '''
    Description
    -----------
    Provide aperture photometry for all the targets specified in the targ_info dictionary
    on a calibrated image (header must be provided).

    Inputs
    ------
    cal        = (float) calibrated image (must be counts NOT counts/sec)
    header     = (string) header for calibrated image with astrometry
    targ_info  = (optional) dictionary of targets in field
    fwhm_flag  = (float) maximum FWHM in arcseconds without flag
    saturation = (float) max counts before saturation in CALIBRATED image

    Outputs
    -------


    Example
    -------

    '''
    # Get targ_info if not supplied
    if targ_info is None:
        targ_info = read_targ_info()

    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))

    # Get preliminary photometry info in image
    char = phot_info(cal,header,targname=targname,targ_info=targ_info)

    band = header['filter'].replace("'","")

    # Apertures are:
    # 0: mean FWHM of all targets
    # 1: mean of all optimal apertures
    # 2: optimal aperture of primary target
    # 3: maximum aperture
    apertures = np.array([np.nanmean(char['fwhm']),
                          np.nanmean(char['optimal_aperture']),
                          char['optimal_aperture'][0],
                          np.nanmin(targ_info['ap_max'])])
    apertures = apertures[np.isfinite(apertures)]

    final_phot = {}
    for i in range(len(names)):
        sflags = []
        position = [char['xcen'][i],char['ycen'][i]]
        if char['snr'][i] < snr_flag:
            sflags.append('snr')

        if char['pa'] < north_flag or char['pa'] > (360.0-north_flag):
            pass
        else:
            sflags.append('north')

        if char['aspect'][i] < aspect_flag:
            sflags.append('shape')

        if char['fwhm'][i] > fwhm_flag:
            sflags.append('fwhm')

        if char['airmass'][i] > airmass_flag:
            sflags.append('airmass')

        if header['CCD-TEMP'] < -50 or header['CCD-TEMP'] > -20:
            sflags.append('temperature')

        flux = []
        count = 0
        for aprad in apertures:
            phot = do_aperture_photometry(cal,header,position[0],position[1],aprad,
                                          targ_info['skyrad'][i],recenter=recenter)
            flux = np.append(flux,phot['flux'])
            if count == 0:
                if phot['skyflux'] > bkg_level:
                    sflags.append('bkg level')
                if phot['normal_pval'] < normal_pflag:
                    sflags.append('bkg dist')
                if phot['maxcounts'] > saturation:
                    sflags.append('saturation')
                normal_pval = phot['normal_pval']
                background = phot['skyflux']
                background_rms = phot['skyfluxrms']
            count += 1
        try:
            phot_dict = {'aperture':apertures,'flux':flux,'background':background,
                         'background_rms':background_rms,'fwhm':char['fwhm'][i],'snr':char['snr'][i],
                         'airmass':char['airmass'],'exptime':header['exptime'],'pa':char['pa'],
                         'flags':sflags, 'bjd':char['bjd'], 'filter':band,
                         'aspect':char['aspect'][i],'xyposition':position,
                         'normal_pval':normal_pval,'airmass':char['airmass'][i]}
        except:
            pdb.set_trace()

        final_phot[names[i]] = phot_dict

    final_phot['bjd'] = char['bjd']
    final_phot['airmass'] = char['airmass']
    final_phot['band'] = band

    return final_phot



def check_target(image,header,x=None,y=None,ra=None,dec=None,skyrad=None,apmax=None):

    pa = posang(header)
    flip = posang(header,flipcheck=True)

    try:
        pah = header['PA']
    except:
        pah = None
        print("No PA keyword in FITS header")
    if pah is None:
        try:
            pah = header['ROT_PA']
        except:
            pah = None
            print("No ROT_PA keyword in FITS header")

    if pah is not None:
        pdiff = pa-pah
        if np.abs(pdiff) > 1:
            print('Position angle difference (calc-header): %.3f'%pdiff)


    photdata = optimal_aperture(image,header,x=x,y=y,ra=ra,dec=dec,plot=False,\
                                       skyrad=skyrad,\
                                       apmax=apmax)
    # Plate scale
    plate_scale = get_plate_scale(header)

    xcen = photdata['xcen']
    ycen = photdata['ycen']
    aspect = photdata['aspect']
    fwhm = photdata['fwhm']*plate_scale
    airmass = photdata['secz']
    opap = photdata['optimal_aperture']
    snr = photdata['snrmax']

    if 'BJD-OBS' not in header.keys():
        ra,dec = xy_to_radec(1024,1024,header)
        coords = SkyCoord(ra,dec,unit=(u.deg,u.deg))
        bjd = jd_to_bjd(header['JD'],coords)
    else:
        bjd = header['BJD-OBS']

    mid_time = bjd + 0.5*header['EXPTIME']/86400.0

    output = {'xcen':xcen,'ycen':ycen,'aspect':aspect,'fwhm':fwhm,'airmass':airmass,
              'optimal_aperture':opap,'snr':snr,'pa':pa, 'bjd':mid_time,'flipped':flip,
              'plate_scale':plate_scale}

    return output


######################################################################
# Return photometric information
def phot_info(cal,header,targname=None,targ_info=None):
    '''
    Description
    -----------
    Returns a dictionary with information about supplied targets for a
    calibrated image with header

    Inputs
    ------
    cal       = (float array) Calibrated image
    header    = (string array) FITS header from calibrated image
    targ_info = (dictionary, optional) Target information

    Outputs
    -------
    info = (dictionary) Photometry information

    Example
    -------

    '''

    # Get targ_info if not supplied
    if targ_info is None:
        targ_info = read_targ_info(targname)

    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))

    # Get position angle for image

    pa = posang(header)
    flip = posang(header,flipcheck=True)

    try:
        pah = header['PA']
    except:
        pah = None
        #print("No PA keyword in FITS header")
    if pah is None:
        try:
            pah = header['ROT_PA']
        except:
            pah = None
            #print("No PA keyword in FITS header")

    if pah is not None:
        pdiff = pa-pah
        if np.abs(pdiff) > 1:
            print('Position angle difference (calc-header): %.3f'%pdiff)

    # Convert ra,dec values to x,y
    x,y = radec_to_xy(targ_info['RAdeg'], targ_info['DECdeg'],header)

    # Plate scale
    plate_scale = get_plate_scale(header)

    aspect = []; fwhm = []; airmass = []; opap = []; snr = []
    xcen = []; ycen = []

    for i in range(len(x)):
        photdata = optimal_aperture(cal,header,x=x[i],y=y[i],plot=False,\
                                       skyrad=targ_info['skyrad'][i],\
                                       apmax=np.min(targ_info['ap_max']))
        xcen = np.append(xcen,photdata['xcen'])
        ycen = np.append(ycen,photdata['ycen'])
        aspect = np.append(aspect,photdata['aspect'])
        fwhm = np.append(fwhm,photdata['fwhm']*plate_scale)
        airmass = np.append(airmass,photdata['secz'])
        opap = np.append(opap,photdata['optimal_aperture'])
        snr = np.append(snr,photdata['snrmax'])

    if 'BJD-OBS' not in header.keys():
        ra,dec = xy_to_radec(1024,1024,header)
        coords = SkyCoord(ra,dec,unit=(u.deg,u.deg))
        bjd = jd_to_bjd(header['JD'],coords)
    else:
        bjd = header['BJD-OBS']

    mid_time = bjd + 0.5*header['EXPTIME']/86400.0

    output = {'xcen':xcen,'ycen':ycen,'aspect':aspect,'fwhm':fwhm,'airmass':airmass,
              'optimal_aperture':opap,'snr':snr,'pa':pa, 'bjd':mid_time,'flipped':flip,
              'plate_scale':plate_scale}

    return output


#----------------------------------------------------------------------#
# do_psf_photometry: do photometry using PSF fitting
#----------------------------------------------------------------------#

def do_psf_photometry(image,header,xpos,ypos,aperture,skyrad,recenter=True,skymed=False):

    '''
    - Can either give multiple positions, or multiple aperture sizes, not both!
    - Can give customized sky radii for each position or a single set of sky radii
      for all sources
    '''
    from photutils.detection import IRAFStarFinder
    from photutils.psf import IntegratedGaussianPRF, DAOGroup, IterativelySubtractedPSFPhotometry
    from photutils.background import MMMBackground, MADStdBackgroundRMS
    from astropy.modeling.fitting import LevMarLSQFitter
    from astropy.stats import gaussian_sigma_to_fwhm
    from photutils.psf.sandbox import DiscretePRF


    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)

    # Set this as a guess?
    sigma_psf = 2.0

    iraffind = IRAFStarFinder(threshold=3.5*std,
                              fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                              minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                              sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=iraffind.fwhm)
    photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                    group_maker=daogroup,
                                                    bkg_estimator=mmm_bkg,
                                                    psf_model=psf_model,
                                                    fitter=LevMarLSQFitter(),
                                                    niters=1, fitshape=(11,11))
    result_tab = photometry(image=image)
    residual_image = photometry.get_residual_image()





    # Checks!
    if length(aperture) > 1 and length(xpos) > 1:
        print(' - cannot give multiple apertures for multiple sources!')
        return None
    if length(xpos) > 1:
        if length(ypos) != length(xpos):
            print(' - coordinates do not have same length!')
            return None
        if length(np.array(skyrad).flatten()) != 2*length(xpos) and length(np.array(skyrad).flatten()) != 2:
            print(' - can either specify one set of sky radii, or the same number as positions!')
            return None

    outputs = {'counts':None, 'counterr':None,
               'flux':None, 'fluxerr':None,
               'xval':None, 'yval':None,
               'xcen':None, 'ycen':None,
               'skycounts':None,'skycounterr':None,
               'skyflux':None, 'skyfluxerr':None}




#----------------------------------------------------------------------#
# night_monitor: do photometry for an entire night on a target that is
# being monitored
#----------------------------------------------------------------------#

def night_monitor(date,write=True,clobber=False,obstype=None,targname=None,
                  targ_info=None,readnoise=False,rotated_flat=False):
    '''
    Description
    -----------
    Calculate the photometry for all the sources in targ_info for the entire night
    of observing, including all bands and all images.

    Inputs
    ------
    date      = (string) date in YYMMDD format
    targ_info = (dictionary) optional target info dictionary
    write     = (boolean) write out photometry in output directory?
    clobber   = (boolean) recompute photometry

    Outputs
    -------
    output = (dictionary) containing photometry information for the entire night
    keys:


    Example
    -------
    phot = night_monitor('20171108',targname='SN2018fgc',write=True,clobber=False)


    '''

    # Read photometry file if it is there
    #final_out = read_photometry(date,obstype=obstype,targname=targname)
    #if final_out is not None and not clobber:
        #return final_out

    # Get targ_info if not supplied
    if targ_info is None:
        targ_info = read_targ_info(obstype=obstype,targname=targname)

    if targ_info is None:
        print('There is no target information available for this source!')
        return None

    # Generate list of dictionary keys for photometry dictionary
    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))

    # Path and observation information
    paths = get_paths(obstype=obstype,targname=targname)

    try:
        info = get_obs(date,obstype=obstype,targname=targname)
        setting = info['setting']
    except:
        print('No summary file... assuming setting = 1')
        setting = 1

    # Load all files (all bands)
    allfiles,fct = get_files(date=date,prefix=targname)

    # Determine which filters were used
    filters = []
    for fname in allfiles:
        header = fits.getheader(fname)
        filters.append(header['filter'].replace("'",""))
    filters = np.unique(filters)

    fstr = ''
    for val in filters:
        fstr = fstr+val+', '

    print('Filters used on '+date+': '+fstr[:-2])
    print('')

    # Loop through all bands observed on that night
    final_out = {}
    try:
        os.mkdir(paths['output'] + str(date))
    except:
        print(str(date) + ' already exists!')

    for band in filters:
        print('Starting batch photometry for '+band+' band on '+date+'...')
        cals = get_cal_frames(date,band=band,targname=targname,obstype=obstype,
                              setting=setting,readnoise=readnoise)

        sfiles,sct =   get_files(date=date,prefix=targname,tag='-'+band,suffix='solved.fits')

        # Loop through each file in specified band
        i = 0
        outdict = {}
        pbar = tqdm(desc = 'Analyzing '+band+' band', total = sct, unit = 'file')
        for sfile in sfiles:
            try:
                image,header = read_image(sfile,plot=False)

                cal = calibrate_image(image,header,cals,rotated_flat=rotated_flat)

                out = aperture_photometry(cal,header,targ_info=targ_info,targname=targname)

                if i == 0:
                    outdict['filename'] = [sfile.split('/')[-1]]
                    outdict['bjd'] = [out['bjd']]
                    outdict['airmass'] = [out['airmass']]
                    for name in names:
                        outdict[name] = [out[name]]
                else:
                    outdict['filename'].append(sfile.split('/')[-1])
                    outdict['bjd'].append(out['bjd'])
                    outdict['airmass'].append(out['airmass'])
                    for name in names:
                        outdict[name].append(out[name])
                i += 1
            except:
                pass
            pbar.update(1)

        pbar.close()
        # Update final_out dictionary according to filter band
        final_out[band] = outdict

    # Write out photometry into output directory
    if write:
        print('Writing out photometry files into: ')
        print('    '+paths['output']+date+'/')
        fname = paths['output']+date+'/photometry_'+date+'.pck'
        fout = open(fname,"w")
        pickle.dump(final_out,fout)

    return final_out


#----------------------------------------------------------------------#
# night_phot: do photometry for an entire night
#----------------------------------------------------------------------#

def night_phot(date,write=True,clobber=False,obstype=None,targname=None,
               targ_info=None,readnoise=False,rotated_flat=False,setting=1,med_dist=False,
               calwrite=True,calarchive=False,distmax=200,saturation=30000.0,pamax=10,
               targRA=None,targDec=None,phot_tag='',tagpre=None,tagsuf=None,
               outdir=None):
    '''
    Description
    -----------
    Calculate the photometry for all the sources in targ_info for the entire night
    of observing, including all bands and all images.

    Inputs
    ------
    date      = (string) date in YYMMDD format
    targ_info = (dictionary) optional target info dictionary
    write     = (boolean) write out photometry in output directory?
    clobber   = (boolean) recompute photometry

    Outputs
    -------
    output = (dictionary) containing photometry information for the entire night
    keys:


    Example
    -------
    phot = night_phot('20171108',targname='SN2018fgc',write=True,clobber=False)


    '''

    if outdir is None:
        # Path and observation information
        paths = get_paths(obstype=obstype,targname=targname)
        outdir = paths['output']

    # Read photometry file if it is there
    final_out = read_photometry(date,obstype=obstype,targname=targname,phot_tag=phot_tag,outdir=outdir)
    if final_out is not None and not clobber:
        return final_out

    # Get targ_info if not supplied
    if targ_info is None:
        targ_info = read_targ_info(obstype=obstype,targname=targname)

    if targ_info is None:
        print('There is no target information available for this source!')
        return None

    # Generate list of dictionary keys for photometry dictionary
    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))

    # Load all files (all bands)
    allfiles,fct = get_files(date=date,prefix=targname)

    # Determine which filters were used
    filters = []
    for fname in allfiles:
        header = fits.getheader(fname)
        filters.append(header['filter'].replace("'",""))
    filters = np.unique(filters)

    fstr = ''
    for val in filters:
        fstr = fstr+val+', '

    print('Filters used on '+date+': '+fstr[:-2])
    print('')

    # Loop through all bands observed on that night
    final_out = {}
    try:
        os.mkdir(outdir + str(date))
    except:
        pass

    for band in filters:
        print('Starting batch photometry for '+band+' band on '+date+'...')
        cals = get_cal_frames(date,band=band,targname=targname,obstype=obstype,
                              setting=setting,readnoise=readnoise,write=calwrite,
                              archive=calarchive)

        if tagpre is not None:
            tag = tagpre+band
        if tagsuf is not None:
            tag = tag+tagsuf
        if tagpre is None and tagsuf is None:
            tag = '-'+band

        files,fct =   get_files(date=date,prefix=targname,tag=tag)

        if fct == 0:
            print('!!! No images found !!!')

        files = np.sort(files)

        print('Vetting images...')
        xpos = np.array([]) ; ypos = np.array([])  ; ccdtemp = [] ; pa = []
        maxcount = []
        pbar = tqdm(desc = 'Checking images', total = fct, unit = 'files')
        for f in files:
            im,header = read_image(f)
            if targRA is None:
                try:
                    ra = header['OBJCTRA']
                    dec = header['OBJCTDEC']
                except:
                    print('Need to input RA and Dec to vet images')
                    return {}
            else:
                ra = targRA
                dec = targDec

            coords = SkyCoord(ra,dec,unit=(u.hour,u.deg))
            RAdeg  = coords.ra.deg
            DECdeg = coords.dec.deg

            # x, y position of the target in the image
            x,y = radec_to_xy(RAdeg,DECdeg,header)
            xr = np.int(np.round(x))
            yr = np.int(np.round(y))
            # Check for saturated pixels at target position
            try:
                maxcount = np.append(maxcount,np.max(im[yr-10:yr+10,xr-10:xr+10]))
            except:
                maxcount = np.append(maxcount,np.nan)
            xpos = np.append(xpos,x)
            ypos = np.append(ypos,y)
            pa = np.append(pa,posang(header))
            ccdtemp = np.append(ccdtemp,header['CCD-TEMP'])
            pbar.update(1)
        pbar.close()

        if med_dist:
            xref = np.median(xpos) ; yref = np.median(ypos)
        else:
            xref = 1023 ; yref = 1023

        # How far is target from the reference position
        dists = np.sqrt((xpos-xref)**2+(ypos-yref)**2)
        # Position angle as min angle from 0 or 180 (allow for flip)
        patest = [min((p-180)%180,180-p%180) for p in pa]
        gargs, = np.where((dists < distmax) & (maxcount < saturation) & (np.abs(patest) < pamax))
        sfiles = [files[arg] for arg in gargs]
        sct = len(sfiles)
        if sct == 0:
            print('No images are suitable for photometry!')

        # Loop through each file in specified band
        i = 0
        outdict = {}
        pbar = tqdm(desc = 'Analyzing '+band+' band', total = sct, unit = 'file')
        for sfile in sfiles:
            print('Attempting '+sfile.split('/')[-1])
            try:
                image,header = read_image(sfile,plot=False)
                if targRA is None:
                    targRA = header['OBJCTRA']
                    targDec = header['OBJCTDEC']
                try:
                    header['HA'] = calculate_ha(header,targRA,targDec)
                except:
                    print('Hour angle cannot be computed')

                cal = calibrate_image(image,header,cals,rotated_flat=rotated_flat)
                # Get mechanical angle of rotator
                try:
                    pa = posang(header)
                    mechang  =  mech_pos(header['HA'],DECdeg,pa,latitiude=obs.lat,offset=cals['info']['offset'])
                except:
                    mechang = np.nan

                out = aperture_photometry(cal,header,targ_info=targ_info,targname=targname)
                if i == 0:
                    outdict['filename'] = [sfile.split('/')[-1]]
                    outdict['bjd'] = [out['bjd']]
                    outdict['mechang'] = [mechang]
                    if 'airmass' in header.keys():
                        outdict['airmass'] = [out['airmass']]
                    else:
                        outdict['airmass'] = [airmass_from_header(header)]
                    for name in names:
                        outdict[name] = [out[name]]
                else:
                    outdict['filename'].append(sfile.split('/')[-1])
                    outdict['bjd'].append(out['bjd'])
                    outdict['mechang'].append(mechang)
                    outdict['airmass'].append(airmass_from_header(header))
                    for name in names:
                        outdict[name].append(out[name])
                i += 1

            except:
                print('!!! Something went wrong with '+sfile)
            pbar.update(1)
        pbar.close()

        # Update final_out dictionary according to filter band
        final_out[band] = outdict

    # Write out photometry into output directory
    if write:
        print('Writing out photometry files into: ')
        print('    '+outdir+date+'/')
        fname = outdir+date+'/photometry_'+date+phot_tag+'.pck'
        fout = open(fname,"w")
        pickle.dump(final_out,fout)

    return final_out



######################################################################
# Read photometry dictionary for a specified night
def read_photometry(date,obstype=None,targname=None,phot_tag='',verbose=False,outdir=None):
    '''
    Description
    -----------
    Read photometry dictionary from local path

    Inputs
    ------
    date = (string) date of observation in YYMMDD format

    Outputs
    -------
    output = (dictionary) photometry dictionary produced by night_phot
             or night_monitor

    Example
    -------
    '''

    if outdir is None:
        paths = get_paths(obstype=obstype,targname=targname)
        outdir = paths['output']
    photfile = outdir+date+'/photometry_'+date+phot_tag+'.pck'
    exist = glob.glob(photfile)

    if len(exist) == 1:
        phot = pickle.load( open( photfile, "rb" ) )
    else:
        if verbose:
            print("No Photometry found for " + obstype + ", " + targname + " on " + date + "!")
        phot = None

    return phot

######################################################################
# Utilities
######################################################################
######################################################################

#----------------------------------------------------------------------
# JD to BJD
#----------------------------------------------------------------------
'''

Follow protocol on docs.astropy.org/en/stable/time/ to convert JD (UTC)
to BJD (TDB)

'''
def jd_to_bjd(jd,coords,location=obs_location):
    jdtime = Time(jd,format='jd')
    delta_bjd = jdtime.light_travel_time(coords,kind='barycentric',location=location)
    bjd = jdtime.tdb + delta_bjd
    return bjd.value


#---------------------------------------------------------------------#
# radec_to_xy:
#----------------------------------------------------------------------#


def radec_to_xy(ra,dec,header):
    """
    Overview:
    ---------
    Takes a ra, a dec, and a header and returns the x,y position of the source in the image
    ra and dec are degrees
    x and y are pixel values
    check header for astrometical slove
    """
    if header['CRVAL1']:
        w = wcs.WCS(header)
        c = SkyCoord(ra,dec, unit=(u.deg, u.deg))
        x, y = wcs.utils.skycoord_to_pixel(c, w)
        return x,y
    else:
        print("radec_to_xy: inadequate astrometry information")
        return None,None



#----------------------------------------------------------------------#
# decimal degree coordinates to hms dms
#----------------------------------------------------------------------#

def deg_to_hmsdms(ra,dec,alwayssign=False):

    coords = SkyCoord(ra,dec,unit=(u.degree,u.degree))
    rastr = str(coords.ra.to_string(u.hour))
    rastr = rastr.replace('h',':')
    rastr = rastr.replace('m',':')
    rastr = rastr.replace('s','')

    decstr = str(coords.dec.to_string(u.degree, alwayssign=alwayssign))
    decstr = decstr.replace('d',':')
    decstr = decstr.replace('m',':')
    decstr = decstr.replace('s','')

    return rastr, decstr

#----------------------------------------------------------------------#
# hms dms to decimal degree coordinates
#----------------------------------------------------------------------#
def hmsdms_to_deg(ra,dec,alwayssign=False):

    coords = SkyCoord(ra,dec,unit=(u.hour,u.degree))
    ra = coords.ra.deg
    dec = coords.dec.deg

    return ra,dec


#---------------------------------------------------------------------#
# xy_to_radec:
#----------------------------------------------------------------------#

def xy_to_radec(x,y,header):
    """
    Overview:
    ---------
    Takes an x, a y, and a header returns ra and dec of source
    x and y are pixel values
    ra and de are degrees
    """
    if header['CRVAL1']:
        w = wcs.WCS(header)
        coords = SkyCoord.from_pixel(x,y,w)
        if length(coords) == 1:
            ra = coords.ra.deg
            dec = coords.dec.deg
        elif length(coords) > 1:
            ra = np.array([bob.ra.deg for bob in coords])
            dec = np.array([bob.dec.deg for bob in coords])
        return ra,dec
    else:
        print("xy_to_radec: inadequate astrometry information")
        return None,None

#---------------------------------------------------------------------#
# done_in:                                                             #
#----------------------------------------------------------------------#

def done_in(tmaster):

    """
    Overview:
    ---------
    Simple routine to print out the time elapsed since input time

    Calling sequence:
    -----------------
    import time
    tstart = time.time()
    (stuff happens here)
    done_in(tstart)

    """

    t = time.time()
    hour = (t - tmaster)/3600.
    if np.floor(hour) == 1:
        hunit = "hour"
    else:
        hunit = "hours"

    minute = (hour - np.floor(hour))*60.
    if np.floor(minute) == 1:
        munit = "minute"
    else:
        munit = "minutes"

    sec = (minute - np.floor(minute))*60.

    if np.floor(hour) == 0 and np.floor(minute) == 0:
        tout = "done in {0:.2f} seconds"
        print(tout.format(sec))
    elif np.floor(hour) == 0:
        tout = "done in {0:.0f} "+munit+" {1:.2f} seconds"
        print(tout.format(np.floor(minute),sec))
    else:
        tout = "done in {0:.0f} "+hunit+" {1:.0f} "+munit+" {2:.2f} seconds"
        print(tout.format(np.floor(hour),np.floor(minute),sec))

    print(" ")

    return



#----------------------------------------------------------------------#
# get_plate_scale: get mean plate scale of image                       #
#----------------------------------------------------------------------#

def get_plate_scale(header,angle_unit='arcsec'):

    # Construct CD matrix from the header
    cd = np.array([[header['CD1_1'],header['CD1_2']],
    [header['CD2_1'],header['CD2_2']]])

    # Return plate scale estimate by taking mean of the
    # eigenvalues of the CD matrix
    ps = np.mean(np.abs(np.linalg.eig(cd)[0]))
    if angle_unit=='arcsec':
        val = 3600.0
    elif angle_unit=='arcmin':
        val = 60.0
    elif angle_unit=='degrees':
        val = 1.0
    elif angle_unit=='radians':
        val = np.pi/180.0

    plate_scale = ps*val

    return plate_scale


#----------------------------------------------------------------------#
# par_angle: parallactic angle of source                               #
#----------------------------------------------------------------------#


def par_angle(ha,dec,latitude=obs.lat):
    '''
    Returns the parallactic angle of a source given the hour angle, declination,
    and latitude of the observatory (default Thacher)

    PA is returned in degrees

    Stolen from T. Robishaw parangle
    http://www.cita.utoronto.ca/~tchang/gbt/procs/pangle/parangle.pro
    Thanks, Tim!
    '''
    har = ha*np.pi/180.0
    decr = dec*np.pi/180.0
    pa = -180/np.pi*\
         np.arctan2(-np.sin(har),
                 np.cos(decr)*np.tan(latitude)-np.sin(decr)*np.cos(har))
    return pa


#----------------------------------------------------------------------#
# mech_pos: find mechanical angle of rotator                           #
#----------------------------------------------------------------------#

def mech_pos(ha,dec,posangle,latitiude=obs.lat,offset=None):
    '''
    Uses par_angle and position angle of an image to calculate
    the mechanical angle of the rotator

    ha  = -1*(3+24.0/60.+41.31/3600.)*15.0
    dec = 48+1/60.0+43.2/3600
    pa  = 180-0.156
    offset = 181.0

    Before that: 181.0?
    9/30/19: mechanical offset 182.08

    '''

    parangle = par_angle(ha,dec)
    mech = parangle + offset - posangle

    if mech < 0:
        mech += 360.0

    return mech


#----------------------------------------------------------------------#
# posang: Position angle of image given solved header                  #
#----------------------------------------------------------------------#

def posang(header,verbose=False,flipcheck=False):
    '''
    Description
    -----------
    Lifted from the internet. It seems like it works, should be tested
    more extensively


    '''

    CD11 = float(header['CD1_1'])
    CD12 = float(header['CD1_2'])
    CD21 = float(header['CD2_1'])
    CD22 = float(header['CD2_2'])

    ## This is my code to interpet the CD matrix in the WCS and determine the
    ## image orientation (position angle) and flip status.  I've used it and it
    ## seems to work, but there are some edge cases which are untested, so it
    ## might fail in those cases.
    ## Note: I'm using astropy units below, you can strip those out if you keep
    ## track of degrees and radians manually.
    if (abs(CD21) > abs(CD22)) and (CD21 >= 0):
        North = "Right"
        positionAngle = 270.*u.deg + np.degrees(np.arctan(CD22/CD21))*u.deg
    elif (abs(CD21) > abs(CD22)) and (CD21 < 0):
        North = "Left"
        positionAngle = 90.*u.deg + np.degrees(np.arctan(CD22/CD21))*u.deg
    elif (abs(CD21) < abs(CD22)) and (CD22 >= 0):
        North = "Up"
        positionAngle = 0.*u.deg + np.degrees(np.arctan(CD21/CD22))*u.deg
    elif (abs(CD21) < abs(CD22)) and (CD22 < 0):
        North = "Down"
        positionAngle = 180.*u.deg + np.degrees(np.arctan(CD21/CD22))*u.deg
    if (abs(CD11) > abs(CD12)) and (CD11 > 0): East = "Right"
    if (abs(CD11) > abs(CD12)) and (CD11 < 0): East = "Left"
    if (abs(CD11) < abs(CD12)) and (CD12 > 0): East = "Up"
    if (abs(CD11) < abs(CD12)) and (CD12 < 0): East = "Down"
    imageFlipped = None
    if North == "Up" and East == "Left": imageFlipped = False
    if North == "Up" and East == "Right": imageFlipped = True
    if North == "Down" and East == "Left": imageFlipped = True
    if North == "Down" and East == "Right": imageFlipped = False
    if North == "Right" and East == "Up": imageFlipped = False
    if North == "Right" and East == "Down": imageFlipped = True
    if North == "Left" and East == "Up": imageFlipped = True
    if North == "Left" and East == "Down": imageFlipped = False

    pa = positionAngle.to(u.deg).value

    if verbose:
        print("Position angle of WCS is {0:.1f} degrees.".format(positionAngle.to(u.deg).value))
        print("Image orientation is North {0}, East {1}.".format(North, East))
        if imageFlipped:
            print("Image is mirrored.")

    if flipcheck:
        return imageFlipped
    else:
        return pa


#----------------------------------------------------------------------#
# Gaussian2D:                                                          #
#----------------------------------------------------------------------#

def Gaussian2D((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    Function to create a 2D Gaussian from input parameters.
    '''
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def calculate_airmass(alt):

    #cos of zenith angle
    cos_zt = np.cos(np.radians(90.0 - alt))
    # Airmass calculation
    airmass = ((1.002432*cos_zt**2) + (0.148386*cos_zt) + 0.0096467 ) / \
              ( (cos_zt**3) + (0.149864*cos_zt**2) + (0.0102963*cos_zt) + 0.000303978)

    return airmass



def airmass_from_header(header,verbose=False):

    try:
        jd = header['JD']
    except:
        if verbose:
            print('No JD info in header')
        return None

    try:
        rastr = header['OBJCTRA']
        decstr = header['OBJCTDEC']
        coords = SkyCoord(rastr,decstr,unit=(u.hour,u.degree))
        ra = np.radians(coords.ra.deg)
        dec = np.radians(coords.dec.deg)
    except:
        if verbose:
            print('Object RA and DEC not present in header')

    try:
        ra,dec = xy_to_radec(1024,1024,header)
        if verbose:
            print('Using RA, Dec, from center pixel')
            print('RA (deg) = %.2f'%ra)
            print('Dec (deg) = %.2f'%dec)
    except:
        if verbose:
            print('Inadequate header info')
        return None
    obs.date = ephem.date(jd-2415020.0) # pyephem uses Dublin Julian Day (why!!??!?)
    star = ephem.FixedBody()
    star._ra = np.radians(ra)
    star._dec = np.radians(dec)
    star.compute(obs)
    if verbose:
        print('Telescope altitude = %.2f'%star.alt)
    airmass = calculate_airmass(np.degrees(star.alt))

    return airmass



def sigmaRejection(data,m=0.1):
    '''
    Description
    -----------
    Returns new array with outliers rejected
    Stolen from J. Luebbers Data Science final exam


    Inputs
    ------
    m = (float) strength of sigma rejection

    Outputs
    -------
    data = (array)

    Example
    -------

    '''
    n = np.sqrt(2.)*erfinv(1-(m/len(data)))
    sigma = mad(data)
    r = [np.median(data)-(n*sigma),np.median(data)+(n*sigma)]
    return np.array([i for i in data if r[0]<i<r[1]])


#----------------------------------------------------------------------
# center_target: find centroid position of a chosen target
#----------------------------------------------------------------------

def center_target(image,xpos,ypos,plot=False,sz=60):
    '''
    Description:
    ------------
    Get new centroid position of target by fitting a Gaussian to the
    supplied x and y position of the target of interest

    '''

    # Create zoom in of target
    try:
        patch = image[int(round(ypos-sz/2)):int(round(ypos+sz/2)),
                      int(round(xpos-sz/2)):int(round(xpos+sz/2))]
    except:
        print(' - failed zooming in on target position')
        return None

    # Plot zoom in with fits overlaid
    sig = mad(patch.flatten())
    med = np.median(patch)
    vmin = med - 5*sig
    vmax = med + 15*sig
    if plot:
        plt.figure(2)
        plt.clf()
        plt.imshow(patch,vmin=vmin,vmax=vmax,cmap='gist_heat',
                   interpolation='nearest',origin='lower')
        plt.scatter(sz/2,sz/2,marker='+',s=200,color='yellow',
                    linewidth=1.5,label='Original position')
        plt.title('zoom in of target star')

    xg = np.linspace(0, sz-1, sz)
    yg = np.linspace(0, sz-1, sz)
    xg, yg = np.meshgrid(xg, yg)


    sigguess = 4/2.355/0.61
    p0 = [np.max(patch[int(round(sz/2-sigguess)):int(round(sz/2+sigguess)),
                       int(round(sz/2-sigguess)):int(round(sz/2+sigguess))]),
          sz/2,sz/2,sigguess,sigguess,0,med]


    # Fit 2D Guassian to target
    params, pcov = opt.curve_fit(Gaussian2D, (xg, yg), patch.ravel(), p0=p0)
    base = params[-1]
    peak = params[0]
    fwhm = np.sqrt(np.abs(params[3]*params[4]))*2.0*np.sqrt(2*np.log(2))
    aspect = min(np.abs(params[3]),np.abs(params[4]))/max(np.abs(params[3]),np.abs(params[4]))
    #norm = params[0] * 2.0 * np.pi * np.abs(params[3] * params[4])
    level = peak*np.array([0.1, 1./np.e, 0.95]) + params[-1]
    xpeak = params[1] + xpos - sz/2
    ypeak = params[2] + ypos - sz/2

    fit = None

    if plot:
        fit = Gaussian2D((xg, yg), *params)
        plt.contour(xg,yg,fit.reshape(sz,sz),level,colors='cyan')
        plt.scatter(params[1],params[2],marker='+',s=200,color='cyan',
                    linewidth=1.5,label='Updated position')

    # Chi Squared of Gaussian fit
    patchrms = np.sqrt(patch)
    xi,yi = np.where(patchrms == 0.0)
    if len(xi) > 0:
        patchrms[xi,yi] = 1.0
    try:
        chisq = np.sum((patch-fit.reshape(sz,sz))**2/patchrms**2)/ \
                (sz**2 - len(params) - 1.0)
    except:
        chisq =  np.nan

    out = {'aspect':aspect, 'fwhm':fwhm, 'peak': peak, 'fit':fit,
           'level':level,'xpeak':xpeak, 'ypeak':ypeak, 'chisq':chisq}

    return out

#----------------------------------------------------------------------
# calculate_ha: Calculate Hour Angle from header information
#----------------------------------------------------------------------

def calculate_ha(header,rastr=None,decstr=None):
    '''
    Description:
    ------------
    Calculate Hour Angle of a specified RA and Dec position given a
    FITS image header

    Input:
    ------
    header(string): FITS header
    ra (string):    Must be in sexigessimal hours
    dec(string):    Must be in sexigessimal degrees

    Output:
    -------
    Hour Angle

    '''
    if rastr is None or decstr is None:
        try:
            rastr = header['OBJCTRA']
            decstr = header['OBJCTDEC']
        except:
            print('calculate_ha: No RA or DEC info included!')
            return None
    coords = SkyCoord(rastr,decstr,unit=(u.hour,u.deg))

    time = Time(header['JD'],format='jd')
    ha = thacher.target_hour_angle(time,coords)
    if ha.hour > 12:
        HA = ha.hour - 24
    else:
        HA = ha.hour
    return HA


def check_headers(files,rastr=None,decstr=None,dosex=False,obstype=None,targname=None):

    fct = len(files)

    # Source extractor command
    if dosex:
        sextractor = '/home/administrator/sextractor-2.19.5/src/sex'

    # Check for which header to use as reference
    xpos = [] ; ypos = []
    pa = [] ; ccdtemp = []
    exptime = []
    ccdtemp = []; naxis1= []; naxis2 = []
    naxis = []; bjd = []
    fwhm = [] ; filename = []
    airmass = []
    pbar = tqdm(desc = 'Checking image headers', total = len(files), unit = 'file')
    for fname in files:
        header = fits.getheader(fname)
        cd = np.array([[header['CD1_1'],header['CD1_2']],
                       [header['CD2_1'],header['CD2_2']]])
        plate_scale = np.mean(np.abs(np.linalg.eig(cd)[0])*3600.0)
        if rastr is None:
            rastr  = header['OBJCTRA']
        if decstr is None:
            decstr = header['OBJCTDEC']
        coords = SkyCoord(rastr,decstr,unit=(u.hour,u.deg))
        RAdeg  = coords.ra.deg
        DECdeg = coords.dec.deg
        x,y    = radec_to_xy(RAdeg,DECdeg,header)
        xpos   = np.append(xpos,x)
        ypos   = np.append(ypos,y)
        pa     = np.append(pa,posang(header))
        try:
            ccdtemp = np.append(ccdtemp,header['CCD-TEMP'])
        except:
            ccdtemp = np.append(ccdtemp,990.0)
        try:
            airmass = np.append(airmass,airmass_from_header(header,verbose=False))
        except:
            if len(header['airmass']) == 1:
                airmass = np.append(airmass,header['AIRMASS'])
            else:
                airmass = np.append(airmass,-999)

        exptime = np.append(exptime,header['EXPTIME'])
        filename = np.append(filename,fname)
        naxis1 = np.append(naxis1,header['NAXIS1'])
        naxis2 = np.append(naxis2,header['NAXIS2'])
        naxis  = np.append(naxis,header['NAXIS'])
        try:
            newbjd = header['BJD-OBS']
        except:
            t = Time(header['JD'],format='jd',scale='utc',location=obs_location)
            ip_peg = SkyCoord(rastr, decstr, unit=(u.hour, u.deg), frame='icrs')
            ltt_bary = t.light_travel_time(ip_peg, 'barycentric')
            tbary = t.tdb + ltt_bary
            newbjd = tbary.value

        bjd    = np.append(bjd,newbjd)

        # Change pwd to file directory
        # This needs updating!! Probably broken
        if dosex:
            paths = get_paths(obstype=obstype,targname=targname)
            os.chdir(paths['output'])
            sexname = fname.replace('Dropbox (Thacher)','DropBox')
            make_sextractor_files()
            os.system(sextractor+' '+sexname)
            sexfiles = SExtractor()
            sexdata = sexfiles.read('temp.cat')
            fwhm = np.append(fwhm,np.median(sexdata['FWHM_IMAGE'][
                (sexdata['ELONGATION'] < 1.05)&(sexdata['FWHM_IMAGE'] > 1)])*plate_scale)
            os.system('rm default.param')
            os.system('rm default.conv')
            os.system('rm default.sex')
            os.system('rm temp.cat')

        pbar.update(1)
    pbar.close()

    out = {'filename':filename, 'xpos':xpos,'ypos':ypos,'PA':pa,'exptime':exptime,'ccdtemp':ccdtemp,
           'bjd':bjd,'naxis1':naxis1,'naxis2':naxis2,'naxis':naxis,'fwhm':fwhm,'airmass':airmass}

    return out



def make_sextractor_files(path='./',detect_thresh=None,analysis_thresh=None,minpix=None):
    '''
    Makes the necessary source extractor files to perform SExtractor on an image
    '''

    if detect_thresh is not None:
        dt = '%.1f'%detect_thresh
    else:
        dt = '1.5'

    if analysis_thresh is not None:
        at = '%.1f'%analysis_thresh
    else:
        at = '1.5'

    if minpix is not None:
        mp = '%.0f'%minpix
    else:
        mp = '3'

    # Default parameter file
    exist = glob.glob(path+'default.param')
    if len(exist) == 1:
        print('Deleting old default.param file')
        os.system('rm -rf '+exist[0])
    with open(path+'default.param', 'a') as the_file:
        the_file.write('NUMBER                   Running object number\n')
        the_file.write('X_IMAGE                  Object position along x                                   [pixel]\n')
        the_file.write('Y_IMAGE                  Object position along y                                   [pixel]\n')
        the_file.write('FLUX_BEST                Best of FLUX_AUTO and FLUX_ISOCOR                         [count]\n')
        the_file.write('FLUXERR_BEST             RMS error for BEST flux                                   [count]\n')
        the_file.write('ALPHA_J2000              Right ascension of barycenter (J2000)                     [deg]  \n')
        the_file.write('DELTA_J2000              Declination of barycenter (J2000)                         [deg]  \n')
        the_file.write('BACKGROUND               Background at centroid position                           [count]\n')
        the_file.write('THRESHOLD                Detection threshold above background                      [count]\n')
        the_file.write('THETA_J2000              Position angle (east of north) (J2000)                    [deg]  \n')
        the_file.write('ELONGATION               A_IMAGE/B_IMAGE                                                  \n')
        the_file.write('FWHM_IMAGE               FWHM assuming a gaussian core                             [pixel]\n')
        the_file.write('FWHM_WORLD               FWHM assuming a gaussian core                             [deg]  \n')

    # Default convolution file
    exist = glob.glob(path+'default.conv')
    if len(exist) == 1:
        print('Deleting old default.conv file')
        os.system('rm -rf '+exist[0])
    with open(path+'default.conv','a') as the_file:
        the_file.write('CONV NORM\n')
        the_file.write('# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.\n')
        the_file.write('1 2 1\n')
        the_file.write('2 4 2\n')
        the_file.write('1 2 1\n')

    # Default SExtractor file
    exist = glob.glob(path+'default.sex')
    if len(exist) == 1:
        print('Deleting old default.sex file')
        os.system('rm -rf '+exist[0])
    with open(path+'default.sex','a') as the_file:
        the_file.write('CATALOG_NAME     temp.cat       # name of the output catalog                    \n')
        the_file.write('CATALOG_TYPE     ASCII_HEAD     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,          \n')
        the_file.write('                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC          \n')
        the_file.write('PARAMETERS_NAME  default.param  # name of the file containing catalog contents  \n')
        the_file.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction) \n')
        the_file.write('DETECT_MINAREA   '+mp+'              # min. # of pixels above threshold              \n')
        the_file.write('DETECT_THRESH    '+dt+'            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2  \n')
        the_file.write('ANALYSIS_THRESH  '+at+'            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2  \n')
        the_file.write('FILTER           Y              # apply filter for detection (Y or N)?	        \n')
        the_file.write('FILTER_NAME      default.conv   # name of the file containing the filter        \n')
        the_file.write('DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds	        \n')
        the_file.write('DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending     \n')
        the_file.write('CLEAN            Y              # Clean spurious detections? (Y or N)?	        \n')
        the_file.write('CLEAN_PARAM      1.0            # Cleaning efficiency                           \n')
        the_file.write('WEIGHT_TYPE      NONE           # type of WEIGHTing: NONE, BACKGROUND,	        \n')
        the_file.write('                                # MAP_RMS, MAP_VAR or MAP_WEIGHT	        \n')
        the_file.write('WEIGHT_IMAGE     weight.fits    # weight-map filename                           \n')
        the_file.write('FLAG_IMAGE       flag.fits      # filename for an input FLAG-image	        \n')
        the_file.write('FLAG_TYPE        OR             # flag pixel combination: OR, AND, MIN, MAX     \n')
        the_file.write('                                # or MOST				        \n')
        the_file.write('PHOT_APERTURES   10             # MAG_APER aperture diameter(s) in pixels       \n')
        the_file.write('PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius> \n')
        the_file.write('PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,       \n')
        the_file.write('                                # <min_radius>				        \n')
        the_file.write('PHOT_AUTOAPERS   0.0,0.0        # <estimation>,<measurement> minimum apertures  \n')
        the_file.write('                                # for MAG_AUTO and MAG_PETRO		        \n')
        the_file.write('SATUR_LEVEL      30000.0        # level (in ADUs) at which arises saturation    \n')
        the_file.write('SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)        \n')
        the_file.write('MAG_ZEROPOINT    21.7           # magnitude zero-point			        \n')
        the_file.write('MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)    \n')
        the_file.write('GAIN             3.6            # detector gain in e-/ADU		        \n')
        the_file.write('GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU	        \n')
        the_file.write('PIXEL_SCALE      0              # size of pixel in arcsec (0=use FITS WCS info) \n')
        the_file.write('SEEING_FWHM      3.5            # stellar FWHM in arcsec		        \n')
        the_file.write('STARNNW_NAME     default.nnw    # Neural-Network_Weight table filename	        \n')
        the_file.write('BACK_TYPE        AUTO           # AUTO or MANUAL			        \n')
        the_file.write('BACK_VALUE       0.0            # Default background value in MANUAL mode       \n')
        the_file.write('BACK_SIZE        64             # Background mesh: <size> or <width>,<height>   \n')
        the_file.write('BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height> \n')
        the_file.write('CHECKIMAGE_TYPE  NONE           # can be NONE, BACKGROUND, BACKGROUND_RMS,      \n')
        the_file.write('                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,    \n')
        the_file.write('                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,    \n')
        the_file.write('                                # or APERTURES				        \n')
        the_file.write('CHECKIMAGE_NAME  check.fits     # Filename for the check-image		        \n')
        the_file.write('MEMORY_OBJSTACK  3000           # number of objects in stack		        \n')
        the_file.write('MEMORY_PIXSTACK  300000         # number of pixels in stack		        \n')
        the_file.write('MEMORY_BUFSIZE   1024           # number of lines in buffer		        \n')
        the_file.write('ASSOC_NAME       sky.list       # name of the ASCII file to ASSOCiate		\n')
        the_file.write('ASSOC_DATA       2,3,4          # columns of the data to replicate (0=all)	\n')
        the_file.write('ASSOC_PARAMS     2,3,4          # columns of xpos,ypos[,mag]			\n')
        the_file.write('ASSOC_RADIUS     2.0            # cross-matching radius (pixels)		\n')
        the_file.write('ASSOC_TYPE       NEAREST        # ASSOCiation method: FIRST, NEAREST, MEAN,	\n')
        the_file.write('                                # MAG_MEAN, SUM, MAG_SUM, MIN or MAX		\n')
        the_file.write('ASSOCSELEC_TYPE  MATCHED        # ASSOC selection type: ALL, MATCHED or -MATCHED\n')
        the_file.write('VERBOSE_TYPE     QUIET          # can be QUIET, NORMAL or FULL			\n')
        the_file.write('HEADER_SUFFIX    .head          # Filename extension for additional headers	\n')
        the_file.write('WRITE_XML        N              # Write XML file (Y/N)?				\n')
        the_file.write('XML_NAME         sex.xml        # Filename for XML output			\n')
        the_file.write('XSL_URL          file:///home/administrator/sextractor-2.19.5/xsl/sextractor.xsl\n')
        the_file.write('                                # Filename for XSL style-sheet                  \n')

    return


def do_sextractor(image, header, outdir='./',detect_thresh=5.0,analysis_thresh=5.0,minpix=3):
    '''
    Run SExtractor on an image, read the output file and return the data in a dictionary

    All auxiliary files are deleted after the procedure has been run

    '''
    # Path to the Source Extractor algorithm on bellerophon
    sextractor = '/home/administrator/sextractor-2.19.5/src/sex'

    # SExtractor wants an image file
    fname = 'temp.fits'
    os.chdir(outdir)
    fits.writeto(fname,image,header)

    # Clean up paths (use sym link to DropBox
    sexname = fname.replace('Dropbox (Thacher)','DropBox')

    # Make the necessary input files
    make_sextractor_files(detect_thresh=detect_thresh,analysis_thresh=analysis_thresh,
                             minpix=minpix)
    # Do it!
    os.system(sextractor+' '+sexname)

    # Read ouput
    sexfiles = SExtractor()
    data = sexfiles.read('temp.cat')

    # Get rid of auxiliary configuration and image files
    os.system('rm default.param')
    os.system('rm default.conv')
    os.system('rm default.sex')
    os.system('rm temp.cat')

    os.system('rm '+fname)

    return data



def make_dophot_param_file(image_in=None,outname=None,outdir='./',clobber=False,
                           fwhm_pix=None,snrmin=None,snrmax=None,skylev=None,gain=3.6,
                           nfitbox=10,maskbox=8,apbox=16,
                           saturation=30000,outputfile='param_default_c'):
    '''
    Make a DoPHOT parameter file in specified directory so that DoPHOT can be run on
    an image
    '''

    if outname is None:
        print("Must specify an output name")
        return

    gst = '%.1f'%gain

    # Check inputs
    if fwhm_pix is not None:
        fp = '%.1f'%fwhm_pix
    else:
        fp = '3.0'

    if snrmin is not None:
        smn = '%.1f'%snrmin
    else:
        smn = '5.0'

    if snrmax is not None:
        smx = '%.1f'%snrmax
    else:
        smx = '1000.0'

    if skylev is not None:
        sl = '%.0f'%skylev
    else:
        sl = '1000'

    sat = '%.0f'%saturation

    fb = '%d'%nfitbox
    mb = '%d'%maskbox
    ab = '%0.1f'%apbox

    # Default parameter file
    exist = glob.glob(outdir+outputfile)
    if len(exist) == 1:
        print('The file '+outputfile+' already exists')
        if clobber:
            print('Deleting old '+outputfile+' file')
            os.system('rm -rf '+exist[0])
        else:
            return

    with open(outdir+outputfile, 'a') as the_file:
        the_file.write("= \n")
        the_file.write("=  Default Parameter File for DoPHOT Version 2.0 .      October 31, 1991; MIT. \n")
        the_file.write("= \n")
        the_file.write("=  FREQUENTLY UPDATED PARAMETERS.  These parameters mostly depend on the       \n")
        the_file.write("=  typical size of stellar objects on a given frame and on the mean sky value  \n")
        the_file.write("=  for the image.  These parameters tend to be different for every frame.      \n")
        the_file.write("=  Also included are the bookkeeping parameters that specify file names.       \n")
        the_file.write("=                                                                              \n")
        the_file.write("FWHM = "+fp+"          Approx FWHM of objects (pixels) along major axis.       \n")
        the_file.write("MINFWHM = 1.0                                                                  \n")
        the_file.write("AXIS_RATIO = 1.0       For star objects.  AR=b/a; b=minor axis.                \n")
        the_file.write("TILT = 0.0             Angle of major axis in degrees; +x=0; +y=90.            \n")
        the_file.write("SKY = "+sl+"             Approximate mean sky value in data numbers.           \n")
        the_file.write("NFITBOX_X = "+fb+"         Size of fit box in the x-direction. \n")
        the_file.write("NFITBOX_Y = "+fb+"         Size of fit box in the y-direction. \n")
        the_file.write("MASKBOX_X = "+mb+"         Size of mask box size in x. \n")
        the_file.write("MASKBOX_Y = "+mb+"         Size of mask box size in y. \n")
        the_file.write("APBOX_X = "+ab+"         Size of aperture photometry box in x. \n")
        the_file.write("APBOX_Y = "+ab+"         Size of aperture photometry box in y. \n")
        the_file.write("IBOTTOM = 1            Lowest allowed data value in data numbers. \n")
        the_file.write("ITOP = "+sat+"           Maximum allowed data value in data numbers.           \n")
        the_file.write("SNTHRESHMIN = "+smn+"      Value of lowest S/N threshold.                      \n")
        the_file.write("SNTHRESHMAX = "+smx+"    Value of maximum S/N threshold.                       \n")
        the_file.write("SNTHRESHDEC = 2.0      The S/N threshold decrement:SNTHRESH(new)=SNTHRESH(old)/SNTHRESHDEC \n")
        the_file.write("SNMIN4FIXPSF = 20.0    The average PSF is not changed anymore for S/N<=SNMIN4FIXPSF \n")
        the_file.write("MASK_FILLVALUE = 0     all pixels in the image which pixel values = MASK_FILLVALUE are ignored \n")
        the_file.write("EPERDN = "+gst+"           Electrons per data number. \n")
        the_file.write("RDNOISE = 3.0          Readout noise in electrons. \n")
        the_file.write("= \n")
        the_file.write("= NEW PARAMETERS!!!! \n")
        the_file.write("= \n")
        the_file.write(" \n")
        the_file.write("MAXSTARS = 1000000	Defines how many stars can be found \n")
        the_file.write("FLOATSCALE = 1		float data values are multiplied by this value \n")
        the_file.write("=			to ensure that the values are smaller than 32767 \n")
        the_file.write("FORCEFITFLAG = 0	Forces fixed values of sigx, sigxy and sigy \n")
        the_file.write("F_SIGX = 0		If FORCEFITFLAG, then value of sigx \n")
        the_file.write("F_SIGXY = 0		If FORCEFITFLAG, then value of sigxy \n")
        the_file.write("F_SIGY = 0		If FORCEFITFLAG, then value of sigy \n")
        the_file.write("SKIPTYPE_6_8 = 1	If SKIPTYPE_6_8 then no type 6 and 8 in output object file  \n")
        the_file.write("= \n")
        the_file.write("AUTOSCALE = 'YES'       Auto-scaling of sizes by FWHM. \n")
        the_file.write("AUTOIBOTTOM = 'NO'     Auto-scaling of IBOTTOM \n")
        the_file.write("FIXPOS = 'NO'          Fix star positions? \n")
        the_file.write("= \n")
        the_file.write("PARAMS_DEFAULT = '"+outputfile+"'   Default parameters file name. \n")
        the_file.write("PARAMS_OUT = ' '		     Output parameters file name. \n")
        the_file.write("IMAGE_IN = '"+image_in+"'               Input image name. \n")
        the_file.write("IMAGE_OUT = ' '                      Output image name.  \n")
        the_file.write("OBJECTS_IN = ' '                     Input object list file name. \n")
        the_file.write("OBJECTS_OUT = 'obj_out_"+outname+"'         Output object list file name. \n")
        the_file.write("EMP_SUBRAS_OUT = 'psf_"+outname+".fits'       Empirical PSF subraster. \n")
        the_file.write("SHADOWFILE_IN = ' '                  Input shadow file name. \n")
        the_file.write("SHADOWFILE_OUT = ' '                 Output shadow file name. \n")
        the_file.write("LOGFILE = 'TERM'                     Log file name.  TERM for screen. \n")
        the_file.write("LOGVERBOSITY = 1                     Verbosity of log file; (0-4). \n")
        the_file.write("APCORRFILE = ' '                     Aperture correction file name. \n")
        the_file.write("= \n")
        the_file.write("= \n")
        the_file.write("=  OCCASIONALLY UPDATED PARAMETERS.  These parameters tend to not change for a \n")
        the_file.write("=  set of images obtained during a single observing run or for frames of a \n")
        the_file.write("=  single field.  The defaults for the flags PSFTYPE, SKYTYPE, and OUTTYPE are \n")
        the_file.write("=  PGAUSS, PLANE, and FULL, respectively. \n")
        the_file.write("= \n")
        the_file.write("RESIDNOISE = 0.5       Fraction of noise to ADD to noise file. \n")
        the_file.write("FOOTPRINT_NOISE = 1.3  Expand stars in noise file by this amount. \n")
        the_file.write("NPHSUB = 1             Limiting surface brightness for subtractions. \n")
        the_file.write("NPHOB = 1              Limiting surface brightness for obliterations. \n")
        the_file.write("ICRIT = 10             Obliterate if # of pixels > ITOP exceeds this. \n")
        the_file.write("CENTINTMAX = 50000.0   Obliterate if central intensity exceeds this. \n")
        the_file.write("CTPERSAT = 40000       Assumed intensity for saturated pixels. \n")
        the_file.write("STARGALKNOB = 2.0      Star/galaxy discriminator. \n")
        the_file.write("STARCOSKNOB = 1.0      Object/cosmic-ray discriminator. \n")
        the_file.write("SNLIM7 = 7.0           Minimum S/N for 7-parameter fit. \n")
        the_file.write("SNLIM = 0.5            Minimum S/N for a pixel to be in fit subraster. \n")
        the_file.write("SNLIMMASK = 4.0        Minimum S/N through mask to identify an object. \n")
        the_file.write("SNLIMCOS  = 3.0        Minimum S/N to be called a cosmic ray. \n")
        the_file.write("NBADLEFT = 0           Ignore pixels closer to the left edge than this. \n")
        the_file.write("NBADRIGHT = 0          Ignore pixels closer to the right edge than this. \n")
        the_file.write("NBADTOP = 0            Ignore pixels closer to the top edge than this. \n")
        the_file.write("NBADBOT = 0            Ignore pixels closer to the bottom edge than this. \n")
        the_file.write("= \n")
        the_file.write("PSFTYPE = 'PGAUSS'        PSF type: (PGAUSS) \n")
        the_file.write("SKYTYPE = 'MEDIAN'        SKY type: (PLANE, HUBBLE, MEDIAN) \n")
        the_file.write("= \n")
        the_file.write("JHXWID = 20              X Half-size of median box (.le. 0 -> autoscale) \n")
        the_file.write("JHYWID = 20              Y (same as above) \n")
        the_file.write("MPREC = 1               Median precision in DN (use .le. 0 for autocalc) \n")
        the_file.write("OBJTYPE_IN = 'COMPLETE'   Input format: (COMPLETE, INTERNAL) \n")
        the_file.write("OBJTYPE_OUT = 'COMPLETE'  Output format: (COMPLETE, INCOMPLETE, INTERNAL) \n")
        the_file.write("= \n")
        the_file.write("= \n")
        the_file.write("=  RARELY UPDATED PARAMETERS.  These are specialized parameters that rarely \n")
        the_file.write("=  need changing even when measuring images of very different sorts of fields \n")
        the_file.write("=  and/or from different telescope/detector combinations. \n")
        the_file.write("= \n")
        the_file.write("NFITITER = 50          Maximum number of iterations. \n")
        the_file.write("NPARAM = 7             Maximum number of PSF fit parameters. \n")
        the_file.write("NFITMAG = 4            No. of PSF parameters to get magnitudes. \n")
        the_file.write("NFITSHAPE = 7          No. of PSF parameters to get shape and mags. \n")
        the_file.write("NFITBOXFIRST_X = 15    Size of fit box in x for first pass. \n")
        the_file.write("NFITBOXFIRST_Y = 15    Size of fit box in y for first pass. \n")
        the_file.write("CHI2MINBIG = 30        Critical CHI-squared for a large object. \n")
        the_file.write("XTRA = 40              We need more S/N if some pixels are missing. \n")
        the_file.write("SIGMA1 = 0.10          Max. frac. scatter in sigma_x for stars. \n")
        the_file.write("SIGMA2 = 0.10          Max. scatter in xy cross term for stars. \n")
        the_file.write("SIGMA3 = 0.10          Max. frac. scatter in sigma_y for stars. \n")
        the_file.write("ENUFF4 = 0.30          Fraction of pixels needed for 4-param fit. \n")
        the_file.write("ENUFF7 = 0.30          Fraction of pixels needed for 7-param fit. \n")
        the_file.write("COSOBLSIZE = 0.9       Size of obliteration box for a cosmic ray. \n")
        the_file.write("APMAG_MAXERR = 0.1     Max anticipated error for aperture phot report. \n")
        the_file.write("BETA4 = 1.0            R**4 coefficient modifier. \n")
        the_file.write("BETA6 = 1.0            R**6 coefficient modifier. \n")
        the_file.write("= \n")
        the_file.write("= \n")
        the_file.write("=  AUTO SCALING PARAMETERS.  These parameters are used only if the auto-scaling \n")
        the_file.write("=  flags are turned on.  Box sizes, and threshold levels can be scaled \n")
        the_file.write("=  according to the FWHM of objects and the sky and readout noise values. \n")
        the_file.write("= \n")
        the_file.write("SCALEFITBOX = 2.5      Size of fit box in units of FWHM. \n")
        the_file.write("FITBOXMIN = 5.0        Smallest allowed fit box size. \n")
        the_file.write("SCALEAPBOX = 6.0       Size of aperture phot box in units of FWHM. \n")
        the_file.write("APBOXMIN = 7.0         Smallest allowed aperture phot box size. \n")
        the_file.write("SCALEMASKBOX = 1.5     Size of mask box in units of FWHM. \n")
        the_file.write("AMASKBOXMIN = 3.0      Smallest allowed mask box size. \n")
        the_file.write("NSIGIBOTTOM = 10.0   Level of IBOTTOM below sky in units of noise. \n")
        the_file.write("= \n")
        the_file.write("= \n")
        the_file.write("=  PARAMETER LIMITS.  These variables limit the legal ranges of the PSF and \n")
        the_file.write("=  sky parameters.  Be sure to understand their function well before changing \n")
        the_file.write("=  any of these values.  Positive values refer to fractional changes; negative \n")
        the_file.write("=  values to absolute changes; zero ABSLIM's turn the corresponding tests off. \n")
        the_file.write("= \n")
        the_file.write("RELACC1 = 0.01         Convergence criterion for sky. \n")
        the_file.write("RELACC2 = -0.03        Convergence criterion for x-position. \n")
        the_file.write("RELACC3 = -0.03        Convergence criterion for y-position. \n")
        the_file.write("RELACC4 = 0.01         Convergence criterion for for central intensity. \n")
        the_file.write("RELACC5 = 0.03         Convergence criterion for sigma-x. \n")
        the_file.write("RELACC6 = 0.1          Convergence criterion for sigma-xy. \n")
        the_file.write("RELACC7 = 0.03         Convergence criterion for sigma-y. \n")
        the_file.write("ABSLIM1 = -1.0e8       Allowed range for sky value. \n")
        the_file.write("ABSLIM2 = -1.0e3       Allowed range for x-position. \n")
        the_file.write("ABSLIM3 = -1.0e3       Allowed range for y-position. \n")
        the_file.write("ABSLIM4 = -1.0e8       Allowed range for central intensity. \n")
        the_file.write("ABSLIM5 = -1.0e3       Allowed range for sigma-x. \n")
        the_file.write("ABSLIM6 = -1.0e3       Allowed range for sigma-xy. \n")
        the_file.write("ABSLIM7 = -1.0e3       Allowed range for sigma-y. \n")
        the_file.write("END \n")

    return



######################################################################
# Under construction
######################################################################

def rundophot(image=None,header=None,filename=None,targname=None,obstype=None,outdir='./',
              tag='',clobber=False,SN=False):

    """
    Runs cdophot on normal images


    Parameters
    ----------
    target: str
        name of the target. e.g. "NGC0865"
    thefile: str, optional
        name of the fits file, if None, default to the first template image of the target
        must be from the same target
    tag: str
        '_sci' for science images, '_temp' for template images
    redo: bool
        if True, rerun cdophot
        if False, exists the function if file already exist

    Writes
    ------
    In /home/student/run_dophot/:
        * Modifies params_default_c
        * Copies image fits file into the directory (WHY?!?)
        * Outputs obj_out file
        * Outputs psf shape image
    """


    # UNIX command for DoPHOT
    cdophot = '/home/student/photpipe/Cfiles/bin/linux/cdophot'

    # Current working directory
    pwd = os.getcwd()

    # Check inputs copy or make DoPHOT target file
    tempfile = 'dophot_targfile.fits'
    os.chdir(outdir)
    if filename is None and (image is not None and header is not None):
        fits.writeto(image,header,tempfile)
    elif filename is not None:
        filename = filename.replace('Dropbox (Thacher)','DropBox')
        shutil.copy(filename,tempfile)
        header = fits.getheader(tempfile)
    else:
        print("Must input an image and header or filename")
        return

    # Make tag include time stamp
    objname = header['OBJECT']
    print('==========================')
    print('Running Dophot...')
    print('Target:  ',objname)
    print('Date:    ',header['DATE-OBS'])
    timestamp = header['DATE-OBS'].replace(':','')
    basename = filename.split('/')[-1]

    outname = objname+'_'+timestamp
    # Don't redo DoPHOT unless clobber is set
    exist = glob.glob('obj_out_'+outname)
    if len(exist) == 1:
        print('DoPHOT results Already exist! Exiting...')
        return

    # Do some stats here to make a good dophot param file!
    #

    # Make parameter file for DoPHOT and put it in the present directory
    make_dophot_param_file(image_in=tempfile,outname=outname,outdir='./')

    ################################
    #Run DoPHOT
    print("Running DoPHOT...")
    os.chdir(outdir)
    os.system(cdophot+' param_default_c')

    try:
        obj_out = pp.read_dophot_output('obj_out_'+targname,filter_bad=filter_bad)
    except:
        print('Something went wrong with DoPHOT.')
        obj_out = None

    os.chdir(pwd)

    return obj_out


def read_dophot_output(filename,outfile=None,type_1_only=False,filter_bad=True):

    header_names = ['i','type','xpos','ypos','fmag','dfmag',
                'flux','dflux','sky','peakflux','sigx',
                'sigxy','sigy','FWHM1','FWHM2','tilt',
                'ext.','chin']

    try:
        obj_out = pd.read_table(filename, names = header_names,sep = "\s+|\t+|\s+\t+|\t+\s+",skiprows = 11,engine='python')
        if filter_bad:
            obj_out = obj_out[obj_out['type']!='saturated'][obj_out['type']!='toofar']
        if type_1_only:
            obj_out = obj_out[obj_out['type']=='0x01']
    except:
        obj_out = pd.read_csv(obj_out_name, names = header_names,skiprows = 11)
        if filter_bad:
            obj_out = obj_out[obj_out['type']!='saturated'][obj_out['type']!='toofar']
        if type_1_only:
            obj_out = obj_out[obj_out['type']=='0x01']

    if outfile is not None:
        pass


    return obj_out
