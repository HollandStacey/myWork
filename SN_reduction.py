import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import thacherphot as tp
from quick_image import *
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
import pickle,os,inspect,pdb,socket
import thacherphot as tp
from astropy.table import Table
import photutils as pu
from statsmodels import robust
from statsmodels.stats.diagnostic import normal_ad
from scipy.special import erfinv
From FITS_tools.hcongrid import hcongrid
from scipy.interpolate import CubicSpline
from tqdm import tqdm

'''
HISTORY
-------
2018-1-4:   First adaptation from Tabby script
2018-1-19:  First attempt to implement this script for SN2017fgc

NOTES
-----
variables that need to change in order to be totally general
1) outpath in get_paths
2) File prefix in night_phot

'''

'''
katie: 
    
gv: start doing

oh: test photutils psf
    
fgc: sigma reject before taking std so accounts for outliers on nights
with really high error.  then can just take a mean and std of sigma rejected
because no outliers
clean up fgc completely


'''
######################################################################
# Global variables that need to be named
#targname = 'SN2018oh'
#prefix = 'SN2018oh'
#targname = 'SN2018gv'
#prefix = 'SN2018gv'
#targname = 'SN2017fgc'
#prefix = 'SN2017fgc'
targname = 'AT2018cow'
prefix = 'AT2018cow'


####################################################################################
# Get appropriate paths
def get_paths(targname):
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
    dpath = (string) path to raw data
    opath = (string) output path
    archive = (string) path to archive
    execpath = (string) path to directory of this file

    Example
    -------
    dpath,opath,archive,execpath = get_paths()

    '''

    # Environment variables
    user = os.environ['USER']
    home = os.environ['HOME']

    # Host not present on all operating systems
    try:
        host = os.environ['HOST']
    except:
        host = socket.gethostname()

    # Logic statements assigning correct paths
    if host == 'bellerophon' and user == 'administrator':
        dpath = "/home/administrator/Dropbox (Thacher)/Archive/"
        opath = '/home/administrator/SNe/'+targname+'/'
        archive = dpath

    if host == 'bellerophon' and user == 'student':
        dpath = "/home/administrator/Dropbox (Thacher)/Archive/"
        opath = '/home/student/SNe/'+targname+'/'
        archive = dpath

    if host == 'mojave' and user == 'jonswift':
        dpath = '/Users/jonswift/Dropbox (Thacher)/Archive/'
        opath = home+'/Astronomy/ThacherObservatory/SNe/'+targname+'/'
        archive = dpath

    if host == 'munch.local' and user == 'jonswift':
        dpath = '/Users/jonswift/Dropbox (Thacher)/Archive/'
        opath = home+'/Astronomy/ThacherObservatory/SNe/'+targname+'/'
        archive = dpath

    if host == 'Katies-MacBook-Air.local' and user == 'katieoneill':
        dpath = '/Users/katieoneill/Dropbox (Thacher)/Astronomy Archive/'
        opath = home + '/Astronomy/Data/'
        archive = dpath

    # Add your local path here if desired...
    # if host == 'yourhost' and user == 'you':
    
    execpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/'
    
    # Create output director if it does not exist already
    if not os.path.isdir(opath):
        mkdircmd = 'mkdir '+opath
        os.system(mkdircmd)
    
    return dpath,opath,archive,execpath


####################################################################################
# Read obs_summary file
def get_summary(targname):
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
    dpath,opath,archive,execpath = get_paths(targname)

    # Read obs_summary file in output directory
    obs = pd.read_csv(opath+targname+'_summary.csv')

    return obs


####################################################################################
# Parse obs_summary file into a dictionary for a given date
def get_obs(date,targname):
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

    dpath,opath,archive,execpath = get_paths(targname)

    obs = get_summary(targname)

    if np.sum(obs['date'] == int(date)) == 0:
        print 'No photometry for '+date+'!'
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
            
    files = glob.glob(opath+date)
    if len(files) == 0:
        mkdircmd = 'mkdir '+opath+date
        os.system(mkdircmd)

    return info

####################################################################################
# Return all dates for which target was observed
def get_dates(targname=targname,prefix=None,counts=False):

    dpath,opath,archive,execpath = get_paths(targname)
    
    dates = []
    for root, dirs, files in os.walk(dpath):
        for file in files:
            if file.startswith(prefix) and file.endswith("solved.fits"):
                dates.append(root.split('/')[-1])
    dates = np.array(dates)
    dates.sort()
    dates,count = np.unique(dates,return_counts=counts)
    
    if counts:
        rval  = dates,count
    else:
        rval = dates
    return rval


####################################################################################
# Calibration frames
def get_cal_frames(date,setting=None,readnoise=False,band='V',flatten=False,targname=None):
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
    flatten = (boolean) use flattened flat fields (use with caution)

    Outputs
    -------
    bias = (float array) bias frame (counts)
    dark = (float array) dark frame (counts/sec)
    flat = (float array) flat field (relative sensitivity)

    Example
    -------
    bias,dark,flat = get_cal_frames('20171108',setting=1,readnoise=True,band='V')

    '''
    # Get paths
    dpath,opath,archive,execpath = get_paths(targname)

    # Make master bias from nightly calibrations, else use master in archive
    biasfiles,bct = tp.get_files(d=dpath+date+'/',prefix='Bias',tag='1X1',suffix='fts')
    if bct > 0:
        bias = tp.master_bias(biasfiles,readnoise=readnoise,tag='_'+date,outdir=opath+date+'/')
    if bct == 0 and setting == 1:
        try:
            bias,bh = fits.getdata(archive+'calfiles/master_bias.fits', 0, header=True)
            print 'Using master biases'
        except:
            print 'No bias frame!'
            bias = None

    # Make master dark from nightly calibrations, else use master in archive
    darkfiles,dct = tp.get_files(d=dpath+date+'/',prefix='Dark',tag='1X1',suffix='fts')
    if dct > 0:
        if bias is None:
            print ''
            print 'DATE: '+date
            print 'WARNING: creating dark frame with no bias!!!'
            pdb.set_trace()
        dark = tp.master_dark(darkfiles,bias=bias,tag='_'+date,outdir=opath+date+'/')
    if dct == 0 and setting == 1:
        try:
            dark,dh = fits.getdata(archive+'calfiles/master_dark.fits', 0, header=True)
            print 'Using master dark'
        except:
            print 'No dark frame!'
            dark = None


    # Flat fields have not been a standard calibration. So read from calfiles archive
    if flatten:
        flatfile = 'master_flat_'+band+'_flattened.fits'
    else:
        flatfile = 'master_flat_'+band+'.fits'

    # Look for correct directory for master flats
    if int(date) <= 20171202:
        fdir = 'Pre-20171202'
    if int(date) >= 20171203 and int(date) <= 20180211:
        fdir = '20171203-20180211'
    if int(date) >= 20180212 and int(date) <= 20180421:
        fdir = '20180212-20180421'
    if int(date) >= 20180422:
        fdir = '20180422'

    try:
        flat,fh = fits.getdata(archive+'calfiles/'+fdir+'/'+flatfile, 0, header=True)
    except:
        print 'No flat frame!'
        flat = None

    return bias,dark,flat



######################################################################
# Master list of target information
def target_info(targname):
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
    targ_info = target_info()

    '''
  

    dpath,opath,archive,execpath = get_paths(targname)
    fname = opath+targname+'_target_info.pck'
    test = glob.glob(fname)
    if len(test) == 1:
        dict = pickle.load( open( fname, "rb" ) )
        return dict
    else:
        print('No target info file found for '+targname)
        return None


######################################################################
# Sigma rejection (for background calculations or whatever else)

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
    sigma = robust.mad(data)*1.4826
    r = [np.median(data)-(n*sigma),np.median(data)+(n*sigma)]
    return np.array([i for i in data if r[0]<i<r[1]])



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
        targ_info = target_info(targname)

    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))
        
    # Get position angle for image
    try:
        pa = header['PA']
    except:
        pa = None
        print("No PA keyword in FITS header")
    try:
        pa = header['ROT_PA']
    except:
        pa = None
        print("No PA keyword in FITS header")

    # Convert ra,dec values to x,y
    w = wcs.WCS(header)
    world = np.array([targ_info['RAdeg'], targ_info['DECdeg']]).T
    pix = w.all_world2pix(world,0) # Pixel coordinates of (RA, DEC)
    x = pix[:,0]
    y = pix[:,1]

    # Plate scale
    cd = np.array([[header['CD1_1'],header['CD1_2']],
    [header['CD2_1'],header['CD2_2']]])

    plate_scale = np.mean(np.abs(np.linalg.eig(cd)[0])*3600.0)
    
    aspect = []
    fwhm = []
    airmass = []
    opap = []
    snr = []
    xcen = []
    ycen = []
    for i in range(len(x)):
        photdata = tp.optimal_aperture(cal,header,x=x[i],y=y[i],plot=False,\
                                       skyrad=targ_info['skyrad'][i],\
                                       apmax=np.min(targ_info['ap_max']))
        xcen = np.append(xcen,photdata['xcen'])
        ycen = np.append(ycen,photdata['ycen'])
        aspect = np.append(aspect,photdata['aspect'])
        fwhm = np.append(fwhm,photdata['fwhm']*plate_scale)
        airmass = np.append(airmass,photdata['secz'])
        opap = np.append(opap,photdata['optimal_aperture'])
        snr = np.append(snr,photdata['snrmax'])

    bjd_mid = header['BJD-OBS'] + 0.5*header['EXPTIME']/86400.0
    
    airmass = np.median(airmass)
    
    output = {'xcen':xcen,'ycen':ycen,'aspect':aspect,'fwhm':fwhm,'airmass':airmass,
            'optimal_aperture':opap,'snr':snr,'pa':pa,'bjd':bjd_mid}

    return output


######################################################################
# Perform aperture photometry on a calibrated image
def aperture_phot(cal,header,position=None,skyrad=None,aprad=None,fraclev=0.75):
    '''
    Description
    -----------
    Perform aperture photometry on a calibrated image at a given position with
    provided sky radii and aperture radius. 

    Inputs
    ------
    cal      = (float) calibrated image (counts)
    header   = (string array) FITS header    
    position = (coordinate pair) x,y pixel position to perform aperture photometry
    skyrad   = (2 element array) The inner and outer pixel radii for sky calculation
    aprad    = (float) radius of photometry aperture.
    fraclev  = (float) Fraction (less than one) above which will be considered when
               calculating the statistics of the sky background and saturation level
               within the aperture.

    Outputs
    -------
    info   = (dictionary) Aperture photometry 

    Example
    -------
    '''


    # First check for saturation
    #---------------------------
    # Create aperture with specified radius
    aperture = pu.CircularAperture(position, r=aprad)
    # Count flux in aperture
    phot = pu.aperture_photometry(cal,aperture,method='exact')
    # Mask data in aperture
    apmask = aperture.to_mask()[0]
    starim = apmask.cutout(cal)
    startemp = apmask.multiply(cal)
    starfracs = startemp/starim
    xinds,yinds = np.where(starfracs > fraclev)
    starvals = starim[xinds,yinds]
    maxcts = np.max(starvals)

    # Find background level
    #----------------------
    # Create background annulus
    bkg_aperture = pu.CircularAnnulus(position, r_in=skyrad[0], r_out=skyrad[1])       
    # Create masked image
    skymask = bkg_aperture.to_mask()[0]
    skyim = skymask.cutout(cal)
    skytemp = skymask.multiply(cal)
    skyfracs = skytemp/skyim
    xinds,yinds = np.where(skyfracs > fraclev)
    skyvals = skyim[xinds,yinds]
    # Test for normally distributed sky pixels
    AD,p = normal_ad(skyvals)

    # Compute median background level
    bkg = np.median(skyvals)/header['exptime']
    bkg_rms = np.std(skyvals)/header['exptime']
        
    # Final flux
    final_phot = pu.aperture_photometry(cal/header['exptime']-bkg,aperture,method='exact')

    # Return info dictionary
    out = {'flux': final_phot['aperture_sum'].tolist()[0],
           'background':bkg, 'background_rms':bkg_rms, 'normal_pval':p, 'max counts':maxcts}
    
    return out


######################################################################
# Perform aperture photometry on all targets.
def cal_phot(cal,header,targ_info=None,targname=None,fwhm_flag=8.0,
             aspect_flag=0.7,north_flag=10.0,snr_flag=5.0,
             airmass_flag=2.0,bkg_level=15.0,saturation=30000.0,
             normal_pflag = 0.01):
    '''
    Description
    -----------
    Provide aperture photometry for all the targets specified in the targ_info dictionary
    on a calibrated image (header must be provided).

    Inputs
    ------
    cal       = (float) calibrated image (must be counts NOT counts/sec)
    header    = (string) header for calibrated image with astrometry
    targ_info = (optional) dictionary of targets in field
    fwhm_flag = (float) maximum FWHM in arcseconds without flag

    Outputs
    -------


    Example
    -------
    '''
    

    # Get targ_info if not supplied
    if targ_info is None:
        targ_info = target_info()

    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))
        
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
    
    pa = []
    
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

        if char['airmass'] > airmass_flag:
            sflags.append('airmass')

        flux = []
        count = 0
        for aprad in apertures:
            phot = aperture_phot(cal,header,position=position,skyrad=targ_info['skyrad'][i],
                                 aprad=aprad)
            flux = np.append(flux,phot['flux'])
            if count == 0:
                if phot['background'] > bkg_level:
                    sflags.append('bkg level')
                if phot['normal_pval'] < normal_pflag:
                    sflags.append('bkg dist')
                if phot['max counts'] > saturation:
                    sflags.append('saturation')
                normal_pval = phot['normal_pval']
                background = phot['background']
                background_rms = phot['background_rms']
            count += 1
        try:
            phot_dict = {'aperture':apertures,'flux':flux,'background':background,
                         'background_rms':background_rms,'fwhm':char['fwhm'][i],'snr':char['snr'][i],
                         'airmass':char['airmass'],'exptime':header['exptime'],'pa':char['pa'],
                         'flags':sflags,'bjd':char['bjd'], 'filter':band, 'aspect':char['aspect'][i],
                         'normal_pval':normal_pval}
        except:
            pdb.set_trace()
        final_phot[names[i]] = phot_dict
        
    final_phot['bjd'] = char['bjd']
    final_phot['airmass'] = char['airmass']
    final_phot['band'] = band

    return final_phot


######################################################################
# Perform aperture photometry on all images on a specified night
def night_phot(date,targ_info=None,write=True,clobber=False,targname=None,prefix=None):
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

    # Read photometry file if it is there
    final_out = read_photometry(targname,date)
    if final_out is not None and not clobber:
        return final_out

    # Get targ_info if not supplied
    if targ_info is None:
        targ_info = target_info(targname)

    # Generate list of dictionary keys
    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))

    # Path and observation information
    dpath,opath,archive,execpath = get_paths(targname)
    info = get_obs(date,targname)
    
    # Load all files (all bands) for Supernova)
    allfiles,fct =   tp.get_files(d=dpath+date+'/',prefix=prefix,
                                  tag='_solved',suffix='fits')

    # Determine which filters were used
    filters = []
    for fname in allfiles:
        header = fits.getheader(fname)
        filters.append(header['filter'].replace("'",""))
    filters = np.unique(filters)

    fstr = ''
    for val in filters:
        fstr = fstr+val+', '

    print 'Filters used on '+date+': '+fstr[:-2]
    print ''

    # Loop through all bands observed on that night
    final_out = {}
    for band in filters:
        print 'Starting batch photometry for '+band+' band on '+date+'...'
        bias,dark,flat = get_cal_frames(date,setting=info['setting'],targname=targname,\
                                        readnoise=False,band=band,flatten=False)
        sfiles,sct =   tp.get_files(d=dpath+date+'/',prefix=prefix,
                                    tag=band+'_solved',suffix='fits')

        # Loop through each file in specified band
        i = 0
        outdict = {}
        pbar = tqdm(desc = 'Analyzing '+band+' band', total = sct, unit = 'file')
        for sfile in sfiles:
            try:
                im,header = read_image(sfile,plot=False)
                texp = header['EXPTIME']
                cal = (im-bias-dark*texp)/flat
                out = cal_phot(cal,header,targ_info=targ_info,targname=targname)
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

        # Update final_out dictionary according to filter band
        final_out[band] = outdict

    # Write out photometry into output directory
    if write:
        print 'Writing out photometry files into: '
        print '    '+opath+date+'/'
        fname = opath+date+'/photometry_'+date+'.pck'
        fout = open(fname,"w")
        pickle.dump(final_out,fout)

    return final_out




######################################################################
# Do photometry for all files in archive
def do_all_phot(clobber=True,targname=None,prefix=None):
    '''
    Description
    -----------
    Loop through all dates with "use" option = 'Y' and perform photometry

    Inputs
    ------
    clobber = (boolean) overwrite photometry if it already exists

    Outputs
    -------
    No output

    Example
    -------
    
    '''

    obs = get_summary(targname)

    dates = obs['date'][obs['use'] == 'Y'].values
    for date in dates:
        date = str(date)
        try:
            phot = night_phot(date,clobber=clobber,targname=targname,prefix=prefix)
        except:
            print 'Photometry failed for '+str(date)
            pass

    return



######################################################################
# Has photometry been done for all nights?
def check_phot_status(targname,ret_missing=False,ret_done=False):
    '''
    Description
    -----------
    Check all the nights of observing that are useable against how many have 
    photometry pickle files (i.e., are done)

    Inputs
    ------

    Outputs
    -------
    List of dates that need photometry performed.

    Example
    -------
    
    '''

    dpath,opath,archive,execpath = get_paths(targname)

    obs = get_summary(targname)

    dates = obs['date'][obs['use'] == 'Y'].values
    ndates = np.float(len(dates))    

    misslist = []
    donelist = []
    for i in range(len(dates)):
        date = dates[i]
        photfile = opath+str(date)+'/photometry_'+str(date)+'.pck'
        exist = glob.glob(photfile)
        if len(exist) == 0:
            misslist.append(date)
        if len(exist) == 1:
            donelist.append(date)

    print('Missing %i of %i nights of photometry'%(len(misslist),ndates))
    percent = (1-np.float(len(misslist))/ndates)*100
    print('Photometry is %.2f%% complete'%percent)

    if ret_missing:
        return misslist
    elif ret_done:
        return donelist
    else:
        return

   
######################################################################
# Read photometry dictionary for a specified night
def read_photometry(targname,date):
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

    Example
    -------
    '''

    dpath,opath,archive,execpath = get_paths(targname)
    photfile = opath+date+'/photometry_'+date+'.pck'
    exist = glob.glob(photfile)

    if len(exist) == 1:
        phot = pickle.load( open( photfile, "rb" ) )
    else:
        phot = None

    return phot

######################################################################

def find_mzp(mzps_relevant,b,d,targname,targs,phot):
    
    targs.remove(targname)
    targs.remove('bjd')
    targs.remove('filename')
    targs.remove('airmass')

    mzp_refs = []
    
    for r in targs:
        
        mzps_each_obs = []
        
        targphot = phot[b][r]
        
        num_obs = len(targphot)
        
        for i in range(num_obs):
            flux = targphot[i]['flux'][0]
        
            mzps_each_obs.append( mzps_relevant[int(r[-1])] + (2.5*np.log10(flux)))
            
        mzp_refs.append(np.median(mzps_each_obs))
        
    mzp_night = np.median(mzp_refs)
    
    mzps_relevant[0] = mzp_night
    
    return mzps_relevant

    
######################################################################
def make_plots(targname,band=None):
    
    '''
    Description
    -----------

    Inputs
    ------
    b = band

    Outputs
    -------

    Example
    -------   
    
    '''
    if band is None:
        print('Must specify a band!')
        return

    dates,dates_counts = get_dates(targname,targname,counts=True)
    
    targinfo = target_info(targname)
        
    mzps_relevant = np.nan_to_num(targinfo[band+'mag'])
    pbar = tqdm(desc='Calculating ', total=len(dates), unit=' nights')

    jds = None
    for d in dates:
                
        try:
            
            phot = read_photometry(targname,d)
            
            targs = phot[band].keys()

            mzps_relevant = find_mzp(mzps_relevant,band,d,targname,targs,phot)
            targs = phot[band].keys()
            
            targs.remove('filename')
            targs.remove('airmass')
            targs.remove('bjd')

            if jds is None:
                jds = {targ: [] for targ in targs}
                fluxes = {targ: [] for targ in targs}
                flux_error = {targ: [] for targ in targs}
                mags = {targ: [] for targ in targs}

            for r in targs:
                
                targphot = phot[band][r]
                
                num_obs = len(targphot)
                    
                night_fluxes = []
                night_jd = []
                night_mags = []
                                    
                for n in range(num_obs):
                        
                    #could update so is the mzp of each image instead of night
                    flux = targphot[n]['flux'][0]
                        
                    if r == targname:
                        mag = -2.5*np.log10(flux) + mzps_relevant[0]
                    else:
                        mag = -2.5*np.log10(flux) + mzps_relevant[int(r[-1])]
                    if band == 'z' and np.isnan(mag):
                        pdb.set_trace()
                    jd = targphot[n]['bjd']
                    flags = targphot[n]['flags']   
                        
                    if 'north' in flags or 'snr' in flags \
                    or 'pa' in flags or 'aspect' in flags \
                    or 'fwhm' in flags or flux < 0:
                        pass
                        
                    night_fluxes = np.append(night_fluxes,flux)
                    night_mags = np.append(night_mags,mag)
                    night_jd = np.append(night_jd,jd)
                    night_err = np.std(night_fluxes)
                                              
                fluxes[r] = np.append(fluxes[r],np.median(night_fluxes))
                mags[r] = np.append(mags[r],np.median(night_mags))
                jds[r] = np.append(jds[r],np.median(night_jd))
                flux_error[r] = np.append(flux_error[r],night_err)
                                        
        except:
            print 'Photometry failed for '+str(d)
            pass
        
        pbar.update(1)  
    pbar.close()
    mag_error = [(1.086*flux_error[i])/fluxes[i] for i in targs]

    c = {targname:'k','ref1':'r','ref3':'m',\
           'ref2':'y','ref5':'c','ref4':'pink'}
    
    return fluxes,mags,jds,flux_error, mag_error,c, targs, band


def check_refs(fluxes,mags,jds,flux_error,c,targs,b):
    
    targs.remove(targname)
    targs.remove('bjd')
    targs.remove('filename')
    targs.remove('airmass')
                        
    plt.figure(1)
    plt.ion()
    plt.clf()
    o = 0
    for r in targs:  
        lc = fluxes[r] / np.sum([fluxes[i] for i in targs if i != r],axis=0)
        lc /= np.median(lc)
        dates_full = Time(jds[r],format='jd').datetime
        dates= np.array([dates_full[i].strftime('%m/%d/%Y') for i in range(len(dates_full))])                    
        plt.scatter(dates,(o*.1)+lc,label=r,c=c[r],s=20) 
        plt.axhline(y=(o*.1)+np.median(lc),ls='--',c=c[r])
        o += 1
    plt.title('Normalized reference star flux in '+str(b)+' band')
    plt.xlabel('Date')
    plt.xticks(rotation=-60,fontsize=7)
    plt.ylabel('Normalized Flux')
    plt.legend(bbox_to_anchor=(.97,.97),loc=2,fontsize=6)
    plt.show()
        
    plt.figure(2)
    plt.clf()
    o = 0
    for r in targs:  
        try:
            plt.errorbar(jds[r],(o*.1)+fluxes[r],label=r,fmt='o',c=c[r],yerr=flux_error[r]) 
            plt.axhline(y=(o*.1)+np.median(fluxes[r]),ls='--',c=c[r])
            o += 1
        except:
            pass  
        else:
            pass
    plt.title('Raw reference star flux in '+str(b)+' band')
    plt.xlabel('JD')
    plt.ylabel('Flux')
    plt.legend(bbox_to_anchor=(.97,.97),loc=2,fontsize=6)
        
    return jds,fluxes,flux_error

    
def check_seeing(b):
    dates,dates_counts = get_dates(targname,targname,counts=True)
        
    fwhm = []
    exptime = []
    pbar = tqdm(desc='Calculating ', total=len(dates), unit=' nights')
    for d in dates:
        try:
            phot = read_photometry(targname,d)
                                        
            snphot = phot[b][targname]
                
            num_obs = len(snphot)

            for n in range(num_obs):
                fwhm = np.append(fwhm,snphot[n]['fwhm'])
                exptime = np.append(exptime,snphot[n]['exptime'])
                    
        except:
            pass
                    
        pbar.update(1)
                    
    plt.figure(10)
    plt.clf()  
    plt.ion()
    plt.hist(fwhm[~np.isnan(fwhm)],bins=150)
    plt.axvline(np.median(fwhm[~np.isnan(fwhm)]),c='k',ls='--',label=str(np.round(np.median(fwhm[~np.isnan(fwhm)]),decimals=2))+' +/- '+str(np.round(np.std(fwhm[~np.isnan(fwhm)],ddof=1),decimals=2)))
    plt.xlabel('FWHM')
    plt.ylabel('Frequency')
    plt.title('Histogram of Seeing')
    plt.legend()
    plt.xlim(0,10)
        
    return fwhm

def convert_jd_date(j):
    dates = []
    
    for i in range(len(j[targname])):
        t = Time(j[targname][i],format='jd')
        dates = np.append(dates,t.datetime)
        
    return dates
    

def make_master_plot(compare_Foley=False,targname=targname,maxerr=1,save=True):
    
    
    dpath,opath,archive,execpath = get_paths(targname)
    gf,gm, gj,gfe,gme,gc,gtargs,gb = make_plots(targname,'g')
    rf,rm, rj,rfe,rme,rc,rtargs,rb = make_plots(targname,'r')
    i_f,im, ij,ife,ime,ic,itargs,ib = make_plots(targname,'i')
    zf,zm, zj,zfe,zme,zc,ztargs,zb = make_plots(targname,'z')
    
    gd = convert_jd_date(gj)
    rd = convert_jd_date(rj)
    i_d = convert_jd_date(ij)
    zd = convert_jd_date(zj)
    #vd = convert_jd_date(vj)
    
    plt.figure(16,figsize=(6,6))
    plt.clf()
    plt.ion()

    inds, = np.where((gme[0] > 0)& (gme[0]<maxerr))
    plt.errorbar(gd[inds],gm[targname][inds],c='yellowgreen',fmt='o',\
                     yerr=gme[0][inds],label='g')

    inds, = np.where((rme[0] > 0)& (rme[0]<maxerr))
    plt.errorbar(rd[inds],rm[targname][inds]+1.5,c='orange',fmt='o',\
                     yerr=rme[0][inds],label='r')
    
    inds, = np.where((ime[0] > 0)& (ime[0]<maxerr))
    plt.errorbar(i_d[inds],im[targname][inds]+3,c='red',fmt='o',\
                     yerr=ime[0][inds],label='i')

    inds, = np.where((zme[0] > 0)& (zme[0]<maxerr))
    plt.errorbar(zd[inds],zm[targname][inds]+4.5,c='maroon',fmt='o',\
                     yerr=zme[0][inds],label='z')
    #plt.errorbar(vd,vm['SN2017fgc'],c='yellow',fmt='o',\
                     #yerr=vme,label='v')
                     
    if compare_Foley == True:
            
        foleydata = np.loadtxt(execpath+'sn2017fgc.phot',dtype='string',delimiter=' ')
        times = []
        bands = []
        mags = []
        uncertainty= []
        for i in range(len(foleydata)):
            times = np.append(times,foleydata[i][0])
            bands = np.append(bands,foleydata[i][1])
            mags = np.append(mags,foleydata[i][2])
            uncertainty = np.append(uncertainty,foleydata[i][3])
    
        times = times.astype('float')
        mags = mags.astype('float')
        uncertainty = uncertainty.astype('float')
        
        ginds = np.where(bands == 'g')
        gtimes_full = Time(times[ginds],format='jd').datetime
        gmags = mags[ginds]
        guncert = uncertainty[ginds]
        plt.errorbar(gtimes_full,gmags,yerr=guncert,fmt='o',c='k',label='Swope g')

        iinds = np.where(bands == 'i')
        itimes_full =  Time(times[iinds],format='jd').datetime
        imags = mags[iinds]
        iuncert = uncertainty[iinds]  
        plt.errorbar(itimes_full,imags+1.5,yerr=iuncert,fmt='o',c='m',label='Swope i')
    
        rinds = np.where(bands == 'r')
        rtimes_full = Time(times[rinds],format='jd').datetime
        rmags = mags[rinds]
        runcert = uncertainty[rinds]
        plt.errorbar(rtimes_full,rmags+3,yerr=runcert,fmt='o',c='blue',label='Swope r')
        
        
        #######
        
        rnan_index = np.argwhere(np.isnan(rm['SN2017fgc']))
        rjnew = np.delete(rj['SN2017fgc'],rnan_index)
        rmnew = np.delete(rm['SN2017fgc'],rnan_index)
        r_dates = np.linspace(np.min(rjnew)-2,np.max(rjnew)+2,10000)
        rcs = CubicSpline(rjnew,rmnew)
        rindex = [np.where(np.round(r_dates) == np.round(times[rinds][i])) for i in range(len(times[rinds]))]
        rresids = []
        for z in range(len(times[rinds])):
            rresids = np.append(rresids,np.median(rcs(r_dates)[rindex[z]])-rmags[z])
            
        gnan_index = np.argwhere(np.isnan(gm['SN2017fgc']))
        gjnew = np.delete(gj['SN2017fgc'],gnan_index)
        gmnew = np.delete(gm['SN2017fgc'],gnan_index)
        g_dates = np.linspace(np.min(gjnew)-2,np.max(gjnew)+2,10000)
        gcs = CubicSpline(gjnew,gmnew)
        gindex = [np.where(np.round(g_dates) == np.round(times[ginds][i])) for i in range(len(times[ginds]))]
        gresids = []
        for z in range(len(times[ginds])):
            gresids = np.append(gresids,np.median(gcs(g_dates)[gindex[z]])-gmags[z])
            
        inan_index = np.argwhere(np.isnan(im['SN2017fgc']))
        ijnew = np.delete(ij['SN2017fgc'],inan_index)
        imnew = np.delete(im['SN2017fgc'],inan_index)
        i_dates = np.linspace(np.min(ijnew)-2,np.max(ijnew)+2,10000)
        ics = CubicSpline(ijnew,imnew)
        iindex = [np.where(np.round(i_dates) == np.round(times[iinds][i])) for i in range(len(times[iinds]))]
        iresids = []
        for z in range(len(times[iinds])):
            iresids = np.append(iresids,np.median(ics(i_dates)[iindex[z]])-imags[z])
            
        plt.figure(25)
        plt.clf()
        plt.ion()
        plt.errorbar(times[rinds],rresids,yerr=np.std(rresids,ddof=1),fmt='o',ls='None',label='r')
        plt.errorbar(times[ginds],gresids,yerr=np.std(gresids,ddof=1),fmt='o',ls='None',label='g')
        plt.errorbar(times[iinds],iresids,yerr=np.std(iresids,ddof=1),fmt='o',ls='None',label='i')
        plt.axhline(y=0,ls='--',c='k')
        plt.xlabel('Date (JD)')
        plt.ylabel('Residual')
        plt.title('Residuals of Cubic Splice of Thacher Data and Swopes')
        plt.legend()
        plt.ylim(-0.5,0.5)
        

    plt.ylim(22,11)
    #plt.xlabel('Date',fontsize=14)
    plt.xticks(rotation=-45,fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylabel('Magnitude',fontsize=14)
    plt.title(targname)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(opath+targname+'_multiband_LC.png',dpi=300)

    pdb.set_trace()
    return
    
def reduce_spectra():
    
    dpath,opath,archive,execpath = get_paths(targname)
    file = execpath + 'lcogtdata-20180213-2/LCOEPO2018A-014_0001410120_fts_20180205_58155/nttSN2018gv_fts_20180205_merge_1.6_58155_1_2df_ex.fits'
    
    spec,raw,bg,noise = fits.getdata(file)
    header  = fits.getheader(file)    

    w = header['XMIN']+(header['XMAX']-header['XMIN'])/(header['NAXIS1']-1)*np.arange(header['NAXIS1'])    #w=XMIN+(XMAX-XMIN)/(NAXIS1-1)*np.arange(NAXIS1)
    
    plt.figure(33)
    plt.ion()
    plt.clf()
    plt.plot(w,spec[0])
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel('ergs$\,\,$cm$^{-2}\,$s$^{-1}\,\AA^{-1}$  $10^{20}$')
    plt.title('Spectrum of SN2018gv')
    
    return
    
    
    
    
    
    
    
    
    
    
