import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import signal
from tqdm import tqdm,trange
import pandas as pd

from FITS_tools.hcongrid import hcongrid

import phot_pipe as pp
from quick_image import read_image, display_image

from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from photutils import DAOStarFinder
from astropy.io import fits
from astropy.io.ascii import SExtractor
from astropy import wcs
from astropy.coordinates import SkyCoord, Angle, EarthLocation
from astropy import units as u
from astropy.time import Time
from astropy.stats import mad_std

from datetime import date

import os,shutil,glob,pdb,pickle,fnmatch

# TODO
# write out targ_info so that it can be reused
# delete param_c files, temp files and obj_out files


def count_galaxies(obstype='SNS',plot=True):

    paths = pp.get_paths(obstype=obstype)

    dates = pp.get_dates(obstype=obstype)

    # Find all SNS files
    allfiles,act = pp.get_files(prefix=obstype+'_')

    # Return only unique galaxy names
    galname = []
    for f in allfiles:
        galname = np.append(galname,f.split('/')[-1].split('_')[1].split('-')[0])
    galname_sort = np.sort(galname)
    galname_unique = np.unique(galname_sort)

    # Loop through and count how many observations for each unique galaxy name
    nobs = np.zeros(len(galname_unique))
    pbar = tqdm(desc = 'Checking galaxy observations', total = len(galname_unique), unit = 'gals')
    time_between = []
    for i in range(len(galname_unique)):
        files,fct = pp.get_files(prefix=obstype+'_'+galname_unique[i])
        if type(fct) is int:
            nobs[i] = fct
        dates = pp.get_dates(targname=galname_unique[i],obstype=obstype)
        for i in np.arange(len(dates)-1)+1:
            recent = dates[i]
            past = dates[i-1]
            d1 = date(int(past[0:4]),int(past[4:6]),int(past[6:]))
            d2 = date(int(recent[0:4]),int(recent[4:6]),int(recent[6:]))
            delta = d2-d1
            t = delta.days
            np.append(time_between,t)
        pbar.update(1)
    pbar.close()
    time_between.sort()
    medtime = np.median(time_between)

    # Make a plot and histogram of the number of observations
    if plot:
        plt.ion()
        plt.figure(1)
        plt.clf()
        bins = np.arange(0,np.max(nobs)+1,1)
        histdata = plt.hist(nobs,bins=bins)
        plt.xlabel('Number of Observations')
        plt.ylabel('Number of Galaxies')
        plt.title('SNS Search Status')
        medobs = np.int(np.median(nobs))
        totobs = np.int(act)
        multiobs = np.int(np.sum(histdata[0][2:]))
        plt.annotate('Total number of observations = '+str(totobs),xy=[0.85,0.8],
                     xycoords='figure fraction',horizontalalignment='right')
        plt.annotate('Total number of galaxies with > 1 observation = '+str(multiobs),xy=[0.85,0.75],
                     xycoords='figure fraction',horizontalalignment='right')
        plt.annotate('Median number of observations per galaxy = '+str(medobs),xy=[0.85,0.7],
                     xycoords='figure fraction',horizontalalignment='right')
        ##plt.annotate('Median time between images for each target = '+str(medtime),xy=[0.85,0.65],
                     ##xycoords='figure fraction',horizontalalignment='right')
        plt.savefig(paths['output']+'SNS_Status.png',dpi=300)

        info = {'name':galname_unique, 'count':len(galname_unique)}
    return



def all_targets_observed(prefix='SNS_'):
    '''
    Return a unique list of targets that have been observed in a particular
    program

    NOTE: this only works for targets that have been observed under a
    program such that the file names start with an obstype tag followed by
    an underscore followed by a target name followed by a dash
    '''

    allfiles,act = pp.get_files(prefix=prefix)
    galname = []
    for f in allfiles:
        if '_' in prefix:
            galname = np.append(galname,f.split('/')[-1].split('_')[1].split('-')[0])
        else:
            galname = np.append(galname,f.split('/')[-1].split('-')[0])
    galname_sort = np.sort(galname)
    galname_unique = np.unique(galname_sort)
    return galname_unique



######################################################################
# Find all galaxies that have template images
def gals_with_template_images():
    '''
    Return galaxy names for which there are template images in the
    archive
    '''

    gals = all_targets_observed()
    tempgals = []
    pbar = tqdm(desc = 'Finding template images', total = len(gals), unit = 'galaxy')
    for g in gals:
        template_file = glob.glob('/home/student/SNS/reference_images/*'+g+'*')
        tct = len(template_file)
        if tct >= 1:
            tempgals = np.append(tempgals,g)
        pbar.update(1)
    pbar.close()

    return tempgals


######################################################################
def gal_positions():
    gals_obs = all_targets_observed()
    radeg_obs = [] ; decdeg_obs = []
    for gal in tqdm(gals_obs):
        files,fct = pp.get_files(prefix='SNS_'+gal)
        if fct > 0:
            h = fits.getheader(files[0])
            radeg_obs = np.append(radeg_obs,Angle(h['OBJCTRA'],unit=u.hour).degree)
            decdeg_obs = np.append(decdeg_obs,Angle(h['OBJCTDEC'],unit=u.deg).degree)

    alldata = pd.read_csv('/home/student/SNS/Pointings.csv')
    radeg = [] ; decdeg = []
    for index, row in alldata.iterrows():
        radeg = np.append(radeg,row['RA'])
        decdeg = np.append(decdeg,row['Dec'])

    ra = Angle(radeg*u.degree)
    ra = ra.wrap_at(180*u.degree)
    dec = Angle(decdeg*u.degree)
    raobs = Angle(radeg_obs*u.degree)
    raobs = raobs.wrap_at(180*u.degree)
    decobs = Angle(decdeg_obs*u.degree)
    fig = plt.figure(figsize=(8,6))
    plt.clf()
    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(ra.radian, dec.radian, color='blue', alpha=0.5,label='Target Field')
    ax.scatter(raobs.radian, decobs.radian, color='red', alpha=1, label='Currently Monitoring')
    ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
    ax.grid(True)
    plt.legend(loc='lower center')
    plt.title('Thacher Obsevatory Supernova Search')
    plt.savefig('/home/student/SNS/Galaxy_Obs_AllSky.png',dpi=300)

    return


######################################################################
# Quick reduction of a supernova to get magnitude
def get_dophot_mag(image=None,header=None,targ_info=None,cals=None,filename=None,
                   obstype='SNe',targname=None,date=None,band=None,
                   snrmin=3,write=False,plot=False,printout=False,
                   sigma_reject=True,matchtol='high'):

    # Check inputs
    if image is None or header is None:
        if filename is None:
            print('No image specified!')
            return None, None
        image,header = read_image(filename)

    if targ_info is None:
        print('No target information!')
        return None, None
    else:
        ra = targ_info['RAdeg'][0]
        dec = targ_info['DECdeg'][0]
        rastr,decstr = pp.deg_to_hmsdms(ra,dec,alwayssign=False)

    if targname is None:
        try:
            targnmame = header['OBJECT']
            print('Using target name from header: '+targname)
        except:
            print('No target name supplied or available')
            return None, None

    if date is None:
        try:
            date = header['DATE-OBS'].split('T')[0].replace('-','')
        except:
            print('No target name supplied or available')
            return None, None

    if band is None:
        try:
            band = header['FILTER']
            print('Using filter band from header: '+band)
        except:
            print('No photometric band supplied or available')
            return None, None

    # Get paths
    paths = pp.get_paths(obstype=obstype,targname=targname)


    if cals is None:
        # Get calibration frames
        cals = pp.get_cal_frames(date=date,band=band,targname=targname,obstype=obstype)

    # Calibrate image
    cal = pp.calibrate_image(image,header,cals,rotated_flat=False)

    # Make a plot if specified
    if plot:
        pp.display_targets(cal,header,targ_info,targname=targname,obstype=obstype,
                           write=write,outfile=targname+'_dophot.png')

    # Get some information about image for DoPHOT
    iminfo = pp.do_sextractor(cal,header,detect_thresh=5.0,analysis_thresh=5.0,
                              minpix=3,outdir=paths['output'])
    fwhm = np.median(iminfo['FWHM_IMAGE'])
    fwhm_as = np.median(iminfo['FWHM_WORLD'])*3600.
    skylev = np.median(iminfo['BACKGROUND'])

    # Delete temporary file if it exists
    outfile = paths['output']+targname+'_dophot.fits'
    try:
        os.system('rm '+outfile)
    except:
        pass
    # Write out file so that it is accessible to DoPHOT
    fits.writeto(outfile,cal,header)

    # Make DoPHOT parameter file
    pp.make_dophot_param_file(image_in=outfile,outname=targname,outdir=paths['output'],
                              fwhm_pix=fwhm,skylev=skylev,snrmin=snrmin,clobber=True)

    # DoPHOT command on bellerophon
    cdophot = '/home/student/photpipe/Cfiles/bin/linux/cdophot'
    # Do DoPHOT
    pwd = os.getcwd()
    os.chdir(paths['output'])
    os.system(cdophot+' param_default_c')
    # Read DoPHOT output file
    try:
        obj_out = pp.read_dophot_output('obj_out_'+targname)
        os.system('rm obj_out_'+targname)
    except:
        return np.nan,np.nan
    os.chdir(pwd)

    # How tightly do we constrain the matches
    if matchtol=='high':
        maxsep = 1.0
    elif matchtol == 'med':
        maxsep = fwhm_as/2.0
    else:
        maxsep = fwhm_as
        
    print('Maximum separation allowed for match = %.2f arcseconds'%maxsep)
    
    # Match sources
    DPra,DPdec = pp.xy_to_radec(obj_out['xpos'].values,obj_out['ypos'].values,header)
    w = wcs.WCS(header)
    DPcoords = SkyCoord(DPra,DPdec,unit=(u.deg,u.deg))
    mag = []
    dmag = []
    for i in range(len(targ_info['RAdeg'])):
        testcoords = SkyCoord(targ_info['RAdeg'][i],targ_info['DECdeg'][i],unit=(u.deg,u.deg))
        sep = testcoords.separation(DPcoords)
        minsep = np.min(sep.arcsec)
        if minsep < maxsep:
            index = np.argmin(sep)
            if i == 0:
                if obj_out['dflux'][index]/obj_out['flux'][index] < 0.1:
                    targflux = obj_out['flux'][index]
                    targflux_err = obj_out['dflux'][index]
                else:
                    targflux = np.nan
                    targflux_err = np.nan
            else:
                if obj_out['dflux'][index]/obj_out['flux'][index] < 0.05:
                    mag.append(targ_info[band+'mag'][i] - 2.5*np.log10(targflux/obj_out['flux'][index]))
        elif i == 0:
            print('Target not found!')
            targflux = np.nan
            targflux_err = np.nan
        else:
            print('No DoPHOT match for reference star '+str(i)+'!')

    if len(mag) == 0:
        return np.nan, np.nan

    # Reject outliers based on std of measurements
    if sigma_reject:
        rej = pp.sigmaRejection(mag)
    else:
        rej = np.array(mag)

    # Final magnitude and error
    finalmag = np.mean(rej)
    if len(rej) > 1:
        finalmag_err = np.std(rej,ddof=1)
    else:
        finalmag_err = np.nan

    # Print results if requested
    if printout:
        print(band+'-band magnitude of '+targname+' = %.3f +/- %0.3f'%(finalmag,finalmag_err))

    return finalmag,finalmag_err



#----------------------------------------------------------------------
# Get references for doPHOT reduction of SN targets
#----------------------------------------------------------------------
def get_SN_targs(targname=None,obstype='SNe',ra=None,dec=None,band='r',
                 write=True,maxMag=15.5,minMag=13,maxStd=0.01,maxErr=0.02,sighi=20,siglo=3,
                 fwhm_max=4.0,skymax=2000.0,dpix_max=100,clobber=False,stackclobber=False):

    if type(ra) == str:
        rastr = ra
    if type(dec) == str:
        decstr = dec

    paths = pp.get_paths(obstype=obstype,targname=targname)
    dates = pp.get_dates(targname)

    outpath = paths['output']
    targpickle = outpath+targname+'_targinfo.pck'
    test = glob.glob(targpickle)

    if len(test) == 1:
        if not clobber:
            print('Reading target information file...')
            targ_info = pickle.load( open( targpickle, "rb" ) )
            return targ_info
        elif clobber:
            os.system('rm '+targpickle)


    files,fct = pp.get_files(prefix=targname,tag=band+'_')

    temp_im,th = pp.all_stack(files,outpath=paths['output'],outname=targname+'_'+band+'_stack',
                              obstype=obstype,targname=targname,calwrite=True,clobber=stackclobber,
                              fwhm_max=fwhm_max,skymax=skymax,dpix_max=dpix_max)

    refs = pp.get_panstarrs_refs(ra=rastr,dec=decstr,maxMag=maxMag,minMag=minMag,maxStd=maxStd,maxErr=maxErr)
    
    targ_info = pp.make_targ_info(refs,ra=rastr,dec=decstr)
    
    pp.display_targets(temp_im,th,targ_info,targname=targname,obstype=obstype,
                       write=write,outfile=targname+'_dophot.png',
                       siglo=siglo,sighi=sighi,fignum=99)

    if write:
        outfile = open(targpickle,"w")
        pickle.dump(targ_info,outfile)

    return targ_info



######################################################################
# For followup targets
######################################################################

#obstype= 'SNe'
#targname = 'SN2019set'
#target_ra = '02:25:16.21'
#target_dec = '24:16:00.1'
#coords = SkyCoord(target_ra,target_dec,unit=(u.hour,u.degree))


#targname = 'SN2019ein'
#rastr = '13:53:29.110'
#decstr = '40:16:31.33'

#targname = 'SN2019ehk'
#rastr = '12:22:56.15'
#decstr = '15:49:34.03'
#coords = SkyCoord(rastr,decstr,unit=(u.hour,u.degree))

#targname = 'SN2019sox'
#rastr = '21:31:24.720'
#decstr = '02:29:39.05'


def SN_dophot_night(targ_info,targname=None,obstype='SNe',date=None,clobber=False,band=None,matchtol='high'):


    paths = pp.get_paths(targname=targname,obstype=obstype)

    outpath = paths['output']+date+'/'
    nightpickle = outpath+'/'+targname+'_'+band+'_photometry.pck'
    test = glob.glob(nightpickle)

    if len(test) == 1:
        if not clobber:
            nightphot = pickle.load(open(nightpickle,'rb'))
            return nightphot
        elif clobber:
            os.system('rm '+nightpickle)
    mag = [] ; magerr = [] ; bjd = []

    files,fct = pp.get_files(prefix=targname,tag=band+'_',date=date)
    if fct > 0:
        # This line creates proper output directory as well as gets cal frames
        cals = pp.get_cal_frames(targname=targname,obstype=obstype,date=date)

        for f in files:
            image,header = read_image(f)
            m,me = get_dophot_mag(image=image,header=header,targ_info=targ_info,
                                  cals=cals,obstype=obstype,targname=targname,
                                  date=date,band=band,
                                  write=False,plot=False,printout=False,
                                  sigma_reject=True,matchtol=matchtol)
            mag.append(m)
            magerr.append(me)
            bjd.append(header['BJD-OBS']+header['EXPTIME']/(2.0*86400.0))
        good = np.isfinite(mag) & np.isfinite(magerr)

        if np.sum(good) >= 1:
            mag = np.array(mag)[good]
            magerr = np.array(magerr)[good]
            bjd = np.array(bjd)[good]
        else:
            print('No good photometry in '+band+' band on '+date)
            mag = [] ; magerr = [] ; bjd = []

    else:
        print('No photometry in '+band+' band on '+date)

    nightphot = {'mag':mag,'mag_err':magerr,'bjd':bjd}

    try:
        outfile = open(nightpickle,"w")
        pickle.dump(nightphot,outfile)
    except:
        pass

    return nightphot



def SN_dophot_batch(targ_info,targname=None,obstype='SNe',clobber=False,nightclobber=False,
                    error_calc='std',matchtol='high'):

    paths = pp.get_paths(targname=targname,obstype=obstype)
    outpath = paths['output']

    pickleout = outpath+targname+'_photometry.pck'
    test = glob.glob(pickleout)
    if len(test) == 1:
        if not clobber:
            phot = pickle.load(open(pickleout,'rb'))
            return phot
        else:
            os.system('rm '+pickleout)

    # Load all files (all bands)
    allfiles,fct = pp.get_files(prefix=targname)

    # Determine which filters were used
    filters = []
    for fname in allfiles:
        header = fits.getheader(fname)
        filters.append(header['filter'].replace("'",""))
    filters = np.unique(filters)

    phot = {}

    for band in filters:
        dates = pp.get_dates(targname=targname)
        bjdall = [] ; magnitude = [] ; magnitude_err = []

        try:
            for date in dates:

                nightpickle = outpath+date+'/'+targname+'_'+band+'_photometry.pck'
                test = glob.glob(nightpickle)

                if len(test) == 1 and not nightclobber:
                    print('Loading photometry data from '+nightpickle)
                    nightphot = pickle.load(open(nightpickle,'rb'))
                else:
                    nightphot = SN_dophot_night(targ_info,targname=targname,obstype=obstype,
                                                date=date,band=band,clobber=True,matchtol=matchtol)

                mag = nightphot['mag']
                magerr = nightphot['mag_err']
                bjd = nightphot['bjd']

                if len(mag) >= 1:
                    avg,sw = np.average(mag,weights=1./magerr**2,returned=True)
                    magnitude = np.append(magnitude,avg)
                    if error_calc == 'std':
                        magnitude_err = np.append(magnitude_err,np.std(mag))
                    elif error_calc == 'conservative':
                        magnitude_err = np.append(magnitude_err,np.max(magerr))
                    elif error_calc == 'stat':
                        magnitude_err = np.append(magnitude_err,1./np.sqrt(sw))
                    bjdavg = np.average(bjd,weights=1./magerr**2,returned=False)
                    bjdall = np.append(bjdall,bjdavg)
                else:
                    print('No photometry in '+band+' band on '+date)
        except:
            print('Something went wrong with photometry in '+band+' band on '+date)

        phot[band] = {'mag':magnitude, 'mag_err':magnitude_err,'bjd':bjdall}

    pickleout = outpath+targname+'_photometry.pck'
    outfile = open(pickleout,"w")
    pickle.dump(phot,outfile)

    return phot



#----------------------------------------------------------------------
# Make full stack of target in all bands
#----------------------------------------------------------------------
def color_images(targname,obstype='SNe',fwhm_max=5.0,skymax=5000.0,dpix_max=200,
                 write=True,sighi=20,siglo=3,clobber=True,
                 fignum=23):

    paths = pp.get_paths(obstype=obstype,targname=targname)
    dates = pp.get_dates(targname)

    allfiles,act = pp.get_files(prefix=targname)

    # Get all unique bands observed from file names
    bands = np.unique(np.array([a.split('_')[-2].split('-')[-1] for a in allfiles]))

    for band in bands:
        files,fct =  pp.get_files(prefix=targname,tag=band+'_')
        if fct >=1:
            stack,sh = pp.all_stack(files,outpath=paths['output'],outname=targname+'_'+band+'_latest_stack',
                                      obstype=obstype,targname=targname,clobber=clobber,
                                      fwhm_max=fwhm_max,skymax=skymax,dpix_max=dpix_max)


def convert_layers(obstype='SNe',targname=None,refband=None,stretch='sqrt',clipfrac=None,
                   siglo=None,sighi=None):

    # siglo = 2, sighi = 150 seems to work pretty well for most images.

    paths = pp.get_paths(obstype=obstype, targname=targname)
    outpath = paths['output']
    outname = targname

    files = glob.glob(outpath+outname+'_*latest_stack.fits')

    headers = {}
    for f in files:
        h = fits.getheader(f)
        band = h['FILTER']
        print(band)
        headers[band] = h

    refhead = headers[refband]
    for f in files:
        im,h = read_image(f)
        band = h['FILTER']

        newim = hcongrid(im,h,refhead)

        if clipfrac is not None:
            maxval = np.max(newim)
            newim[newim > fracclip*maxval] = fracclip*maxval

        if siglo is not None:
            sig = rb.std(newim)
            med = np.median(newim)
            clipval = med-siglo*sig
            newim[newim <= clipval] = clipval

        if sighi is not None:
            sig = rb.std(newim)
            med = np.median(newim)
            clipval = med+sighi*sig
            newim[newim >= clipval] = clipval

        if stretch == 'sqrt':
            minval = np.nanmin(newim)
            newim = newim - minval
            newim = np.sqrt(newim)

        if stretch == 'log':
            minval = np.nanmin(newim)
            newim = newim - minval + 1
            newim = np.log10(newim)

        fits.writeto(outpath+'_temp.fits',newim.astype('float32'),refhead)

        cmd = 'convert '+outpath+'_temp.fits -format TIFF -depth 32 '+outpath+outname+'_'+band+'.tif'
        doit = os.system(cmd)

        rmcmd = 'rm '+outpath+'_temp.fits'
        doit = os.system(rmcmd)


#----------------------------------------------------------------------
# Stack the latest night of data
#----------------------------------------------------------------------

def get_plot(targname=None,obstype='SNe',band='r',
             fwhm_max=5.0,skymax=5000.0,dpix_max=200,
             write=True,sighi=20,siglo=3,clobber=True,
             fignum=22,date=None):

    paths = pp.get_paths(obstype=obstype,targname=targname)
    dates = pp.get_dates(targname)

    if date is None:
        latest = dates[-1]
    else:
        latest = date

    print('Obtaining data from '+latest+'...')

    files,fct = pp.get_files(prefix=targname,tag=band+'_',date=latest)

    temp_im,th = pp.all_stack(files,outpath=paths['output'],outname=targname+'_'+band+'_latest_stack',
                              obstype=obstype,targname=targname,clobber=clobber,
                              fwhm_max=fwhm_max,skymax=skymax,dpix_max=dpix_max)

    if temp_im is None:
        print('Data do not pass quality criteria')
        return

    header = fits.getheader(files[0])
    ras = th['OBJCTRA']
    decs = th['OBJCTDEC']

    display_image(temp_im,fignum=fignum,figsize=(8,6),siglo=siglo,sighi=sighi)

    # Get the World Coordinate System information from the header
    w = wcs.WCS(th)

    c = SkyCoord(ras,decs, unit=(u.hour, u.deg))
    x, y = wcs.utils.skycoord_to_pixel(c, w)

    plt.plot(x,y,'o',ms=15,markerfacecolor='none',
             markeredgewidth=2,label=targname)

    plt.legend(loc=2,bbox_to_anchor=(1.01,1.01))

    return



def do_SN_phot(targname=None,clobber=False,nightclobber=False,targclobber=False,stackclobber=False,
               maxMag=15.5,minMag=13,maxStd=0.01,skymax=3000,maxErr=0.01,matchtol='high',
               ra=None,dec=None):
    '''
    ra and dec need to be strings in the form "HH:MM:SS.S" and "DD:MM:SS.S"
    
    '''
    
    print('Starting photometry on '+targname)
    allfiles,afct = pp.get_files(prefix=targname)
    header = fits.getheader(allfiles[afct/2])
    if ra is None and dec is None:
        rastr = header['OBJCTRA']
        decstr = header['OBJCTDEC']
    else:
        rastr = ra
        decstr = dec
        
    targ_info = get_SN_targs(targname=targname,ra=rastr,dec=decstr,maxMag=maxMag,minMag=minMag,
                             maxErr=maxErr,maxStd=maxStd,skymax=skymax,clobber=targclobber,
                             stackclobber=stackclobber)
    plt.close()

    phot = SN_dophot_batch(targ_info,targname=targname,clobber=clobber,
                           nightclobber=nightclobber,matchtol=matchtol)

    return phot


def do_all_SN_phot():

    SN = all_targets_observed(prefix='SN201')
    for targname in SN:
        phot = do_SN_phot(targname=targname)
    return None



def update_SN(targname,closeplot=False,clobber=True,nightclobber=False,targclobber=False,stackclobber=False,
              maxMag=15.5,minMag=12,maxStd=0.01,maxErr=0.02,square=False,skymax=3000,matchtol='high',
              ra=None,dec=None):
    '''
    
    '''
    
    badcoords = ['SN2020uxz','SN2020rcq']
    if targname in badcoords and ra is None and dec is None:
        print('Known coordinate mismatch: Please use TNS coordinates!')
        
    paths = pp.get_paths(obstype='SNe',targname=targname)
    phot = do_SN_phot(targname=targname,clobber=clobber,nightclobber=nightclobber,targclobber=targclobber,
                      stackclobber=stackclobber,maxMag=maxMag,minMag=minMag,maxStd=maxStd,maxErr=maxErr,
                      skymax=skymax,matchtol=matchtol,ra=ra,dec=dec)
    outfile1 = paths['output']+targname+'_lightcurve.png'
    outfile2 = paths['output']+targname+'_lightcurve_square.png'
    plot_SN_phot(phot,targname=targname,outfile=outfile1,write=True)
    if closeplot:
        plt.close()

    if square:
        plot_SN_phot_square(phot,targname=targname,outfile=outfile2,write=True)
        if closeplot:
            plt.close()

    return


def plot_SN_phot(phot,offset=1,write=True,outfile=None,targname=None,num=1):
    plt.ion()
    plt.figure(num,figsize=(12,6))
    plt.clf()
    i = 0
    offset = 1
    for band in ['g','V','r','i','z']:
        if band in phot.keys():
            if band == 'V':
                color='green'
            if band == 'g':
                color='blue'
            if band == 'r':
                color='red'
            if band == 'i':
                color='brown'
            if band == 'z':
                color='black'
            if len(phot[band]['mag']) > 0:
                   t = Time(phot[band]['bjd'],scale='tdb',format='jd')
                   dt = t.datetime
                   mag = phot[band]['mag']+offset*i
                   plt.errorbar(dt,mag,yerr=phot[band]['mag_err'],fmt='o',
                                color=color,label=band+'-band +'+str(offset*i))
                   i += 1
        else:
            pass
    plt.gca().invert_yaxis()
    plt.legend(numpoints=1,loc='best')
    plt.ylabel('Magnitude + offset',fontsize=14)
    plt.xlabel('Coordinated Universal Time',fontsize=14)
    plt.grid(linestyle=':')
    if targname:
        plt.title(targname,fontsize=16)
    plt.tight_layout()

    if write:
        plt.savefig(outfile,dpi=300)

    return

def plot_SN_phot_square(phot,offset=1,write=True,outfile=None,targname=None,num=1):
    plt.ion()
    plt.figure(num,figsize=(10,10))
    plt.clf()
    i = 0
    offset = 1
    for band in ['g','V','r','i','z']:
        if band in phot.keys():
            if band == 'V':
                color='green'
            if band == 'g':
                color='blue'
            if band == 'r':
                color='red'
            if band == 'i':
                color='brown'
            if band == 'z':
                color='black'
            if len(phot[band]['mag']) > 0:
                   t = Time(phot[band]['bjd'],scale='tdb',format='jd')
                   dt = t.datetime
                   mag = phot[band]['mag']+offset*i
                   plt.errorbar(dt,mag,yerr=phot[band]['mag_err'],fmt='o',
                                color=color,label=band+'-band +'+str(offset*i))
                   i += 1
        else:
            pass
    plt.gca().invert_yaxis()
    plt.legend(numpoints=1,loc='best')
    plt.ylabel('Magnitude + offset',fontsize=14)
    plt.xlabel('Coordinated Universal Time',fontsize=14)
    plt.grid(linestyle=':')
    if targname:
        plt.title(targname,fontsize=16)
    plt.tight_layout()

    if write:
        plt.savefig(outfile,dpi=300)

    return


def difference_image(temp_im,th,sci_im,sh,outpath=None,outname=None,write=False,clobber=False):

    # Do we need to redo difference image?
    if outpath!=None and outname!=None:
        if len(glob.glob(outpath+outname)) == 1:
            if clobber:
                os.system('rm '+outpath+outname)
            else:
                diffim,header = read_image(outpath+outname)
                return diffim, header

    # Get target name from the header
    targname = sh['OBJECT'].split('_')[1]

    # Setup correct paths
    paths = pp.get_paths(obstype='SNS',targname=targname)

    # Subtract background from images for cleaner difference image
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(sci_im, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    bkgt = Background2D(temp_im, (50, 50), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    # Align science image with template image
    sci_im = hcongrid(sci_im-bkg.background,sh,th)

    temp_im = temp_im-bkgt.background

    # Get information on the sources in the science and template images
    # using sextractor
    tempdata = pp.do_sextractor(temp_im,th,detect_thresh=5.0,analysis_thresh=5.0,
                                minpix=3,outdir=paths['output'])
    scidata = pp.do_sextractor(sci_im,sh,detect_thresh=5.0,analysis_thresh=5.0,
                               minpix=3,outdir=paths['output'])

    temp_fwhm = np.median(tempdata['FWHM_WORLD']*3600.0) # in arcseconds
    sci_fwhm = np.median(scidata['FWHM_WORLD']*3600.0)   # in arcseconds
    temp_sig = temp_fwhm/(2*np.sqrt(2*np.log(2)))
    sci_sig = sci_fwhm/(2*np.sqrt(2*np.log(2)))

    pstemp = pp.get_plate_scale(th)
    temp_sig_pix = temp_sig/pstemp
    pssci = pp.get_plate_scale(sh)
    sci_sig_pix = sci_sig/pssci

    # Create Gaussian smoothing kernel
    k = np.sqrt(max(temp_sig_pix,sci_sig_pix)**2-min(temp_sig_pix,sci_sig_pix)**2)
    d = int(round(10*k))
    x = np.linspace(0, d,d)
    y = np.linspace(0, d,d)
    x, y = np.meshgrid(x, y)
    kernel = pp.Gaussian2D((x, y),1.,d/2.,d/2.,k,k,0.,0.) #2d gaussian with sigma k, normalized
    kernel = (kernel/np.sum(kernel)).reshape(int(d),int(d))


    # Smooth whichever image needs smoothing
    if temp_fwhm > sci_fwhm:
        sci_im =  signal.fftconvolve(sci_im, kernel, mode='same')
    else:
        temp_im =  signal.fftconvolve(temp_im, kernel, mode='same')

    tempdata = pp.do_sextractor(temp_im,th,detect_thresh=5.0,analysis_thresh=5.0,
                                minpix=3,outdir=paths['output'])
    scidata = pp.do_sextractor(sci_im,th,detect_thresh=5.0,analysis_thresh=5.0,
                               minpix=3,outdir=paths['output'])

    # Scale the images to the same flux level
    fratio = []
    for i in range(len(scidata)):
        x,y = scidata['X_IMAGE'][i],scidata['Y_IMAGE'][i]
        dists = np.sqrt((x-tempdata['X_IMAGE'])**2+(y-tempdata['Y_IMAGE'])**2)
        argmin = np.argmin(dists)
        if dists[argmin] < 1.0:
            fratio = np.append(fratio,scidata['FLUX_BEST'][i]/tempdata['FLUX_BEST'][argmin])

    scale = np.median(fratio)

    # Do the subtraction!!!
    diffim = sci_im - temp_im*scale

    # Update header dates and times to coincide with science image
    # This seems clunky and needs revisiting
    th['EXPTIME'] = sh['EXPTIME']
    th['EXPOSURE'] = sh['EXPOSURE']
    th['UT'] = sh['UT']
    th['TIME-OBS'] = sh['TIME-OBS']
    th['DATE'] = sh['DATE']
    th['DATE-OBS'] = sh['DATE-OBS']
    th['JD'] = sh['JD']
    try:
        th['JD-HELIO'] = sh['JD-HELIO']
    except:
        pass
    try:
        th['JD-OBS'] = sh['JD-OBS']
    except:
        pass
    try:
        th['HJD-OBS'] = sh['HJD-OBS']
    except:
        pass
    try:
        th['BJD-OBS'] = sh['BJD-OBS']
    except:
        pass
    try:
        th['HA'] = sh['HA']
        th['OBJCTHA'] = sh['OBJCTHA']
    except:
        pass
    th['FWHM'] = max([temp_fwhm, sci_fwhm])

    if write:
        fits.writeto(outpath+outname,diffim,th)

    return diffim, th

def check_diff_im(diffim,header,outpath=None,outname=None,write=False,plot=False,
                  detect_thresh=3.0,analysis_thresh=3.0,minpix=3):
    '''
    This should output a dictionary with
    - date, RA, Dec, x, y, diff_fwhm, elongation
    '''

    # Get basic info
    targname = header['OBJECT'].split('_')[1]
    paths = pp.get_paths(obstype='SNS',targname=targname)
    date = header['DATE-OBS'].split('T')[0].replace('-','')
    outpath = paths['output']+date+'/'

    ##Find the raw science image
    files,fct = pp.get_files(prefix='SNS_'+targname)
    files.sort()
    snum = files.index(fnmatch.filter(files,'*'+date+'*')[0])
    sci_im,sh = read_image(files[snum])

    # Display the difference image
    if plot:
        display_image(diffim)

    # Get information on the sources in the difference image
    diffdata = pp.do_sextractor(diffim,header,detect_thresh=detect_thresh,
                                analysis_thresh=analysis_thresh,
                                minpix=minpix,outdir=outpath)

    # Use a box size of 5 times the FWHM of the image
    boxsize = int(np.round(np.median(diffdata['FWHM_IMAGE'])*5))

    # How far are the sources from the center of the image?
    dists = np.sqrt((1023-diffdata['X_IMAGE'])**2+(1023-diffdata['Y_IMAGE'])**2)

    # Loop thorugh all potential sources and decide if they might be legitimate
    # plot the best candidates in yellow
    xvec = []; yvec = []; ravec = []; decvec = []
    datevec = []; fwhmvec = []; elongvec = []
    fluxvec = []; fluxerrvec = []
    if len(diffdata) <= 5:
        for i in range(len(diffdata)):
            xi = int(np.round(diffdata['X_IMAGE'][i]-boxsize/2))
            xf = int(np.round(diffdata['X_IMAGE'][i]+boxsize/2))
            yi = int(np.round(diffdata['Y_IMAGE'][i]-boxsize/2))
            yf = int(np.round(diffdata['Y_IMAGE'][i]+boxsize/2))
            elon = diffdata['ELONGATION'][i]
            box = diffim[yi:yf,xi:xf]
            pos = np.sum(box[box>0])
            neg = np.sum(box[box<0])
            xs = diffdata['X_IMAGE'][i]
            ys = diffdata['Y_IMAGE'][i]

            xint = np.int(np.round(xs))
            yint = np.int(np.round(ys))
            if xint<= 100: xmin = 0
            else: xmin = xint-100
            if xint>=1948: xmax = 2048
            else: xmax = xint+100
            if yint<=0: ymin = 0
            else: ymin = yint-100
            if yint>=1948: ymax = 2048
            else: ymax = yint+100
            box2 = sci_im[xmin:xmax,ymin:ymax]
            notsaturated = np.sum(box2>30000) < 10

            if np.abs(neg) < 0.10*pos and np.abs(np.min(box)) < 0.1*np.max(box) and \
               dists[i] < 500 and elon < 5 and notsaturated:
                color='yellow'
                ra,dec = pp.xy_to_radec(xs,ys,header)
                xvec = np.append(xvec,xs) ; yvec = np.append(yvec,ys)
                ravec = np.append(ravec,ra) ; decvec = np.append(decvec,dec)
                datevec = np.append(datevec,date)
                fwhmvec = np.append(fwhmvec,diffdata['FWHM_WORLD'][i]*3600)
                elongvec = np.append(elongvec,diffdata['ELONGATION'][i])
                fluxvec = np.append(fluxvec,diffdata['FLUX_BEST'][i])
                fluxerrvec = np.append(fluxerrvec,diffdata['FLUXERR_BEST'][i])
                # Need to write out information about candidate to a text file!
            else:
                color='green'
            if plot:
                plt.plot(diffdata['X_IMAGE'][i],diffdata['Y_IMAGE'][i],'o',color=color,ms=15,fillstyle='none')

        DFout = pd.DataFrame({'date':datevec,'ra':ravec,'dec':decvec,'x':xvec,'y':yvec,'flux':fluxvec,
                              'fluxerr':fluxerrvec,'fwhm':fwhmvec,'elong':elongvec},
                             columns=['date','ra','dec','x','y','flux','fluxerr','fwhm','elong'])
        return DFout
    else:
        return


def find_SN(target,band='r',clobber=False,write=False):
    '''
    Loop through each image for specified target and search for new transient candidates
    '''

    # Get appropriate paths
    paths = pp.get_paths(targname=target,obstype='SNS')

    # Find all files for SNS target
    files,fct = pp.get_files(prefix='SNS_'+target,tag='-'+band)
    files.sort()

    try:
        tempfile = glob.glob('/home/student/SNS/reference_images/*'+target+'*.fits')
        temp_im,th = read_image(tempfile[0])
        tnum = files.index(fnmatch.filter(files,'*'+tempfile.split('/')[-1].split('_')[0]+'*'))

    except:
        tempfile = files[0]
        tnum = 0
        temp_im,th = read_image(tempfile)


    ######################################################################
    # Take first image as template and search second image for transients
    # After that, create a template from all previous images, and search for
    # transients in the difference image
    # If there are candidates:
    #     make postage stamps for each
    #     output candidate info into targname directory.

    if (len(files[tnum+1:])>0):

        pbar = tqdm(desc = 'Searching for SNe', total = len(files[tnum+1:]), unit = 'file')
        for i in np.arange(len(files[tnum+1:])-1)+1:
            sci_date = files[i].split('/')[-2]
            outpath = paths['output']+sci_date+'/'+target+'_'+sci_date+'_'

            # Get science image
            scifile = glob.glob(outpath+'science_image.fits')
            if len(scifile) == 1 and not clobber:
                sci_im,sh = read_image(scifile[0])
            else:
                if len(scifile) == 1 and clobber:
                    os.system('rm '+scifile[0])
                sci_raw,sh = read_image(files[i])
                sci_cals = pp.get_cal_frames(date=sci_date,targname=target,band=band,
                                             obstype='SNS',write=False)
                sci_im = pp.calibrate_image(image=sci_raw,header=sh,rotated_flat=False,
                                            cals=sci_cals)

            # Get summary file if there is one
            summaryfile = glob.glob(paths['output']+'search_summary.csv')
            if len(summaryfile) == 1:
                DFsummary = pd.read_csv(summaryfile[0])
                if len(DFsummary) > 0:
                    cdates = DFsummary['date'].values
                    cdate  = np.sort(np.array(cdates))[0]
                    fdates = np.array([int(a.split('/')[-2]) for a in files])
                    maxi  = min(i,np.max(np.where(fdates <= cdate)))
            else:
                maxi = i

            # Get or make difference image
            difffile = glob.glob(outpath+'difference_image.fits')
            if len(difffile) == 1 and not clobber:
                diff_im,dh = read_image(difffile[0])
            else:
                if len(difffile) == 1 and clobber:
                    os.system('rm '+difffile[0])
                diff_im,dh = difference_image(temp_im,th,sci_im,sh)

            # Check difference image
            diff_check = check_diff_im(diff_im,dh)


            if len(summaryfile) == 1:
                # Are there candidates for this date already in summaryfile? If so, do you
                # want to overwrite?
                # Need code here
                DFsummary = DFsummary.append(diff_check)
            else:
                DFsummary = diff_check

            print('Saving science image, template image, and difference image from '+
                target+' on '+sci_date+' as fits files')
            fits.writeto(outpath+'science_image.fits',sci_im,sh,overwrite=True)
            fits.writeto(outpath+'template_image.fits',temp_im,th,overwrite=True)
            fits.writeto(outpath+'difference_image.fits',diff_im,dh,overwrite=True)
            if len(diff_check) >= 1 and len(diff_check) <= 5:
                print('There are '+str(len(diff_check))+' candidates on '+sci_date+'!')
                # Append diff_check information to summary file CSV file or create new one.
                for ii in range(len(diff_check)):
                    mptitle = str(diff_check['ra'][ii])+'_'+str(diff_check['dec'][ii])+'_'+target+'_'+str(diff_check['date'][ii])
                    make_patch(diff_im,diff_check['x'][ii],diff_check['y'][ii],
                               title=mptitle+' - Difference Image',
                               outpath=outpath,outname=mptitle+'_difference',
                               write=True)
                    make_patch(temp_im,diff_check['x'][ii],diff_check['y'][ii],
                               title=mptitle+' - Template Image',
                               outpath=outpath,outname=mptitle+'_template',
                               write=True)
                    x,y = pp.radec_to_xy(diff_check['ra'][ii],diff_check['dec'][ii],sh)
                    make_patch(sci_im,x,y,
                               title=mptitle+' - Science Image',
                               outpath=outpath,outname=mptitle+'_science',
                               write=True)
            else:
                print("Did not find anything of interest :'(")
            DFsummary.to_csv(paths['output']+'search_summary.csv',index=False)
            pbar.update(1)
        pbar.close()
        return

    else:
        print('Not enough images!')

def make_patch(image,xpos,ypos,sz=100,title=None,siglo=3.0,sighi=5.0,outpath='./',
               outname='patch',write=False):
    yround = int(round(ypos))
    xround = int(round(xpos))
    patch = image[yround-sz/2:yround+sz/2+1,xround-sz/2:xround+sz/2+1]

    position = (xpos-np.round(xpos)+sz/2,ypos-np.round(ypos)+sz/2)

    sig = mad_std(patch)
    med = np.median(patch)
    vmin = med - siglo*sig
    vmax = med + sighi*sig
    plt.figure(101)
    plt.clf()
    plt.imshow(patch,vmin=vmin,vmax=vmax,cmap='gist_heat',interpolation='nearest',origin='lower')
    plt.scatter(sz/2,sz/2,marker='+',s=200,color='yellow',linewidth=1.5,label='Original position')
    plt.title(title)
    if write:
        testfile = glob.glob(outpath+outname+'.png')
        if len(testfile) == 1:
            os.system('rm '+testfile[0])
        plt.savefig(outpath+outname+'.png',dpi=300)

def update_candidate_log(diffdata, index,targname=None,obstype=None):
    # Need to figure out protocol for outputting candidate information
    # Probably don't need a function for this. Pandas can handle this in one or two lines.
    pass

'''

gals = all_targets_observed()
obstype = 'SNS'
gal = gals[100]
targname = gal
'''
def check_gal(targname=None,obstype='SNS'):
    '''
    Need to check first template image.
    What if template image sucks!?!?
    How do you decide that your template image sucks?
    '''


    # Get paths and dates that target has been observed
    paths = pp.get_paths(targname=targname,obstype=obstype)
    dates = pp.get_dates(targname,obstype)

    # Look for target and template files
    targfiles,tct = pp.get_files(prefix='SNS_'+targname)
    tempfiles,tempct = pp.get_files(prefix='SNST_'+targname)

    # If there is a template image, start with that
    if tempct == 1:
        template, th = read_image(tempfiles[0])
        tdate = tempfiles[0].split('/')[-2]
        tcals = pp.get_cal_frames(tdate,obstype=obstype,targname=targname)
        temp_im = pp.calibrate_image(template,th,tcals,rotated_flat=False)
    # Otherwise use the earliest image as the "template"
    elif len(dates) > 1:
        tfiles,tfct =  pp.get_files(prefix='SNS_'+targname,date=dates[0])
        if tfct == 1:
            cals = pp.get_cal_frames(dates[0],obstype=obstype,targname=targname)
            traw,th = read_image(tfiles[0])
            temp_im = pp.calibrate_image(traw,th,cals,rotated_flat=False)
        elif tfct > 1:
            temp_im,th = pp.all_stack(tfiles,outpath=paths['output'],outname=targname+'_stack',
                                obstype=obstype,targname=targname,calwrite=True,clobber=True,
                                fwhm_max=5,dpix_max=100)
    # If there has only been one observation, then there is nothing to check!
    else:
        print('No SNST template image for '+g)
        print('and not enough observations to make one!')
        return None

    # If there are enough images to do difference imaging, then proceed
    usedates = dates if tempct == 1 else dates[1:]

    for d in usedates:
        sfiles,sfct = pp.get_files(prefix='SNS_'+targname,date=d)
        if sfct == 1:
            cals = pp.get_cal_frames(d,obstype=obstype,targname=targname)
            sraw,sh = read_image(sfiles[0])
            sci_im = pp.calibrate_image(sraw,sh,cals,rotated_flat=False)
        elif sfct > 1:
            sci_im,sh = pp.all_stack(files,outpath=paths['output'],outname=targname+'_stack',
                                obstype=obstype,targname=targname,calwrite=True,clobber=True,
                                fwhm_max=5,dpix_max=100)

        # Path and outname for difference image
        outpath = paths['output']+d+'/'
        outname = targname+'_diffim.fits'
        # Compute difference image and write out
        diffim, header = sn.difference_image(temp_im,th,sci_im,sh,write=True,
                                             outpath=outpath,outname=outname)


        poi = sn.check_diff_im(diffim,header)

        # If length of poi is greater than zero then

        # run DoPHOT on science image

        # get information about candidates from doPHOT output

        # Write entry into SNS/targname/targname_candidates.txt


        # Then move on to next image. If candidates exist, don't update template





def ZF_v(t, t0, tb, a1, a2, s, A):

    v  = A*( (t-t0)/tb )**a1 * (1 + ((t-t0)/tb)**(s*(a1-a2)))**(-1/s)

    return v


def ZF_L(t, t0, tb, a1, a2, s, Ap):

    L = Ap*( (t-t0)/tb )**(2*(a1 + 1)) * (1 + ((t-t0)/tb)**(s*(a1-a2)))**(-2/s)

    return L

def ZF_m(t, t0, tb, a1, a2, s, C):

    m = -2.5*np.log10(ZF_L(t, t0, tb, a1, a2, s, C))

    return m

from scipy.optimize import minimize
'''
def ZF_nll(params,args=(time,mags,magerr)):

    t0, tb, a1, a2, s, C = params

    mp = ZF_m(time, t0, tb, a1, a2, s, C)


    nll = -0.5*np.sum( (mags - mp)**2/magerr**2 + np.log(2*np.pi*magerr**2) )

    return nll

t0 = 0.0
tb = 20.0
a1 = 0.1
a2 = -2.0
s = 1.0
A = 10.0
t = np.linspace(0,100,1000)

v = ZF_v(t,t0,tb,a1,a2,s,A)
plt.ion()
plt.figure(1)
plt.clf()
plt.plot(t,v,'-')

Ap = 1000.0
L = ZF_L(t, t0, tb, a1, a2, s, Ap)
plt.figure(2)
plt.clf()
plt.plot(t,L,'-')

C = 1.0
m = ZF_m(t, t0, tb, a1, a2, s, C)
plt.figure(3)
plt.clf()
plt.plot(t,m,'-')
plt.ylim(5,0)


plt.figure(4)
plt.clf()
plt.errorbar(time,mags,yerr=magerr,fmt='o',color='k')
plt.plot(t,m)

initParams = [t0, tb, a1, a2, s, C]

results = minimize(ZF_nll, initParams, args=(time,mags,magerr), method='Nelder-Mead')

print results.x
'''


def gals_with_obs(write=True,clobber=False):
    '''
    Returns galaxy names for which there are more than one image

    Outputs: Pickle file with galaxy names
    '''
    fname = '/home/student/SNS/reference_images/obsgals.pck'
    test = glob.glob(fname)
    if len(test) == 1 and not clobber:
        obsgals = pickle.load( open( fname, "rb" ) )
        return obsgals
    elif len(test) ==1 and clobber:
        command = 'rm ' +fname
        os.system(command)
    gals = sn.all_targets_observed()
    obsgals = []
    pbar = tqdm(desc = 'Finding images', total = len(gals), unit = 'galaxy')
    for g in gals:
        obs_files,fct = pp.get_files(prefix='SNS_'+g)
        if fct > 1:
            obsgals = np.append(obsgals,g)
        pbar.update(1)
    pbar.close()
    if write:
        outfile = open(fname, "w")
        pickle.dump(obsgals,outfile)
        outfile.close()

    return obsgals

def make_template_image(obsgals, band = 'r'):
    '''
    Makes and calibrates all template images of galaxies with more than one obs_summary

    Inputs: Obsgals pickle file

    Outputs: png files and fits files of template images in reference image directory
    '''
    pbar = tqdm(desc = 'Creating Templates', total = len(obsgals), unit = 'image')
    #obsgals = pd.read_pickle('/home/student/SNS/reference_images/obsgals')
    for g in obsgals:
        #final_files = glob.glob('/home/student/SNS/reference_images/'+g+'_template.png')
        #if len(final_files) ==1:
            #continue
        temp_file,fct = pp.get_files(prefix='SNST_'+g)
        if fct == 0:
            dates = pp.get_dates(targname=g, obstype='SNS')
            date = dates[0]
            temp_file,fct = pp.get_files(prefix='SNS_'+g,date=date)
            temp_file = temp_file[0]
        else:
            temp_file = temp_file[0]
            date = temp_file.split('/')[-2]
        temp_im,th = read_image(temp_file)
        targname = g
        temp_cals = pp.get_cal_frames(date=date,targname=targname,band=band,
                                         obstype='SNS',write=False)
        cal_temp_im = pp.calibrate_image(image=temp_im,header=th,rotated_flat=False,
                                        cals=temp_cals)
        outpath = '/home/student/SNS/reference_images'
        #files = glob.glob(outpath)
        #if len(files) == 0:
        #    mkdircmd = 'mkdir '+outpath
        #    os.system(mkdircmd)

        #write out the fits file
        fits.writeto(outpath+'/'+date+'_'+g+'_template.fits',cal_temp_im,th,overwrite=True)

        fignum = 2
        plt.figure(fignum)
        display_image(cal_temp_im,fignum=fignum)
        plt.title(g)
        plt.savefig(outpath+'/'+date+'_'+g+'_template.png',dpi=300)
        pbar.update(1)
    pbar.close()
    return

def fix_problem_temps(problem_temps,band = 'r'):
    '''
    Moves bad templates into bad_temps directory and makes new templates from the second date

    Input: SNS_template_check.csv

    Output: new template pngs and new template fits
    '''
    problem_temps = pd.read_csv('SNS_template_check.csv')
    pbar = tqdm(desc = 'Creating Templates', total = len(problem_temps['Name of Galaxy']), unit = 'image')
    for t in problem_temps['Name of Galaxy']:

        #create and calibrate new temp from second date
        dates = pp.get_dates(targname=t, obstype='SNS')
        date = dates[1]
        temp_file,fct = pp.get_files(prefix='SNS_'+t,date=date)
        temp_file = temp_file[0]
        temp_im,th = read_image(temp_file)
        targname = t
        temp_cals = pp.get_cal_frames(date=date,targname=targname,band=band,
                                         obstype='SNS',write=False)
        cal_temp_im = pp.calibrate_image(image=temp_im,header=th,rotated_flat=False,
                                        cals=temp_cals)
        #move bad temp png and fits file into bad_temps folder
        fname = date+'_'+t+'_template.png'
        ffits_name = date+'_'+t+'_template.fits'
        os.system('mv' +' ' +fname +' ' +'/home/student/SNS/reference_images/bad_temps')
        os.system('mv'+' '+ffits_name+' '+'/home/student/SNS/reference_images/bad_temps')
        outpath = '/home/student/SNS/reference_images'
        #write out new fits file
        fits.writeto(outpath+'/'+date+'_'+t+'_template.fits',cal_temp_im,th,overwrite=True)

        #output png in reference_images directory
        fignum = 2
        plt.figure(fignum)
        display_image(cal_temp_im,fignum=fignum)
        plt.title(t)
        plt.savefig(outpath+'/'+date+'_'+t+'_template.png',dpi=300)
        pbar.update(1)
    pbar.close()
    return


def search_images():
    '''
    Creates a composite image of the SNS candidate in the temp im, sci im, and diff im

    Input: search_summary.csv

    Output: composite composite images as pngs
    '''
      #  Creates a composite image of the SNS candidate in the temp im, sci im, and diff im

       # Input: search_summary.csv

       # Output: composite composite images as pngs
       # ''
    ##placeholder until search_summary csvs are made for other gals##
    SNS_dirs = glob.glob('/home/student/SNS/SNS_NGC*')
    pbar = tqdm(desc = 'Creating Composite Images', total = len(SNS_dirs), unit = 'galaxies')
    for dir in SNS_dirs:
        #read search_summary.csv
        name = dir.split('/')[-1]
        target = name.split('_')[-1]
        #check to see if there is already a composite image
        composite_ims = glob.glob('/home/student/SNS/search_images/'+target+'*')
        if len(composite_ims) <= 1:
            summaryfile = glob.glob(dir+'/'+'search_summary.csv')
            if len(summaryfile) == 1:
                df = pd.read_csv(summaryfile[0])
                for i,row in df.iterrows():
                    d = df['date'][i]
                    dstr = str(d)
                    date = dstr[:8]
                    xpos=df['x'][i]
                    ypos=df['y'][i]
                    candidate=i+1
                    #get template, science, and diff fits files
                    temp = glob.glob('/home/student/SNS/reference_images/*'+target+'_template.fits')
                    template_image = temp[0]
                    science_image = glob.glob(dir+'/'+date+'/'+target+'_'+date+'_science_image.fits')[0]
                    difference_image = glob.glob(dir+'/'+date+'/'+target+'_'+date+'_difference_image.fits')[0]
                    #load images
                    temp,temph = read_image(template_image)
                    sci, scih = read_image(science_image)
                    diff, diffh = read_image(difference_image)


                    defaultsz = 300

                    # Convert RA and Dec to pixel coordinates
                    wt = wcs.WCS(temph)
                    ws = wcs.WCS(scih)
                    wd = wcs.WCS(diffh)

                    raval,decval = wd.wcs_pix2world(xpos,ypos,1)
                    world = np.array([[raval, decval]])

                    pixs = ws.wcs_world2pix(world,1) # Pixel coordinates of (RA, DEC)
                    xs = pixs[0,0]
                    ys = pixs[0,1]

                    pixt = wt.wcs_world2pix(world,1) # Pixel coordinates of (RA, DEC)
                    xt = pixt[0,0]
                    yt = pixt[0,1]

                    # maximum size of the image so that patch does not go off the edge of image
                    xszd = min(min(defaultsz,2048-xpos),xpos)
                    yszd = min(min(defaultsz,2048-ypos),ypos)
                    szd = min(xszd,yszd)

                    xszt = min(min(defaultsz,2048-xt),xt)
                    yszt = min(min(defaultsz,2048-yt),yt)
                    szt = min(xszt,yszt)

                    xszs = min(min(defaultsz,2048-xs),xs)
                    yszs = min(min(defaultsz,2048-ys),ys)
                    szs = min(xszs,yszs)

                    sz = np.min([szd,szt,szs])

                    # Ensure an odd number of pixels (so target is in center)
                    sz = sz - 1 if sz % 2 == 0 else sz

                    # Get pixel positions in each image
                    ydr = int(round(ypos))
                    xdr = int(round(xpos))
                    diff_patch = diff[ydr-sz/2:ydr+sz/2+1,xdr-sz/2:xdr+sz/2+1]
                    posd = (xpos-xdr+sz/2,ypos-ydr+sz/2)

                    ysr = int(round(ys))
                    xsr = int(round(xs))
                    sci_patch = sci[ysr-sz/2:ysr+sz/2+1,xsr-sz/2:xsr+sz/2+1]
                    poss = (xs-xsr+sz/2,ys-ysr+sz/2)

                    ytr = int(round(yt))
                    xtr = int(round(xt))
                    temp_patch = temp[ytr-sz/2:ytr+sz/2+1,xtr-sz/2:xtr+sz/2+1]
                    post = (xt-xtr+sz/2,yt-ytr+sz/2)

                    # Set plot parameters
                    offset = 15
                    length = 20
                    siglo = 2
                    sighi = 5

                    # The stretch will need to be a bit different in raw images
                    siglor = 1.5
                    sighir = 3

                    plt.ion()
                    plt.figure(10,figsize=(6,9),constrained_layout=False)
                    gs = gridspec.GridSpec(3, 2)
                    gs.update(wspace=0.025, hspace=0.15) # set the spacing between axes.

                    ax1 = plt.subplot(gs[0, 0])
                    vmin = np.median(temp_patch) - siglor*np.std(temp_patch)
                    vmax = np.median(temp_patch) + sighir*np.std(temp_patch)
                    ax1.imshow(temp_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
                    ax1.plot([post[0]+offset, post[0]+offset+length],
                             [post[1], post[1]],lw=2,color='cyan')
                    ax1.plot([post[0], post[0]],
                             [post[1]+offset, post[1]+offset+length],lw=2,color='cyan')
                    plt.title('Template',color='black')
                    ax1.axis('off')

                    ax2 = plt.subplot(gs[0, 1])
                    vmin = np.median(sci_patch) - siglor*np.std(sci_patch)
                    vmax = np.median(sci_patch) + sighir*np.std(sci_patch)
                    ax2.imshow(sci_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
                    ax2.plot([poss[0]+offset, poss[0]+offset+length],
                             [poss[1], poss[1]],lw=2,color='cyan')
                    ax2.plot([poss[0], poss[0]],
                             [poss[1]+offset, poss[1]+offset+length],lw=2,color='cyan')
                    plt.title('Science',color='black')
                    ax2.axis('off')

                    ax3 = plt.subplot(gs[1:3, 0:2])
                    vmin = np.median(diff_patch) - siglo*np.std(diff_patch)
                    vmax = np.median(diff_patch) + sighi*np.std(diff_patch)
                    ax3.imshow(diff_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
                    ax3.plot([posd[0]+offset, posd[0]+offset+length],
                             [posd[1], posd[1]],lw=2,color='cyan')
                    ax3.plot([posd[0], posd[0]],
                             [posd[1]+offset, posd[1]+offset+length],lw=2,color='cyan')
                    plt.title('Difference',color='black')
                    ax3.axis('off')

                    plt.suptitle(target+'  '+date+'  candidate '+str(candidate),fontsize=18)

                    plt.savefig(target+'_'+date+'_'+str(candidate)+'.png',dpi=300)
                pbar.update(1)
            else:
                print('There is no summary file for '+dir)
        else:
            print('composite images already exist for '+target)
    pbar.close()
    return
def search_images():
    '''
    Creates a composite image of the SNS candidate in the temp im, sci im, and diff im

    Input: search_summary.csv

    Output: composite composite images as pngs
    '''
    SNS_dirs = glob.glob('/home/student/SNS/SNS_*')
    pbar = tqdm(desc = 'Creating Composite Images', total = len(SNS_dirs), unit = 'galaxies')
    for dir in SNS_dirs:
        #read search_summary.csv
        name = dir.split('/')[-1]
        target = name.split('_')[-1]
        #check to see if there is already a composite image
        composite_ims = glob.glob('/home/student/SNS/search_images/'+target+'*')
        if len(composite_ims) <= 1:
            summaryfile = glob.glob(dir+'/'+'search_summary.csv')
            if len(summaryfile) == 1:
                df = pd.read_csv(summaryfile[0])
                for i,row in df.iterrows():
                    try:
                        d = df['date'][i]
                        dstr = str(d)
                        date = dstr[:8]
                        xpos=df['x'][i]
                        ypos=df['y'][i]
                        candidate=i+1
                        #get template, science, and diff fits files
                        temp = glob.glob('/home/student/SNS/reference_images/*'+target+'_template.fits')
                        template_image = temp[0]
                        science_image = glob.glob(dir+'/'+date+'/'+target+'_'+date+'_science_image.fits')[0]
                        difference_image = glob.glob(dir+'/'+date+'/'+target+'_'+date+'_difference_image.fits')[0]
                        #load images
                        temp,temph = read_image(template_image)
                        sci, scih = read_image(science_image)
                        diff, diffh = read_image(difference_image)


                        defaultsz = 300

                        # Convert RA and Dec to pixel coordinates
                        wt = wcs.WCS(temph)
                        ws = wcs.WCS(scih)
                        wd = wcs.WCS(diffh)

                        raval,decval = wd.wcs_pix2world(xpos,ypos,1)
                        world = np.array([[raval, decval]])

                        pixs = ws.wcs_world2pix(world,1) # Pixel coordinates of (RA, DEC)
                        xs = pixs[0,0]
                        ys = pixs[0,1]

                        pixt = wt.wcs_world2pix(world,1) # Pixel coordinates of (RA, DEC)
                        xt = pixt[0,0]
                        yt = pixt[0,1]

                        # maximum size of the image so that patch does not go off the edge of image
                        xszd = min(min(defaultsz,2048-xpos),xpos)
                        yszd = min(min(defaultsz,2048-ypos),ypos)
                        szd = min(xszd,yszd)

                        xszt = min(min(defaultsz,2048-xt),xt)
                        yszt = min(min(defaultsz,2048-yt),yt)
                        szt = min(xszt,yszt)

                        xszs = min(min(defaultsz,2048-xs),xs)
                        yszs = min(min(defaultsz,2048-ys),ys)
                        szs = min(xszs,yszs)

                        sz = np.min([szd,szt,szs])

                        # Ensure an odd number of pixels (so target is in center)
                        sz = sz - 1 if sz % 2 == 0 else sz

                        # Get pixel positions in each image
                        ydr = int(round(ypos))
                        xdr = int(round(xpos))
                        diff_patch = diff[ydr-sz/2:ydr+sz/2+1,xdr-sz/2:xdr+sz/2+1]
                        posd = (xpos-xdr+sz/2,ypos-ydr+sz/2)

                        ysr = int(round(ys))
                        xsr = int(round(xs))
                        sci_patch = sci[ysr-sz/2:ysr+sz/2+1,xsr-sz/2:xsr+sz/2+1]
                        poss = (xs-xsr+sz/2,ys-ysr+sz/2)

                        ytr = int(round(yt))
                        xtr = int(round(xt))
                        temp_patch = temp[ytr-sz/2:ytr+sz/2+1,xtr-sz/2:xtr+sz/2+1]
                        post = (xt-xtr+sz/2,yt-ytr+sz/2)

                        # Set plot parameters
                        offset = 15
                        length = 20
                        siglo = 2
                        sighi = 5

                        # The stretch will need to be a bit different in raw images
                        siglor = 1.5
                        sighir = 3

                        plt.ion()
                        plt.figure(10,figsize=(6,9),constrained_layout=False)
                        gs = gridspec.GridSpec(3, 2)
                        gs.update(wspace=0.025, hspace=0.15) # set the spacing between axes.

                        ax1 = plt.subplot(gs[0, 0])
                        vmin = np.median(temp_patch) - siglor*np.std(temp_patch)
                        vmax = np.median(temp_patch) + sighir*np.std(temp_patch)
                        ax1.imshow(temp_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
                        ax1.plot([post[0]+offset, post[0]+offset+length],
                                 [post[1], post[1]],lw=2,color='cyan')
                        ax1.plot([post[0], post[0]],
                                 [post[1]+offset, post[1]+offset+length],lw=2,color='cyan')
                        plt.title('Template',color='black')
                        ax1.axis('off')

                        ax2 = plt.subplot(gs[0, 1])
                        vmin = np.median(sci_patch) - siglor*np.std(sci_patch)
                        vmax = np.median(sci_patch) + sighir*np.std(sci_patch)
                        ax2.imshow(sci_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
                        ax2.plot([poss[0]+offset, poss[0]+offset+length],
                                 [poss[1], poss[1]],lw=2,color='cyan')
                        ax2.plot([poss[0], poss[0]],
                                 [poss[1]+offset, poss[1]+offset+length],lw=2,color='cyan')
                        plt.title('Science',color='black')
                        ax2.axis('off')

                        ax3 = plt.subplot(gs[1:3, 0:2])
                        vmin = np.median(diff_patch) - siglo*np.std(diff_patch)
                        vmax = np.median(diff_patch) + sighi*np.std(diff_patch)
                        ax3.imshow(diff_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
                        ax3.plot([posd[0]+offset, posd[0]+offset+length],
                                 [posd[1], posd[1]],lw=2,color='cyan')
                        ax3.plot([posd[0], posd[0]],
                                 [posd[1]+offset, posd[1]+offset+length],lw=2,color='cyan')
                        plt.title('Difference',color='black')
                        ax3.axis('off')

                        plt.suptitle(target+'  '+date+'  candidate '+str(candidate),fontsize=18)

                        plt.savefig(target+'_'+date+'_'+str(candidate)+'.png',dpi=300)
                        pbar.update(1)
                    except:
                        log = open('mistakes.txt','a')
                        log.write('There is something wrong with target ' +target)
                        log.close()
            else:
                print('There is no summary file for '+dir)
        else:
            print('composite images already exist for '+target)
        for i,row in df.iterrows():
            d = df['date'][i]
            dstr = str(d)
            date = dstr[:8]
            xpos=df['x'][i]
            ypos=df['y'][i]
            candidate=i+1
            #get template, science, and diff fits files
            template_image = glob.glob('/home/student/SNS/reference_images/'+'*'+name+'_template.fits')[0]
            science_image = glob.glob(dir+'/'+date+'/'+name+'_'+date+'_science_image.fits')[0]
            difference_image = glob.glob(dir+'/'+date+'/'+name+'_'+date+'_difference_image.fits')[0]
            #load images
            temp,temph = read_image(template_image)
            sci, scih = read_image(science_image)
            diff, diffh = read_image(difference_image)


            defaultsz = 300

            # Convert RA and Dec to pixel coordinates
            wt = wcs.WCS(temph)
            ws = wcs.WCS(scih)
            wd = wcs.WCS(diffh)

            raval,decval = wd.wcs_pix2world(xpos,ypos,1)
            world = np.array([[raval, decval]])

            pixs = ws.wcs_world2pix(world,1) # Pixel coordinates of (RA, DEC)
            xs = pixs[0,0]
            ys = pixs[0,1]

            pixt = wt.wcs_world2pix(world,1) # Pixel coordinates of (RA, DEC)
            xt = pixt[0,0]
            yt = pixt[0,1]

            # maximum size of the image so that patch does not go off the edge of image
            xszd = min(min(defaultsz,2048-xpos),xpos)
            yszd = min(min(defaultsz,2048-ypos),ypos)
            szd = min(xszd,yszd)

            xszt = min(min(defaultsz,2048-xt),xt)
            yszt = min(min(defaultsz,2048-yt),yt)
            szt = min(xszt,yszt)

            xszs = min(min(defaultsz,2048-xs),xs)
            yszs = min(min(defaultsz,2048-ys),ys)
            szs = min(xszs,yszs)

            sz = np.min([szd,szt,szs])

            # Ensure an odd number of pixels (so target is in center)
            sz = sz - 1 if sz % 2 == 0 else sz

            # Get pixel positions in each image
            ydr = int(round(ypos))
            xdr = int(round(xpos))
            diff_patch = diff[ydr-sz/2:ydr+sz/2+1,xdr-sz/2:xdr+sz/2+1]
            posd = (xpos-xdr+sz/2,ypos-ydr+sz/2)

            ysr = int(round(ys))
            xsr = int(round(xs))
            sci_patch = sci[ysr-sz/2:ysr+sz/2+1,xsr-sz/2:xsr+sz/2+1]
            poss = (xs-xsr+sz/2,ys-ysr+sz/2)

            ytr = int(round(yt))
            xtr = int(round(xt))
            temp_patch = temp[ytr-sz/2:ytr+sz/2+1,xtr-sz/2:xtr+sz/2+1]
            post = (xt-xtr+sz/2,yt-ytr+sz/2)

            # Set plot parameters
            offset = 15
            length = 20
            siglo = 2
            sighi = 5

            # The stretch will need to be a bit different in raw images
            siglor = 1.5
            sighir = 3

            plt.ion()
            plt.figure(10,figsize=(6,9),constrained_layout=False)
            gs = gridspec.GridSpec(3, 2)
            gs.update(wspace=0.025, hspace=0.15) # set the spacing between axes.

            ax1 = plt.subplot(gs[0, 0])
            vmin = np.median(temp_patch) - siglor*np.std(temp_patch)
            vmax = np.median(temp_patch) + sighir*np.std(temp_patch)
            ax1.imshow(temp_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
            ax1.plot([post[0]+offset, post[0]+offset+length],
                     [post[1], post[1]],lw=2,color='cyan')
            ax1.plot([post[0], post[0]],
                     [post[1]+offset, post[1]+offset+length],lw=2,color='cyan')
            plt.title('Template',color='black')
            ax1.axis('off')

            ax2 = plt.subplot(gs[0, 1])
            vmin = np.median(sci_patch) - siglor*np.std(sci_patch)
            vmax = np.median(sci_patch) + sighir*np.std(sci_patch)
            ax2.imshow(sci_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
            ax2.plot([poss[0]+offset, poss[0]+offset+length],
                     [poss[1], poss[1]],lw=2,color='cyan')
            ax2.plot([poss[0], poss[0]],
                     [poss[1]+offset, poss[1]+offset+length],lw=2,color='cyan')
            plt.title('Science',color='black')
            ax2.axis('off')

            ax3 = plt.subplot(gs[1:3, 0:2])
            vmin = np.median(diff_patch) - siglo*np.std(diff_patch)
            vmax = np.median(diff_patch) + sighi*np.std(diff_patch)
            ax3.imshow(diff_patch,vmin=vmin,vmax=vmax,cmap='gist_heat',origin='lower')
            ax3.plot([posd[0]+offset, posd[0]+offset+length],
                     [posd[1], posd[1]],lw=2,color='cyan')
            ax3.plot([posd[0], posd[0]],
                     [posd[1]+offset, posd[1]+offset+length],lw=2,color='cyan')
            plt.title('Difference',color='black')
            ax3.axis('off')

            plt.suptitle(name+'  '+date+'  candidate '+str(candidate),fontsize=18)

            plt.savefig(name+'_'+date+'_'+str(candidate)+'.png',dpi=300)
        pbar.update(1)
        return
    pbar.close()
    return



def find_all_SN():
    SNS_dirs = glob.glob('/home/student/SNS/SNS_*')
    for dir in SNS_dirs:
        target = dir.split('_')[-1]
        search_summary = glob.glob(dir+'/search_summary.csv')
        if len(search_summary) == 1:
            print('search_summary already exists for '+target)
            continue
        else:
            print('finding SN in '+target)
            find_SN(target)
