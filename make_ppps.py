import phot_pipe as pp
import glob,os,pdb
from astropy.io import fits
from tqdm import tqdm
import numpy as np
import robust as rb
from quick_image import *

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, Angle
# depreciated. Use simple_norm
#from astropy.visualization import scale_image
from astropy import units as u
from astropy.io.ascii import SExtractor
from photutils import CircularAperture, SkyCircularAperture, CircularAnnulus, aperture_photometry
from FITS_tools.hcongrid import hcongrid

from photutils import DAOStarFinder

from quick_image import *


def get_all_targets(verbose=True):
    '''
    Script to find all PPP targets observed to date (with solved data in the Archive)

    targets = get_all_targets(verbose=True)
    '''

    files,fct = pp.get_files(prefix='PPP')
    targets = np.unique(np.array([f.split('/')[-1].split('-')[0].split('_')[-1] for f in files]))
    if verbose:
        print('PPP targets observed to date:')
        for t in targets:
            print(t)
    return targets


def make_sextractor_files(path='./'):

    # Default parameter file
    exist = glob.glob(path+'default.param')
    if len(exist) == 1:
        print 'Deleting old default.param file'
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
        print 'Deleting old default.conv file'
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
        print 'Deleting old default.sex file'
        os.system('rm -rf '+exist[0])
    with open(path+'default.sex','a') as the_file:
        the_file.write('CATALOG_NAME     temp.cat       # name of the output catalog                    \n')		      
        the_file.write('CATALOG_TYPE     ASCII_HEAD     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,          \n')
        the_file.write('                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC          \n')	      
        the_file.write('PARAMETERS_NAME  default.param  # name of the file containing catalog contents  \n')
        the_file.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction) \n')
        the_file.write('DETECT_MINAREA   3              # min. # of pixels above threshold              \n')
        the_file.write('DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2  \n')
        the_file.write('ANALYSIS_THRESH  1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2  \n')
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
        the_file.write('PIXEL_SCALE      1.0            # size of pixel in arcsec (0=use FITS WCS info) \n')	
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


def check_headers(obstype='PPP',targname=None,band='Luminance'):

    # Get paths
    paths = pp.get_paths(obstype=obstype,targname=targname)

    # Get all files in a chosen band
    files,fct = pp.get_files(prefix=obstype+'_'+targname,tag=band)

    # Source extractor command
    sextractor = '/home/administrator/sextractor-2.19.5/src/sex'

    # Check for which header to use as reference
    xpos = [] ; ypos = []
    pa = [] ; ccdtemp = []
    exptime = []
    ccdtemp = []; naxis1= []; naxis2 = []
    naxis = []; bjd = []
    fwhm = [] ; filename = []
    pbar = tqdm(desc = 'Checking image headers', total = len(files), unit = 'file')
    for fname in files:
        header = fits.getheader(fname)
        cd = np.array([[header['CD1_1'],header['CD1_2']],
                       [header['CD2_1'],header['CD2_2']]])
        plate_scale = np.mean(np.abs(np.linalg.eig(cd)[0])*3600.0)
        rastr  = header['OBJCTRA']
        decstr = header['OBJCTDEC']
        coords = SkyCoord(rastr,decstr,unit=(u.hour,u.deg))
        RAdeg  = coords.ra.deg
        DECdeg = coords.dec.deg
        x,y    = pp.radec_to_xy(RAdeg,DECdeg,header)
        xpos   = np.append(xpos,x)
        ypos   = np.append(ypos,y)
        pa     = np.append(pa,pp.posang(header))
        try:
            ccdtemp = np.append(ccdtemp,header['CCD-TEMP'])
        except:
            ccdtemp = np.append(ccdtemp,990.0)
        exptime = np.append(exptime,header['EXPTIME'])
        filename = np.append(filename,fname)
        naxis1 = np.append(naxis1,header['NAXIS1'])
        naxis2 = np.append(naxis2,header['NAXIS2'])
        naxis  = np.append(naxis,header['NAXIS'])
        bjd    = np.append(bjd,header['BJD-OBS'])

        # Change pwd to file directory
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
           'bjd':bjd,'naxis1':naxis1,'naxis2':naxis2,'naxis':naxis,'fwhm':fwhm}

    return out


def stack_images(files,refhead,HDR=None,write=True,obstype=None,targname=None,weighted_mean=False):
    '''
    Update this to do a weighted mean of the frames according to the inverse
    variance of each image
    '''

    #file_dict,HDR=None,fwhm_max=3.0,pa_max=10,write=True):

    stack = None
    pbar = tqdm(desc = 'Stacking images', total = len(files), unit = 'file')
    for fname in files:
        temp,temph = read_image(fname)
        date = fname.split('/')[-2]
        band = temph['filter']
        cd = np.array([[temph['CD1_1'],temph['CD1_2']],
                       [temph['CD2_1'],temph['CD2_2']]])
        plate_scale = np.mean(np.abs(np.linalg.eig(cd)[0])*3600.0)
        cals = pp.get_cal_frames(date,band=band,write=True,archive=True,verbose=False,
                                 targname=targname,obstype=obstype,setting=1)
        if cals['bias'] is None:
            print('No calibration frames found for '+fname.split('/')[-1]+'!!!')
        else:
            cal = pp.calibrate_image(temp,temph,cals,rotated_flat=False,domask=False)
            std = rb.std(cal)
            median = np.median(cal)
            w = wcs.WCS(temph)
            # Align image with astrometry from the refrerence image
            xsz,ysz = np.shape(cal)
            newim = np.reshape(hcongrid(cal,temph,refhead),(xsz,ysz,1))
            if HDR is not None:
                test = np.copy(newim[:,:,0])
                zinds = np.where(test == 0)
                test[zinds] = 1
                diff = (test - HDR*temph['EXPTIME'])
                binds = np.where(np.abs(diff) > 200)
                test[binds] = HDR[binds]*temph['EXPTIME']
                test/= temph['EXPTIME']
                newim[:,:,0] = test
            if stack is None:
                stack = np.copy(newim)
                weight = 1./std**2
            else:
                stack = np.append(stack,newim/temph['EXPTIME'],axis=2)
                weight = np.append(weight,1./std**2)
        pbar.update(1)
    pbar.close()

    try:
        xsz,ysz,zsz = np.shape(stack)
    except:
        #    return None,None
        print "shape fail"
        return None

    if weighted_mean:
        print('Feature not implemented yet')
    else:
        if zsz == 1:
            final_med = stack[:,:,0]
        if zsz == 0:
            print "size fail"
            return None
        if zsz > 1:
            # Median filter
            final_med = np.nanmedian(stack,axis=2)

    return final_med


def make_layer(obstype='PPP',targname=None,band=None,dpix_max=100.0,dtheta_max=10.0,
               fwhm_max=3.0,HDR=True,ccd_settemp=-30.0,ccd_tol=3.0,write=True):

    paths = pp.get_paths(obstype=obstype, targname=targname)
    files,fct = pp.get_files(prefix=obstype+'_'+targname,tag=band,date=None,
                             suffix='solved.fits',raw=False,clean=False)

    # Set output
    outpath = paths['output']
    outname = targname+'_'+band

    # See if output files exist, if so delete them
    testfile = glob.glob(outpath+outname+'.fits')
    if len(testfile) != 0:
        rmcmd = 'rm '+outpath+outname+'.fits'
        os.system(rmcmd)

    testfile = glob.glob(outpath+outname+'.png')
    if len(testfile) != 0:
        rmcmd = 'rm '+outpath+outname+'.png'
        os.system(rmcmd)


    # Check headers
    hinfo = check_headers(targname=targname,band=band)

    # Find reference header (refhead) from file_dict
    xmean = np.median(hinfo['xpos'])
    ymean = np.median(hinfo['ypos'])
    dists =  np.sqrt((hinfo['xpos']-xmean)**2+(hinfo['ypos']-ymean)**2)
    patest = np.array([min((p-90)%90,90-p%90) for p in hinfo['PA']])

    # Keep on the files whose distances are less than dpix_max from median position of target
    gargs, = np.where((hinfo['fwhm'] > 0.5)&(hinfo['fwhm'] < fwhm_max)&
                      (np.abs(patest) < dtheta_max)& (dists < dpix_max) &
                      (hinfo['ccdtemp'] < (ccd_settemp + ccd_tol)) &
                      (hinfo['ccdtemp'] > (ccd_settemp - ccd_tol)))
    keepers = hinfo['filename'][gargs]
    exptime = hinfo['exptime'][gargs]
    dists = dists[gargs]

    if len(keepers) == 0:
        print('Criteria too stringent! No files left')
        return None,None

    refhead  = fits.getheader(keepers[np.argmin(dists)])

    nexptime = len(np.unique(exptime))
    
    if nexptime > 1 and HDR:
        minexp = np.min(exptime)
        minargs, = np.where(exptime == minexp)
        otherargs, = np.where(exptime != minexp)
    
        # Separate files according to exptime for HDR purposes
        shortfiles = np.array(keepers)[minargs]
        otherfiles = np.array(keepers)[otherargs]
        
        # Stack short exposures
        HDframe = stack_images(shortfiles,refhead,obstype=obstype,targname=targname)

        # Stack the rest of the images
        layer = stack_images(otherfiles,refhead,HDR=HDframe,obstype=obstype,targname=targname)

    else:
        if nexptime == 1 and HDR:
            print('There is only '+str(nexptime)+' exposure times for this source')
            print('HDR algorithm will not work')
            print('Using straight median')
        # Stack the rest of the images
        layer = stack_images(keepers,refhead,obstype=obstype,targname=targname)


    if write:
        display_image(layer)
        plt.savefig(outpath+outname+'.png',dpi=300)
        fits.writeto(outpath+outname+'.fits',layer.astype('float32'),refhead)

    return layer.astype('float32'), refhead


def convert_layers(obstype='PPP',targname=None,refband=None,stretch='sqrt',clipfrac=None,
                   siglo=None,sighi=None):

    # siglo = 2, sighi = 150 seems to work pretty well for most images.
    
    paths = pp.get_paths(obstype=obstype, targname=targname)
    outpath = paths['output']
    outname = targname

    files = glob.glob(outpath+outname+'_*.fits')

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

    return

    
def prep_pixinsight(obstype='PPP',targname=None,dpix_max=100.0,dtheta_max=10.0,
                    fwhm_max=3.0,ccd_settemp=-30.0,ccd_tol=3.0):


    # Source extractor command
    sextractor = '/home/administrator/sextractor-2.19.5/src/sex'


    # Get paths
    paths = pp.get_paths(obstype=obstype,targname=targname)

    outpath = paths['dropbox']+'Astronomy/PPP/'+targname+'/PixInsight/'
    if not os.path.isdir(outpath):
        outpath = outpath.replace(' ','\ ').replace('(','\(').replace(')','\)')
        print('Making directory: '+outpath)
        mkdircmd = 'mkdir '+outpath
        os.system(mkdircmd)

    
    # Get all dates of observations
    dates = pp.get_dates(targname,obstype)

    # Loop through dates, calibrate and vet images, then write good ones out to PixInsight directory
    for d in dates:

        # Get files
        allfiles,fct = pp.get_files(prefix=obstype+'_'+targname,tag='',date=d)

        # Determine which filters were used
        filters = []
        for fname in allfiles:
            header = fits.getheader(fname)
            filters.append(header['filter'].replace("'",""))
        filters = np.unique(filters)

        fstr = ''
        for val in filters:
            fstr = fstr+val+', '

        print('Filters used on '+d+': '+fstr[:-2])
        print('')

        for filt in filters:
            # Get cal frames
            cals = pp.get_cal_frames(d,band=filt,targname=targname,obstype=obstype)

            # Get files in specified filter
            files,fct =  pp.get_files(prefix=obstype+'_'+targname,tag='-'+filt+'_',date=d)

            # Check files
            pbar = tqdm(desc = 'Checking images on '+d+' in '+filt+' band', total = fct, unit = 'file')
            for fname in files:
                header = fits.getheader(fname)
                cd = np.array([[header['CD1_1'],header['CD1_2']],
                               [header['CD2_1'],header['CD2_2']]])
                plate_scale = np.mean(np.abs(np.linalg.eig(cd)[0])*3600.0)
                rastr  = header['OBJCTRA']
                decstr = header['OBJCTDEC']
                coords = SkyCoord(rastr,decstr,unit=(u.hour,u.deg))
                RAdeg  = coords.ra.deg
                DECdeg = coords.dec.deg
                x,y    = pp.radec_to_xy(RAdeg,DECdeg,header)
                #xpos   = np.append(xpos,x)
                #ypos   = np.append(ypos,y)
                dpos = np.sqrt((x-1024.)**2 + (y-1024)**2)
            
                pa     = pp.posang(header)
                patest = min((pa-180)%180,180-pa%180)
            
                try:
                    ccdtemp = header['CCD-TEMP']
                except:
                    ccdtemp = np.nan

                if np.abs(ccdtemp - ccd_settemp) < ccd_tol and dpos < dpix_max and patest < dtheta_max:
                    # Change pwd to file directory
                    os.chdir(outpath)
                    sexname = fname.replace('Dropbox (Thacher)','DropBox')
                    make_sextractor_files()
                    os.system(sextractor+' '+sexname)
                    sexfiles = SExtractor()
                    sexdata = sexfiles.read('temp.cat')
                    fwhm = np.median(sexdata['FWHM_IMAGE'][
                        (sexdata['ELONGATION'] < 1.05)&(sexdata['FWHM_IMAGE'] > 1)])*plate_scale
                    os.system('rm default.param')
                    os.system('rm default.conv')
                    os.system('rm default.sex')
                    os.system('rm temp.cat')

                    if fwhm < fwhm_max:
                        im,h = read_image(fname)
                        cal = pp.calibrate_image(im,h,cals,rotated_flat=False)
                        outname = fname.split('/')[-1].split('.')[0]+'_cal'+'.fits'
                        fits.writeto(outpath+outname, np.float32(cal), h)
                        
                pbar.update(1)                
            pbar.close()


