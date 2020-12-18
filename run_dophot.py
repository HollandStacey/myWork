##########################################
from tqdm import tqdm, trange
import sys
sys.path.append('/home/student/python/photometry')
sys.path.append('/home/student/python/utils')
from quick_image import *
import pickle
import pandas as pd
import numpy as np
import os
from astropy.io import fits
import time
from datetime import datetime
import matplotlib.pyplot as plt
from ast import literal_eval
import phot_pipe as pp
from panstarrs_query import *
import FITS_tools.hcongrid
import shutil
from astropy.wcs import WCS
#target = 'NGC6570'

def rundophot(target,thefile=None,tag='_sci',redo = False,SN=False):
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
        if False, exists the function

    Writes
    ------
    In /home/student/run_dophot/:
        * Modifies params_default_c
        * Copies image fits file into the directory
        * Outputs obj_out file
        * Outputs psf shape image
    """
    #Make tag include time stamp
    dophot_dir = '/home/student/run_dophot/'
    print '=========================='
    if thefile == None:
        thefile = glob.glob('/home/student/SNS/templates/SNST_'+target+'*.fit')
    else:
        thefile = glob.glob(thefile)
    if len(thefile) == 0:
        print "Input file: ",thefile," does not exist! Exiting..."
        return
    else:
        #Read image files
        #template  = '/home/student/SNS/templates/stacked/'+target+'_stacked.fit'
        print 'Running Dophot...'
        print 'target:  ', target
        thefile = thefile[0]

        basename = thefile.split('/')[-1]

        if SN:
            timestamp = '_'+basename.split('-')[1]
        else:
            timestamp = '_'+basename.split('_')[1].split('-')[1]
        #images = glob.glob('/home/student/SNS/'+target+'/*.fit')
        #diffim, tim, th, sim, sh = diff(template,images[0])
        print 'Input file: ', thefile
        done_files = [os.path.basename(x) for x in glob.glob(dophot_dir+'*')]
        if 'psf_out_'+target+timestamp+tag+'.fits' in done_files and basename in done_files and redo==False:
            print 'DoPHOT results Already exist! Exiting...'
            return

    ###################################
        print 'Modifying DoPhot parameters file...'
        #Move stuff to the dophot directory and modify parameters file (pm)
        if basename not in done_files:
            shutil.copy(thefile,dophot_dir)
        with open(dophot_dir+'param_default_c') as f:
            pm = f.readlines()

        pm[47] = "IMAGE_IN = '"+basename+"'               Input image name.\n"
        #pm[12] = "IMAGE_IN = '"+target+"_stacked.fit'               Input image name.\n"
        #pm[48] =  "IMAGE_OUT = 'image_out_"+target+timestamp+tag+".fits'    Output image name.\n"
        pm[50] =  "OBJECTS_OUT = 'obj_out_"+target+timestamp+tag+"'         Output object list file name.\n"
        #pm[17] = "=ERRORS_OUT = 'errors_out_"+target+timestamp+tag+"       Errors on fit to be output if output type is internal\n"
        #pm[18] =  "SHADOWFILE_OUT = 'shad_out_"+target+timestamp+tag+"'           Output shadow file name.\n"
        pm[51] = "EMP_SUBRAS_OUT = 'psf_out_"+target+timestamp+tag+".fits'       Empirical PSF subraster.\n"
        #pm[22] =  "=COVARS_OUT = 'covar_out_"+target+timestamp+tag+"'    File to contain covariance matrices\n"

        textfile = open(dophot_dir+'param_default_c', 'w')
        for i in range(len(pm)):
            textfile.write(pm[i])
        textfile.close()
    ################################
        #Run DoPHOT
        print "Running DoPHOT..."
        os.chdir(dophot_dir)
        os.system('cdophot param_default_c')
        #os.system('./dophot pm')
        #if show_results:
    return


def run_diff_dophot(target, thefile,tag='_diff',redo = False):
    """
    Runs cdophot on differential images


    Parameters
    ----------
    target: str
        name of the target. e.g. "NGC0865"
    thefile: str, optional
        name of the fits file, if None, default to the first template image of the target
        must be from the same target
    tag: str
        normally does not change, default to '_diff'
    redo: bool
        if True, rerun cdophot
        if False, exists the function

    Writes
    ------
    In /home/student/diff_dophot/:
        * Modifies pm
        * Copies image fits file into the directory
        * Outputs obj_out file
        * Outputs psf shape image
    """
    print '=========================='
    thefile = glob.glob(thefile)
    if len(thefile) == 0:
        print "Input file: ",thefile," does not exist! Exiting..."
        return
    else:
        ##################################
        #Read image files
        #template  = '/home/student/SNS/templates/stacked/'+target+'_stacked.fit'
        print 'Running Dophot...'
        print 'target:  ', target
        thefile = thefile[0]

        template_name = thefile.split('/')[-1]


        timestamp = '_'+template_name.split('_')[1].split('-')[1]

        #images = glob.glob('/home/student/SNS/'+target+'/*.fit')
        #diffim, tim, th, sim, sh = diff(template,images[0])
        print 'Input file: ', thefile
        done_files = [os.path.basename(x) for x in glob.glob('/home/student/DoPHOT_C/*')]
        if 'psf_out_'+target+timestamp+tag+'.fits' in done_files and template_name in done_files and redo==False:
            print 'DoPHOT results Already exist! Exiting...'
            return

        ###################################
        print 'Modifying DoPhot parameters file...'
        #Move stuff to the dophot directory and modify parameters file (pm)
        dophot_dir = '/home/student/diff_dophot/'
        #if thefile not in glob.glob(dophot_dir+'*'):
        #    shutil.copy(thefile,dophot_dir)
        with open(dophot_dir+'pm') as f:
            pm = f.readlines()

        pm[12] = "IMAGE_IN = '"+template_name+"'               Input image name.\n"
        #pm[12] = "IMAGE_IN = '"+target+"_stacked.fit'               Input image name.\n"
        pm[13] =  "IMAGE_OUT = 'image_out_"+target+timestamp+tag+".fits'    Output image name.\n"
        pm[15] =  "OBJECTS_OUT = 'obj_out_"+target+timestamp+tag+"'         Output object list file name.\n"
        pm[17] = "=ERRORS_OUT = 'errors_out_"+target+timestamp+tag+"       Errors on fit to be output if output type is internal\n"
        pm[18] =  "SHADOWFILE_OUT = 'shad_out_"+target+timestamp+tag+"'           Output shadow file name.\n"
        pm[19] = "EMP_SUBRAS_OUT = 'psf_out_"+target+timestamp+tag+".fits'       Empirical PSF subraster.\n"
        pm[22] =  "=COVARS_OUT = 'covar_out_"+target+timestamp+tag+"'    File to contain covariance matrices\n"

        textfile = open(dophot_dir+'pm', 'w')
        for i in range(len(pm)):
            textfile.write(pm[i])
        textfile.close()
        ################################
        #Run DoPHOT
        print "Running DoPHOT..."
        os.chdir(dophot_dir)
        os.system('cdophot pm')
        #if show_results:
    return



def match_catalogue(target, thefile, w, dophotoutput=None, 
tag = None, search_rad = 3.,redo_panstarrs=True,save=True,redo=False,vett=False,SN=False):
    """
    Get PanSTARRS data and match with DoPHOT output for photometric calibration


    Parameters
    ----------
    target: str
        the target you are considering
    thefile: str
        the filename of the input fits image
    dophotoutput: pandas dataframe, optional
        manually input pandas datafram as obj_out file to match with
    tag: str
        '_sci' or '_temp' or '_diff'
    search_rad: int
        number of pixles around each source to search in
    redo_panstarrs: bool
        if True, requery the server and downloads a new panstarrs catalogue for the image
    save: bool
        if True, saves the matched catalogue
    redo: bool
        if True, rematches the obj_out catalogue with the panstarrs catalogue
    vett: bool
        if True, only preforms the function on sources in obj_out whose type is '0x01' (i.e. perfect star shapes)

    Returns
    -------
    Pandas dataframe of matched sources 
    """
    dophot_dir = '/home/student/run_dophot/'
    basename = thefile.split('/')[-1]
    if SN:
            timestamp = '_'+basename.split('-')[1]
    else:
       timestamp = '_'+basename.split('_')[1].split('-')[1]
    outfile = dophot_dir+'matched_'+target+timestamp+tag

    if outfile in glob.glob(dophot_dir+'matched_*') and redo == False:
        #print "Matched catalogue already exist! Exiting..."
        return pd.read_csv(outfile)
    else:
    
        if dophotoutput == None:
            obj_out = pd.read_table(dophot_dir+'obj_out_'+target+timestamp+tag,names = ['i','type','xpos','ypos','fmag','dfmag','flux','dflux','sky','peakflux','sigx','sigxy','sigy','FWHM1','FWHM2','tilt','ext.','chin'],sep = "\s+|\t+|\s+\t+|\t+\s+",skiprows = 11)
            if vett ==False:
                dophotoutput = obj_out
            else:
                dophotoutput = obj_out[obj_out['type']=='0x01']

        panstarrs_df = get_panstarrs_refs(target, thefile, radius = 10.5,equinox='J2000',coordformat='sex',maxMag=20.,maxStd=5.,maxErr=0.5,redo=redo_panstarrs,tag=tag,SN=SN)
        #panstarrs_query_sorted(ra=c.ra.hourangle,dec=c.dec.deg,maxMag=20.,maxStd=5.,maxErr=0.5)
        xpixs = []
        ypixs = []
        for i in range(len(panstarrs_df)):
            c = SkyCoord(panstarrs_df['raMean'].loc[i]+' '+panstarrs_df['decMean'].loc[i], unit=(u.hourangle, u.deg))
            px, py = w.wcs_world2pix(c.ra.deg, c.dec.deg, 0)
            xpixs.append(px)
            ypixs.append(py)

        panstarrs_df['px'] = xpixs
        panstarrs_df['py'] = ypixs
                #plt.scatter(px, py, facecolors = 'none',edgecolors='red' )
        matched_df = pd.DataFrame(columns = dophotoutput.keys().values)
        r = search_rad
        objName = []
        rMeanApMag = []
        rMeanApMagErr = []
        rMeanApMagStd = []

        for s in tqdm(panstarrs_df.index):
            for t in dophotoutput.index:
                if dophotoutput['xpos'][t] < panstarrs_df['px'][s]+r and dophotoutput['xpos'][t] > panstarrs_df['px'][s]-r and dophotoutput['ypos'][t] < panstarrs_df['py'][s]+r and dophotoutput['ypos'][t] > panstarrs_df['py'][s]-r:
                    matched_df.loc[len(matched_df)+1]=dophotoutput.loc[t] 
                    rMeanApMag.append(panstarrs_df['rMeanApMag'].loc[s])
                    rMeanApMagErr.append(panstarrs_df['rMeanApMagErr'].loc[s])
                    rMeanApMagStd.append(panstarrs_df['rMeanApMagStd'].loc[s])
                    objName.append(panstarrs_df['objName'].loc[s])
        matched_df['rMeanApMag'] = rMeanApMag
        matched_df['rMeanApMagErr'] = rMeanApMagErr
        matched_df['rMeanApMagStd'] = rMeanApMagStd
        matched_df['objName'] = objName
        #matched_df['aperture_flux'] = matched_df['aperture_flux'].astype(float)
        matched_df = matched_df[matched_df['flux'] > 0]    
        if save:
            matched_df.to_csv(outfile)
    return matched_df


##############################
#Show image
#(This should be solid too)
##############################
def show(im,figure=None,title='image', sighi = 5, siglo = 5, png=False, pngname=None,show=False):
    """
    Quick visualization function
    """
    if type(im)== str:
        im = fits.getdata(im,ignore_missing_end=True)
    ysz,xsz = im.shape
    aspect = np.float(xsz)/np.float(ysz)
    plt.figure(figure,figsize=(8*aspect*1.2,5))
    plt.clf()
    sig = rb.std(im)
    med = np.median(im)
    vmin = med - siglo*sig
    vmax = med + sighi*sig
    plt.imshow(im,vmin=vmin,vmax=vmax,cmap='gray',interpolation='nearest',origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("pixel number")
    plt.ylabel("pixel number")
    if png:
        plt.savefig(pngname,dpi=300,overwrite = True)
    if show:
        plt.show()
    return


def get_panstarrs_refs(target,thefile = None,tag=None,radius = 10.5,equinox='J2000',coordformat='sex',maxMag=20.,maxStd=5.,maxErr=0.5,redo=False,SN=False):

    if thefile == None:
        thefile = glob.glob('/home/student/SNS/templates/SNST_'+target+'*.fit')
    else:
        thefile = glob.glob(thefile)
    thefile = thefile[0]
    template_name = thefile.split('/')[-1]
    basename = thefile.split('/')[-1]
    if SN:
        timestamp = '_'+basename.split('-')[1]
    else:
       timestamp = '_'+basename.split('_')[1].split('-')[1]
    outfile = '/home/student/SNS/templates/stacked/panstarrs/'+target+timestamp+tag+'_data_get.table'
    if len(glob.glob(outfile)) != 0 and redo ==False:
        panstarrs_df = pd.read_table(outfile,sep=',',index_col=0,skiprows=[1])
        return panstarrs_df
    else:
        im, h = fits.getdata(thefile,header=True)
        #target = imfile.split('/')[-1].split('_')[0]
        c = SkyCoord(h['RA']+' '+h['Dec'], unit=(u.hourangle, u.deg))
        df = panstarrs_query_sorted(ra=c.ra.hourangle,dec=c.dec.deg,radius=radius,equinox=equinox, maxStd=maxStd,maxErr=maxErr, maxMag=maxMag)
        #(ra=None,dec=None,radius=10.5, equinox='J2000',coordformat='sex',maxMag=10,maxStd=0.01,maxErr=0.05,verbose=True):
        print df
        df.to_csv(outfile)
        panstarrs_df = pd.read_table(outfile,sep=',',index_col=0,skiprows=[1])
        return panstarrs_df


def show_template(target):
    templates = glob.glob('/home/student/SNS/templates/SNST_'+target+'*.fit')
    if len(templates) == 0:
        print 'No templates exist for this target!'
        return
    for t in templates:
        template_name = t.split('/')[-1]
        im,h = fits.getdata(t,header=True)
        show(im,title=template_name)
        return



#photometry on reference stars --> subtraction --> photometry of SN --> compare
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


def subtract_images(template, science, convolve=True, photometric_scale = True,hcongrid=True,save=True):
        template_name = science.split('/')[-1]
        timestamp = '_'+template_name.split('_')[1].split('-')[1]
        try:
            temp_psf = fits.getdata (glob.glob('/home/student/DoPHOT_C/psf_out_'+target+'*_temp.fits')[0])
            sci_psf = fits.getdata (glob.glob('/home/student/DoPHOT_C/psf_out_'+target+timestamp+tag+'.fits')[0])
        except:
            print 'No Templste or Science PSF for ',science,', skipping convolution!'
            convolve = False

        
        #science = glob.glob('/home/student/SNS/'+target+'/*.fit')[0]
        #psfs = glob.glob('/home/student/DoPHOT_C/psf*')
        #psf = fits.getdata('/home/student/DoPHOT_C/psf_out_'+target+'.fits')
        #10,25,2,3
        temp_im, th = fits.getdata(template,header=True)
        sci_im,sh = fits.getdata(science,header=True)

        if convolve:
            """
            CONVOLVE THE IMAGES
            """
            #Fit 2D gaussian to science psf
            nr_sci_psf = (sci_psf/float(np.sum(sci_psf))).ravel()
            x = np.linspace(0, sci_psf.shape[0], sci_psf.shape[0])
            y = np.linspace(0, sci_psf.shape[1], sci_psf.shape[1])
            x, y = np.meshgrid(x, y)
            initial_guess = (1.,32.,32.,5.,5.,0,10.)
            sci_popt, sci_pcov = opt.curve_fit(Gaussian2D, (x, y), nr_sci_psf,p0=initial_guess)
            sci_gauss = Gaussian2D((x, y), *sci_popt)
            sci_sigma = np.sqrt(sci_popt[3]*sci_popt[4])

            #Fit 2D gaussian to template psf
            nr_temp_psf = (temp_psf/float(np.sum(temp_psf))).ravel()
            x = np.linspace(0, temp_psf.shape[0], temp_psf.shape[0])
            y = np.linspace(0, temp_psf.shape[1], temp_psf.shape[1])
            x, y = np.meshgrid(x, y)
            initial_guess = (1.,32.,32.,5.,5.,0,10.)
            temp_popt, temp_pcov = opt.curve_fit(Gaussian2D, (x, y), nr_temp_psf, p0=initial_guess)
            temp_gauss = Gaussian2D((x, y), *temp_popt)
            temp_sigma = np.sqrt(temp_popt[3]*temp_popt[4])

            k = np.sqrt(max(temp_sigma,sci_sigma)**2-min(temp_sigma,sci_sigma)**2)
            d = round(10*k)
            x = np.linspace(0, d,d)
            y = np.linspace(0, d,d)
            x, y = np.meshgrid(x, y)
            kernel = Gaussian2D((x, y),1.,d/2.,d/2.,k,k,0.,0.) #2d gaussian with sigma k, normalized
            kernel = (kernel/np.sum(kernel)).reshape(int(d),int(d))


            if temp_sigma > sci_sigma:
                img = sci_im
            else:
                img = temp_im
            """
            kernel_ft = fftpack.fft2(kernel, shape=img.shape[:2], axes=(0, 1))
            img_ft = fftpack.fft2(img, axes=(0, 1))
            img2_ft = kernel_ft * img_ft
            img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real
            # clip values to range
            convolved_img = np.clip(img2, 0, 1)
            """
            from scipy import signal
            convolved_img =  signal.fftconvolve(img, kernel, mode='same')

            if temp_sigma > sci_sigma:
                sci_im = convolved_img
            else:
                temp_im = convolved_img
        if hcongrid:
            try:
                sci_im = FITS_tools.hcongrid.hcongrid(sci_im,sh,th)
            except:
                print 'Hcongrid Failed! ', science
                return

        if photometric_scale:
            sci_matched_df = match_catalogue(target,sci,w=WCS(template),redo_panstarrs=False,save=True,tag='_sci',redo=True)
            temp_matched_df = match_catalogue(target,template,w=WCS(template),redo_panstarrs=False,save=True,tag='',redo=False)
            sci_mzp = (sci_matched_df['rMeanApMag'].values + 2.5*np.log10(sci_matched_df['aperture_flux'].values/60.))
            temp_mzp = (temp_matched_df['rMeanApMag'].values + 2.5*np.log10(temp_matched_df['aperture_flux'].values/300.))
            try:
                R = 10**(-0.4*(np.mean(temp_mzp)-np.mean(sci_mzp)))
                sci_im = R*sci_im
            except:
                print "Scaling Failed!! ",science
        diff = sci_im/sh['EXPTIME']-temp_im/th['EXPTIME']
        if save:
            outname = '/home/student/diff_dophot/diff'+template_name
            fits.writeto(outname,diff,sh,overwrite=True)
        return diff


