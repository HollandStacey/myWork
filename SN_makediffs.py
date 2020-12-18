import phot_pipe as pp
import SN_reduction_script as sn
from recursive_glob import recursive_glob
from quick_image import *
from tqdm import tqdm
from astropy.stats import mad_std
import glob,os
import fnmatch


def mkalldiffs(mv=False,dirname='old',divs=1,div=0):

    '''
    Finds all galaxies with template images. Then, for each image with one or more observations
    after the template, it creates a difference image for each night. All difference images are
    made by comparing the latest image with a specified initial template.

    Turn mv to True if you would like to move all old files to a new directory, then specify a
    directory name.

    If you would like to spread this effort over multiple cores, enter how many divisions you
    would like to make and which section you would like to allocate the current loop to, starting
    at 0.
    '''

    if mv:
        dir = '/home/student/SNS/'+dirname+'/'
        os.system('mkdir /home/student/SNS/'+dirname)
        os.system('mv *GC* '+dir)
        os.system('mv 2MASX* '+dir)

    gals = sn.gals_with_template_images()

    perloop = len(gals)/divs
    loopstart = perloop*div
    loopend = perloop*(div+1)
    if (loopend>=len(gals)):

	loopend = len(gals)+1

    pbar = tqdm(desc = 'Making difference images', total = len(gals), unit = 'galaxies')

    for g in gals[loopstart:loopend]:

        try:

            paths = pp.get_paths(targname=g,obstype='SNS')

            files,fct = pp.get_files(prefix='SNS_'+g)
            files.sort()

            raw_temp = glob.glob('/home/student/SNS/reference_images/*'+g+'*.fits')[0]
            temp_im,th = read_image(raw_temp)

            date = raw_temp.split('/')[-1].split('_')[0]
            try:
                i = files.index(fnmatch.filter(files,'*'+date+'*')[0])

            except:
                e = open('/home/student/mo/diff_errors2.txt','a+')
                i = -1
                e.write('There was an error with the template image for '+g+'\n')
                e.close()

            if len(files[i+1:]) >= 1:

                for f in files[i+1:]:

                    sci_date = f.split('/')[-2]
                    outpath = paths['output']+sci_date+'/'+g+'_'+sci_date

                    scifile = glob.glob(outpath+'_science.fits')
                    if len(scifile) == 1:
                        os.system('rm '+scifile[0])
                    sci_raw,sh = read_image(f)
                    sci_cals = pp.get_cal_frames(date=sci_date,targname=g,band='r',
                                                 obstype='SNS',write=False)
                    sci_im = pp.calibrate_image(image=sci_raw,header=sh,rotated_flat=False,
                                                cals=sci_cals)

                    difffile = glob.glob(outpath+'_difference.fits')
                    if len(difffile) == 1:
                        os.system('rm '+difffile[0])
                    diff_im,dh = sn.difference_image(temp_im=temp_im,th=th,sci_im=sci_im,sh=sh)

                    print('Saving science and difference images from '+
                          g+' on '+sci_date+' as fits files')
                    fits.writeto(outpath+'_science.fits',sci_im,sh,overwrite=True)
                    fits.writeto(outpath+'_difference.fits',diff_im,dh,overwrite=True)

                    print('Saving science and difference images as png files')
                    display_image(sci_im,siglo=2,sighi=7,title=g+'_'+sci_date+'_sci')
                    plt.savefig(outpath+'_science.png',dpi=300)
                    plt.clf()
                    display_image(diff_im,siglo=2,sighi=7,title=g+'_'+sci_date+'_diff')
                    plt.savefig(outpath+'_difference.png',dpi=300)
                    plt.clf()

        except:

            h = open('/home/student/mo/diff_errors2.txt','a+')
            h.write('Unidentified error with '+g+'\n')
            h.close()

        pbar.update(1)
    pbar.close()

