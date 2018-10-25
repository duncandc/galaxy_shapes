# Parameter Files

This directory contains "parameter" files of the form: `calibObj-[RUN]-[CAMCOL].fits`.  There are 2535 files, which take up ~ 6.3G of space in total.

The files can be downloaded using the `./download_all_files.sh` script, which downloads the files listed in `file_list.txt`.

The filenames are of the form:  `calibObj-[RUN]-[CAMCOL].fits`.

Each object in the NYU VGAC appears in one of these files.  An object can be accessed by opening the file that matches an object's RUN and CAMCOL information in the `object_sdss_imaging.fits` file.  The row in the associated calibObj file is given by the CALIBOBJ_POSITION column in `object_sdss_imaging.fits`. 
