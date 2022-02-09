#!/usr/bin/env python

import fnmatch
import os
import sys
import numpy as np
import nibabel as nib
def grp_mean(input_path):
    '''read input image files, return mean'''
    sample = 0
    img = 0

    for root, dirnames, filenames in os.walk(input_path):
        for filename in fnmatch.filter(filenames, '*.nii.gz'):
            nib_file = nib.load(os.path.join(root, filename))
            data = np.asanyarray(nib_file.dataobj).clip(min=0)
            img = img + data
            sample = sample + 1

    if sample == 0:
        raise Exception(
            '{} is empty or does not contain .nii.gz files ... please check path/file contents and rerun script'.format(
                input_path))

    return (img / sample)

def mean_nii_export(smoothed_mean_img, outdir, outfile, mask_vt_filename):
    '''export smoothed, masked, mean nii file with annotation metadata'''
    img = nib.load(mask_vt_filename)
    header = img.header
    affine = img.affine
    nii = nib.Nifti1Image(smoothed_mean_img * np.asanyarray(img.dataobj),
                          header=header,
                          affine=affine)
    nib.save(nii, outdir + "/" + outfile + "_mean.nii.gz")

if len(sys.argv) < 3:
   print('Wrong syntax!!!')
   print("Usage: python3 miracl_stats_heatmap_group.py <path to data group directory> <path to the annotation file> <sigma>")
   sys.exit()

#arguments
grp_path=sys.argv[1]
annotation_path=sys.argv[2]
sigma=sys.argv[3]
outdir=os.getcwd()

#call functions
img1= grp_mean(grp_path)
smooth_img1=gaussian_filter(img, sigma=(sigma, sigma, sigma))
mean_nii_export(img1, outdir, "group1", annotation_path)
