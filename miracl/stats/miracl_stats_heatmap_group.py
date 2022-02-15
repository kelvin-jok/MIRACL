#!/usr/bin/env python

import fnmatch
import os
import sys
from math import ceil, nan
import numpy as np
import nibabel as nib
import reg_svg

def grp_mean(input_path, brain_template, outdir, x, y, z, percentile):
    '''read input image files, return mean and shape. Calls reg_svg script to generate registration-to-input data check svg animation'''
    sample = 0
    img = 0
    max_val = 0

    for root, dirnames, filenames in os.walk(input_path):
        for filename in fnmatch.filter(filenames, '*.nii.gz'):
            nib_file = nib.load(os.path.join(root, filename))
            data = np.asanyarray(nib_file.dataobj).clip(min=0)

            #smooth and square root image data for svg script to re-scale and account for positive-skewed data in visualization
            smooth_img = gaussian_filter(np.sqrt(data), sigma=(2, 2, 2))

            # check max of default slices (-45 , -30, -15, 0, 15, 30, 45 from center axis index) in each direction to be used for colourmap max value argument '-cr' in reg_svg script
            max_val=max(max_val, np.amax(smooth_img[:,:,list(range(nib_file.shape[2]//2-45,nib_file.shape[2]//2+46,15))]))
            max_val=max(max_val, np.amax(smooth_img[list(range(nib_file.shape[0]//2-45,nib_file.shape[0]//2+46,15)),:,:]))
            max_val=max(max_val, np.amax(smooth_img[:,list(range(nib_file.shape[1]//2-45,nib_file.shape[1]//2+46,15)), :]))

            smooth_nifti=nib.Nifti1Image(smooth_img, affine=nib_file.affine)
            nib.save(smooth_nifti, outdir +"/smooth_img.nii.gz")
            #send to reg_svg script for registration quality check svg animation
            reg_svg.main(['-f', brain_template, '-r', outdir + "/smooth_img.nii.gz", '-o',"".join((outdir,"/","reg_check_", filename.split(".nii.gz")[0])), '-cr', str(int(max_val*percentile/100)), str(int(max_val))])
            os.remove(outdir + "/smooth_img.nii.gz")

            img = img + data
            sample = sample + 1

    if sample == 0:
        raise Exception(
            '{} is empty or does not contain .nii.gz files ... please check path/file contents and rerun script'.format(
                input_path))

    return (img / sample, np.shape(img))

def mean_nii_export(smoothed_mean_img, outdir, outfile, mask_vt_filename):
    '''export smoothed, masked, mean nii file with annotation metadata'''
    img = nib.load(mask_vt_filename)
    header = img.header
    affine = img.affine
    nii = nib.Nifti1Image(smoothed_mean_img * np.asanyarray(img.dataobj),
                          header=header,
                          affine=affine)
    nib.save(nii, outdir + "/" + outfile + "_mean.nii.gz")

def slice_extract(input_path, cut_coords, x, y, z, atlas):
    '''get image slices, check if slice is blank image. Raise error if blank image'''
    img = nib.load(input_path)
    img = np.asanyarray(img.dataobj)
    slices = []
    blank_slices =""
    if x != -1:
        x_slices = []
        for i in (cut_coords[x]):
            if np.min(img[:, :, i]) != np.max(img[:, :, i]):
                x_slices.append(img[:, :, i])
            else:
                blank_slices="\n".join((blank_slices,"{} AXIS s/sagittal slice {} is blank ".format(atlas, i)))
        slices.append(x_slices)
    if y != -1:
        y_slices = []
        for i in (cut_coords[y]):
            if np.min(img[i, :, :]) != np.max(img[i, :, :]):
                y_slices.append(img[i, :, :])
            else:
                blank_slices="\n".join((blank_slices,"{} AXIS c/coronal slice {} is blank".format(atlas, i)))
        slices.append(y_slices)
    if z != -1:
        z_slices = []
        for i in (cut_coords[z]):
            if np.min(img[:, i, :]) != np.max(img[:, i, :]):
                z_slices.append(img[:, i, :])
            else:
                blank_slices="\n".join((blank_slices, "{} AXIS a/axial slice {} is blank".format(atlas, i)))
        slices.append(z_slices)
    if len(blank_slices)>0:
        raise Exception(blank_slices)
    else:
        return (slices)

def slice_display(atlas, sagittal, coronal, axial, x, y, z):
    '''slicing parameters'''
    img_shape = np.shape(np.asanyarray(nib.load(atlas).dataobj))
    cut_len = max(x, y, z) + 1
    cut_coords = []

    img_shape = [img_shape[2], img_shape[0], img_shape[1]]  # arrange as [L, P, I] order from [P, I, L]
    # default slices
    if type(sagittal[-1]) == float and type(coronal[-1]) == float and type(axial[-1]) == float:
        for i in range(3):
            cut_coords.append(
                list(range(ceil(img_shape[i] / 17 * 3), ceil(img_shape[i] / 17) * 15 + 1, ceil(img_shape[i] / 17 * 2))))
        cut_len = 3
    # custom slices
    else:
        ind = 0
        for axis, tag in zip([sagittal, coronal, axial], ["s/sagittal", "c/coronal", "a/axial"]):
            if type(axis[-1]) != float:
                cut_coords.append(list(range(axis[0], axis[0] + (axis[2] - 1) * axis[1] + 1, axis[1])))
                print("AXIS {} slices {}".format(tag, cut_coords[-1]))
                if max(cut_coords[-1]) > img_shape[ind]:
                    raise Exception("AXIS {} slice {}, is greater than max slice size {}".format(tag, max(cut_coords[-1]),img_shape[ind]))
                if min(cut_coords[-1]) < 0:
                    raise Exception("AXIS {} slice {} is below zero".format(tag, min(cut_coords[-1])))
            ind = ind + 1
    return (cut_len, cut_coords)

if len(sys.argv) < 4:
   print('Wrong syntax!!!')
   print("Usage: python3 miracl_stats_heatmap_group.py <path to data group directory> <path to the annotation file> <path to the average brain atlas> <sigma>")
   sys.exit()

#arguments
grp_path=sys.argv[1]
# retrieve Atlas paths
mask=sys.argv[2]
brain_template=sys.argv[3]
sigma=sys.argv[4]
percentile=10
outdir=os.getcwd()

#slice positions and planes
x=0
y=1
z=2
sagittal=[nan] #[start_slice slice_increment number_of_slices number_of_rows number_of_columns]
coronal=[nan]
axial[nan]

#call functions

# extract Atlas slices for background and outline
cut_len, cut_coords = slice_display(mask, sagittal, coronal, axial, x, y, z)
mask_slices = slice_extract(mask, cut_coords, x, y, z, mask.split("/")[-1])
temp_slices = slice_extract(brain_template, cut_coords, x, y, z, brain_template.split("/")[-1])
img1= grp_mean(grp_path, brain_template, outdir, x, y, z, percentile)
smooth_img1=gaussian_filter(img, sigma=(sigma, sigma, sigma))
mean_nii_export(img1, outdir, "group1", mask)
