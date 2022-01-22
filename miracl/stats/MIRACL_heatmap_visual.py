#!/usr/bin/env python

## This script take location of segmented data and plot heatmaps.


import matplotlib
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, show
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image
from nilearn.plotting import plot_anat, show
from nilearn.plotting import plot_stat_map
import sys


if len(sys.argv) < 4:
   print('Wrong syntax!!!')
   print("Usage: python3 MIRACL_heatmap_visual.py <path to input data> <path to the average brain atlas> <path to the annotation file>")
   sys.exit()

input_path=sys.argv[1]
brain_atlas_path=sys.argv[2]
mask_vt_filename = sys.argv[3]



##  ----Initial testing code ---------------------
'''
mean_haxby = mean_img("./data/mo_test1.nii.gz")
plotting.plot_img(mean_haxby)
plt.show()

for smoothing in range(0, 25, 5):
    smoothed_img = image.smooth_img("./data/mo_test1.nii.gz", smoothing)
    plotting.plot_epi(smoothed_img,title="Smoothing %imm" % smoothing)

plt.show()


plotting.plot_img(./mgourbran-projects/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz, output_file="./Con3L.pdf", display_mode='z', cut_coords=3, black_bg=False)
'''

## --------   Code working on new data --------------------------------------------------------------

# mean of data from contolled subjects
#fmri_img = concat_imgs("/data/proj-miracl-dataset/vox_clarity/ctrl/*.nii.gz")
mean_img_ctrl = mean_img(input_path + "/proj-miracl-dataset/vox_clarity/ctrl/*.nii.gz")
smoothed_img_ctrl = image.smooth_img(mean_img_ctrl,  fwhm=0.25)
plot_stat_map(smoothed_img_ctrl, bg_img=brain_atlas_path + "/average_template_25um.nii.gz", output_file="/data/ctrl_sub_mean_plot.png", display_mode='mosaic', threshold='auto', black_bg=False, colorbar=True)


# mean of data from treated subjects
mean_img_trt = mean_img(input_path + "/proj-miracl-dataset/vox_clarity/treated/*.nii.gz")
smoothed_img_trt = image.smooth_img(mean_img_trt,  fwhm=0.25)
plot_stat_map(smoothed_img_trt, bg_img=brain_atlas_path + "/average_template_25um.nii.gz", output_file="/data/treated_sub_mean_plot.png", display_mode='mosaic', threshold='auto', black_bg=False, colorbar=True)


# plotting brain region on top of mean brain atlas
z_slice = -5
# mask_vt_filename = '/data/annotations/annotation_hemi_combined_25um.nii.gz'
fig = plt.figure(figsize=(4, 5.4), facecolor='k')
display = plot_anat('/data/average_template_25um.nii.gz', display_mode='z', figure=fig, black_bg=False)
display.add_contours(mask_vt_filename, contours=1, antialiased=False, linewidths=4., levels=[0], colors=['red'])
display.savefig("/data/brain_region__plot.png")
# show()


##  -------------------Test code for old sample data wild and MO --------------------------------

#fig1 = plt.figure(figsize=(4, 5.4), facecolor='k')
#smoothed_img = image.smooth_img("/data/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz",  fwhm=0.25)
#smoothed_img.to_filename("Con3L-demo-smooth.nii.gz")
#plotting.plot_img("/data/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz", output_file="/data/Con3L.tiff", bg_img="./Con2L-demo-smooth.nii.gz", threshold=0.00175, black_bg=False)
#initial_plot=plotting.plot_img("/data/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz", bg_img="/data/average_template_25um.nii.gz", cut_coords=[-5], display_mode='z', threshold='auto', black_bg=True, colorbar=True, figure=fig1)
#plot_stat_map("/data/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz", bg_img="/data/average_template_25um.nii.gz", display_mode='mosaic', threshold='auto', black_bg=False, colorbar=True)
#show()


'''
## plotting after smoothing the images

# MO
smoothed_img = image.smooth_img("./mgourbran-projects/mo/MO7L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz",  fwhm=0.25)
smoothed_img.to_filename("MO7L-demo-smooth.25.nii.gz")


# WILD
smoothed_img = image.smooth_img("./mgourbran-projects/wild/Con2L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz",  fwhm=0.25)
smoothed_img.to_filename("Con2L-demo-smooth.nii.gz")

# plot

plotting.plot_img("./MO7L-demo-smooth.25.nii.gz", display_mode='ortho', threshold='auto', black_bg=True, colorbar=True)
plotting.plot_img("./Con2L-demo-smooth.nii.gz", display_mode='ortho', threshold='auto', black_bg=True, colorbar=True)

# plotting over a background image
## --- Here the background is the smoothed image.

#plotting.plot_img("./mgourbran-projects/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz", bg_img="./Con2L-demo-smooth.nii.gz", display_mode='ortho', threshold='auto', black_bg=True, colorbar=True)

'''


# plotting over a background image
## --- Here the background is the smoothed image.

#plotting.plot_img("./mgourbran-projects/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz", bg_img="./Con2L-demo-smooth.nii.gz", display_mode='o$

######  plotting superimposed images

'''
z_slice = -5
mask_vt_filename = '/data/annotations/annotation_hemi_combined_25um.nii.gz'
fig = plt.figure(figsize=(4, 5.4), facecolor='k')
display = plot_anat('/data/average_template_25um.nii.gz', display_mode='z', figure=fig, black_bg=False)
#display = plot_anat('/data/wild/Con3L_voxelized_seg_seg_trap2_channel_allen_space.nii.gz', display_mode='z', cut_coords=[z_slice], figure=fig)
#display = plot_anat(initial_plot, display_mode='z', cut_coords=[z_slice], figure=fig)
display.add_contours(mask_vt_filename, contours=1, antialiased=False, linewidths=4., levels=[0], colors=['red'])
#display.add_contours(smoothed_img, contours=1, antialiased=False, linewidths=4., levels=[1], colors=['red'])
show()

'''
