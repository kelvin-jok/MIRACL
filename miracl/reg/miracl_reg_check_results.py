#!/usr/bin/env python
# Maged Goubran @ 2016, mgoubran@stanford.edu 

# coding: utf-8

import argparse
#import commands
import os
import subprocess
import sys

from miracl import ATLAS_DIR

def helpmsg():
    return '''Usage: miracl_reg_check_results.py 

Checks registration results

    A GUI will open to choose your:

        - < reg final dir > : directory with final registration volumes and labels

    ----------

    For command-line / scripting

    Usage: miracl reg check_reg -f [reg final folder] -v [visualization software] -s [reg space (clarity or
    allen)]

    Example: miracl reg check_reg -f reg_final -v itk -s clarity

    Arguments (required):

        -f Input final registration folder

    Optional arguments:

        -m Hemisphere (split or combined)
        -v Visualization software: itkSNAP 'itk' (default) or freeview 'free'
        -s Registration Space of results: clarity (default) or allen

        '''


# Dependencies:
#
#    Python 2.7, itksnap or freeview

def parsefn():
    if len(sys.argv) >= 3 and sys.argv[-2] == 'reg' and sys.argv[-1] == 'check':
        parser = argparse.ArgumentParser(description='', usage=helpmsg())

    else:
        parser = argparse.ArgumentParser(description='', usage=helpmsg())
        # check if pars given
        parser.add_argument('-f', '--folder', type=str, help="reg final folder", required=True)
        parser.add_argument('-v', '--viz', type=str, help="Visualization software")
        parser.add_argument('-s', '--space', type=str, help="Registration Space")
        parser.add_argument('-m', '--hemi', type=str, help="Hemisphere (split or combined)")

    return parser


def parse_inputs(parser, args):
    if isinstance(args, list):
        args, unknown = parser.parse_known_args()

    if sys.argv[-2] == 'reg' and sys.argv[-1] == 'check':
        print("Running in GUI mode")

        miracl_home = os.environ['MIRACL_HOME']

        indirstr = subprocess.check_output(
            '%s/conv/miracl_conv_gui_options.py -t "Registration Check Options" -d "%s" -f "%s" "%s" -c "%s" "%s" "%s"' % (
            miracl_home, 'Final registration directory', 'Registration Space (def = clarity)', 'Hemisphere (def = combined)', 
            'Visualization Software', 'itk', 'freeview'),
            shell=True,
            stderr=subprocess.PIPE)

        # put args in args dict iff the user has input a value
        gui_sts = indirstr.decode()
        gui_sts = gui_sts.split('\n')
        gui_args = {}
        for st in gui_sts:
            if st:
                st_split = st.split(':')
                if st_split[1]:
                    gui_args[st_split[0]] = st_split[1]

        if 'Final registration directory ' not in gui_args:
            sys.exit("Please provide location for final registration directory")
        else:
            indir = gui_args['Final registration directory '][1:]

        space = gui_args.get('Registration Space (def = clarity) ', 'clarity')
        hemi = gui_args.get('Hemisphere (def = combined) ', 'combined')
        viz = gui_args.get('Visualization Software ', 'itk')

    else:
        print("\n running in script mode \n")

        assert isinstance(args.folder, str)
        indir = args.folder

        if not os.path.exists(indir):
            sys.exit('%s does not exist ... please check path and rerun script' % indir)

        if args.viz is None:
            viz = 'itk'
            print("\n software not specified ... choosing itkSNAP")
        else:
            assert isinstance(args.viz, str)
            viz = args.viz

        if args.space is None:
            space = 'clarity'
            print("\n registration space not specified ... choosing clarity space")
        else:
            assert isinstance(args.space, str)
            space = args.space

        if args.hemi is None:
            hemi = 'combined'
            print("\n hemisphere not specified ... choosing combined")
        else:
            assert isinstance(args.hemi, str)
            hemi = args.hemi

    return indir, viz, space, hemi


# ---------

def main(args):
    parser = parsefn()
    indir, viz, space, hemi = parse_inputs(parser, args)

    snaplut = os.path.join(ATLAS_DIR, "ara/ara_snaplabels_lut.txt")

    if viz == "itk":
        # check for itk
        status, result = subprocess.getstatusoutput("which itksnap")

        if status == 256:
            print('\n itkSNAP is not installed or not in your path! \n')
            sys.exit()

        if space == "clarity":

            print("\n Viewing downsampled CLARITY volume with registered Allen labels using itkSNAP ...\n")

            subprocess.check_call(
                'itksnap -g %s/clar_downsample_res??um.nii.gz -s %s/annotation_hemi_%s_??um_clar_downsample.nii.gz -l %s' % (
                    indir, indir, hemi, snaplut), shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)

        else:

            # get res
            # name = str(glob.glob("%s/clar_downsample_res??um.nii.gz" % indir))
            # res = int(filter(str.isdigit, name))
            res = 10

            print("\n Viewing registered CLARITY volume in Allen space with labels using itkSNAP ...\n")

            subprocess.check_call(
                'itksnap -g %s/clar_allen_space.nii.gz -o $allen%d -s $lbls%d -l %s' % (indir, res, res, snaplut),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)

    else:

        status, result = subprocess.getstatusoutput("which freeview")

        if status == 256:
            print('\n Freeview (from FREESURFER) is not installed or not in your path! \n')
            sys.exit()

        if space == "clarity":

            print("\n Viewing downsampled CLARITY volume with registered Allen labels using Freeview ...\n")

            # w lut not working
            # 'freeview -v %s/clar_downsample_res??um.nii.gz -v %s/annotation_hemi_combined_??um_clar_downsample.nii.gz:lut=$freelut'

            subprocess.check_call(
                'freeview -v %s/clar_downsample_res??um.nii.gz -v %s/annotation_hemi_%s_??um_clar_downsample.nii.gz'
                % (indir, indir, hemi), shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)

        else:

            print("\n Viewing registered CLARITY volume in Allen space with labels using Freeview ...\n")

            subprocess.check_call(
                'freeview -v %s/clar_allen_space.nii.gz -v $allen25 -v $lbls25' % indir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)


if __name__ == "__main__":
    main(sys.argv)
