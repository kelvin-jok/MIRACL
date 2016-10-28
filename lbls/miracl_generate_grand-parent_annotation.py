#!/usr/bin/env python
# Maged Goubran @ 2016, mgoubran@stanford.edu

# coding: utf-8


import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import re
import argparse
from os.path import basename
from datetime import datetime


# ---------
# help fn

def helpmsg(name=None):
    return '''mouse_generate_grand-parent_annotation.py -p [parent level (default: 3)] -m [hemisphere: split or combined (default: combined)] -v [voxel size in um: 10, 25 or 50 (default: 10)]

    Computes features of segmented image and summarizes them per label

    example: mouse_feat_extract.py -p 3 -m split -v 10
    '''


# ---------
# Get input arguments

parser = argparse.ArgumentParser(description='Sample argparse py', usage=helpmsg())
parser.add_argument('-p', '--pl', type=int, help="parent level", required=False)
parser.add_argument('-m', '--hemi', type=str, help="hemisphere mirrored or not", required=False)
parser.add_argument('-v', '--res', type=int, help="voxel size in um", required=False)

args = parser.parse_args()
pl = args.pl
hemi = args.hemi
res = args.res

# check if pars given

if args.pl is None:
    pl = 3
    print("parent level not specified ... choosing default value of %d" % pl)

if args.hemi is None:
    hemi = "combined"
    print("hemisphere not specified ... choosing default value of %s" % hemi)

if args.res is None:
    res = 25
    print("voxel size not specified ... choosing default value of %dum" % res)

# --- Init pars ---

lblsplit = 20000  # number added to contra side
maxannotlbl = 13000  # > max lbl in ipsi

# ------------


def getalllbls(data):

    # get unique lbls
    lbls = np.unique(list(data))
    lbls = lbls[lbls > 0]  # discard negative lbls

    return lbls


def getlblparent(lbls, clarinfo, lbl, pl, lblsplit, maxannotlbl):

    # path id    
    path = clarinfo.structure_id_path[clarinfo.id == lbl]

    # remove /
    numpath = re.sub('[/]', ' ', str(path))

    # get digits
    digpath = [int(s) for s in numpath.split() if s.isdigit()]
    digpath = digpath[1:]  # drop 1st index

    # get parent
    if len(path) == 0:
        parent = lbl
    elif len(digpath) < pl:
        parent = digpath[0]
    else:
        parent = digpath[-pl]

    if np.max(lbls) > lblsplit:
        parent = parent + lblsplit if lbl > maxannotlbl else parent

    return parent


def saveniiparents(parentdata, vx, outnii):

    # save parent data as nifti
    mat = np.eye(4)
    mat[0, 0] = vx
    mat[1, 1] = vx
    mat[2, 2] = vx

    # Create nifti
    nii = nib.Nifti1Image(parentdata, mat)

    # nifti header info
    nii.header.set_data_dtype(np.float32)
    nib.save(nii, outnii)


def main():

    startTime = datetime.now()

    # load annotations
    print("Reading ABA annotation with %s hemispheres and %d voxel size" % (hemi, res))
    nii = '/Users/mgoubran/workspace/clarity_project/aba/annotation/annotation_hemi_%s_%dum.nii.gz' % (hemi, res)
    img = nib.load(nii)
    data = img.get_data()

    # load structure graph
    print("Reading ABA ontology structure_graph")
    abastrctcsv = "/Users/mgoubran/workspace/clarity_project/aba/aba_mouse_structure_graph_hemi_combined.csv"
    abagraph = pd.read_csv(abastrctcsv)

    # get lbls
    lbls = getalllbls(data)

    # loop over intensities
    parentdata = data

    print("Computing parent labels at parent-level/generation %d" % pl)

    for l in range(len(lbls) - 1):
        lbl = lbls[l]
        parent = getlblparent(lbls, abagraph, lbl, pl, lblsplit, maxannotlbl)
        # replace val
        parentdata[parentdata == lbl] = parent

    vx = img.header.get_zooms()[0]
    orgname = basename(nii).split('.')[0]
    outnii = '%s_parent-level_%s.nii.gz' % (orgname, pl)
    saveniiparents(parentdata, vx, outnii)

    #orient

    print ("\n Grand-parent labels generation done in %s ... Have a good day!\n" % (datetime.now() - startTime))

if __name__ == "__main__":
    main()