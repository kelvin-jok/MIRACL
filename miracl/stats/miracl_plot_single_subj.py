"""
Created on Mon Jul 19 16:50:17 2021

@author: mgoubran
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import scipy.stats
from miracl.stats import stats_gui_plot_single_subj

# inputs
# csv file[s]
# sorting variable (must be in the table)
# threshold (default 0.75)

def helpmsg():
    return '''Usage: miracl stats plot_subj 

    Plots Allen label stats of input volume 

    A GUI will open to choose your:

        - < input volume > 

        - < registered Allen labels >

    ----------

    For command-line / scripting

    Usage: miracl_plot_single_subj.py -i [input volume] -l [reg Allen labels] -o [ out csv ]

    Example: miracl_plot_single_subj.py -i sta_streamlines_density_stats_depth_6_angle_35.csv -s Count -l 0.65

    Arguments (required):

        -i Input volume

    Optional arguments:

	    -d Directory Output 

        -s Sort values by, options are:

            Mean or StdD or Max or Min or Count or Vol(mm^3)

            Mean -> mean intensity values
            Count -> number of voxels

        -th  threshold


        '''

    # Dependencies:

    #     ImageMaths (ANTs)
    #     Python 2.7


def parsefn():
    if len(sys.argv) >= 3 and sys.argv[-2] == 'stats' and sys.argv[-1] == 'plot_single':
        parser = argparse.ArgumentParser(description='', usage=helpmsg())
    else:
        parser = argparse.ArgumentParser(description='', usage=helpmsg())

        parser.add_argument('-i', '--in_csv', type=str, help="CSV file (generated from `miracl lbls stats`")
        parser.add_argument('-s', '--sort', type=str, help="Value to sort data by", default='Mean')
        parser.add_argument('-th', '--threshold', type=float, help="Quantile threshold", default=0.75)
        parser.add_argument('-d', '--dir_outfile', type=str, help="Directory Output", default=os.getcwd())

        return parser


def parse_inputs(parser, args):
    if sys.argv[-2] == 'stats' and sys.argv[-1] == 'plot_subj':

        print("Running in GUI mode")

        # pass the results of the gui here
        args = stats_gui_plot_single_subj.main()

        if hasattr(args,"run")==False:
            print("Plot Single Subject GUI Window was closed")
            sys.exit()  

    else:
        if isinstance(args, list):
            args, unknown = parser.parse_known_args()
    
        print("\n running in script mode \n")

    try:
        input_csv = args.in_csv
        sort = args.sort
        threshold = args.threshold
        output_dir = args.dir_outfile
    except NameError as err:
        print(err)

    return input_csv, sort, threshold, output_dir

def generate_plot(input_csv, sort, threshold, output_dir):
    """
    """
    # read csv, sort by value, drop quantile
    in_df = pd.read_csv(input_csv)
    in_df_sort = in_df.sort_values([sort], ascending=False)
    in_df_sort_thresh = in_df[in_df[sort] >= in_df_sort[sort].quantile(threshold)]

    # set output filename
    filename_no_ext = input_csv.split('.csv')[0]
    
    # generate two plots, based on whether or not an acronym should be used
    sns.set(font_scale=1.2)
    f = sns.barplot(sort,"acronym",data=in_df_sort_thresh, palette="Blues_r")
    f.set(xlabel=f'{sort}', ylabel='Region acronym')
    plt.tight_layout()
    f.figure.savefig('{}/{}_barplot_acronyms.png'.format(output_dir, filename_no_ext.rsplit('/', 1)[-1]), dpi=300)
    print('{}/{}_barplot_acronyms.png'.format(output_dir, filename_no_ext.rsplit('/', 1)[-1]))
    plt.figure(figsize=(20,20))
    f = sns.barplot(sort,"name",data=in_df_sort_thresh, palette="Blues_r")
    f.set(xlabel=f'{sort}', ylabel='Region')
    plt.yticks(rotation=30)
    plt.tight_layout()
    f.figure.savefig('{}/{}_barplot.png'.format(output_dir, filename_no_ext.rsplit('/', 1)[-1]), dpi=300)


def main(args):
    parser = parsefn()
    input_csv, sort, threshold, output_dir = parse_inputs(parser, args)

    # run sta tract generation
    print('Generating plot for single subject\n')
    generate_plot(input_csv, sort, threshold, output_dir)


if __name__ == "__main__":
    main(sys.argv)
