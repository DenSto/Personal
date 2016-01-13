"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.extras import plot_tools


def main(filename, start, count, output):
    integral(filename, start, count, output)

def integral(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['ZF']
    with h5py.File(filename, mode='r') as file:
        time = file['scales']['sim_time']
        xaxis = file['scales']['x']['1.0']
        dset = file['tasks']['ZF']
        dset2 = file['tasks']['chi']
	
        maxtime=time[len(time)-1]
        maxX=xaxis[len(xaxis)-1]
        f = open('zf_set','w')
        f2 = open('zf_par','w')
        f3 = open('chi_set','w')
        for j in range(0,len(time)):
            f3.write('%f %f \n' %(time[j],dset2[j][0]))
            for k in range(0,len(xaxis)):
                 f.write('%f %f %f\n' %( time[j], xaxis[k] ,dset[j][k][0] ))	
            f.write('\n')	
        f2.write('xr=%f\nyr=%f' %(maxtime,maxX))
        f.close()
        f2.close()
        f3.close()


    return


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

