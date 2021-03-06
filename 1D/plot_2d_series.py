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
#    integral(filename, start, count, output)
    midplane(filename, start, count, output)


def midplane(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['phi','w','phi2','w2']
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'mid_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 2, 2
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call plotting helper (dset axes: [t, x, y, z])
                dset = file['tasks'][task]
                logger.info('dset %s length  %s' %(dset,len(dset.shape)))
                #if (n > 1):
                    #dset = np.log10(np.absolute(dset))
                    #dset = file.create_dataset('log_%d' %n, data=dset) 
 #               image_axes = (2, 1)
 #               data_slices = (index, slice(None), slice(None), 0)
                plot_tools.plot_bot_3d(dset,0, index, axes=axes, title=task, even_scale=True)
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


def integral(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['ZF']
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'ZF_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 1, 1
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)
    with h5py.File(filename, mode='r') as file:
        time = file['scales']['sim_time']
        xaxis = file['scales']['x']['1.0']
        dset = file['tasks']['ZF']
    
        f = open('zf_set','w')
        for j in range(0,len(time)):
            for k in range(0,len(xaxis)):
                 f.write('%f %f %f\n' %( time[j], xaxis[k] ,dset[j][k][0] ))
            f.write('\n')
        f.close()

    return

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call plotting helper (dset axes: [t, x, y, z])
                dset = file['tasks'][task]
                logger.info('dset %s length %s' %(dset,len(dset.shape)))
                image_axes = (1, 1)
                data_slices = (index, slice(None), slice(None), 0)
                plot_tools.plot_bot(dset,(1,2),(slice(None),slice(None)))
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)

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

