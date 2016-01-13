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
    tasks2D = ['ZF','RS','phix']
    tasks2D_PS = ['phix_ft']
    tasks1D = ['Wsq','Cov','ZFmag','chi']

    ftasks1d = open('tasks_1D','w')
    ftasks2d = open('tasks_2D','w')


    with h5py.File(filename, mode='r') as file:
        time = file['scales']['sim_time']
        xaxis = file['scales']['x']['1.0']
        yaxis = file['scales']['y']['1.0']
        maxtime=time[len(time)-1]
        maxX=xaxis[len(xaxis)-1]
        for task in tasks1D:
            ftasks1d.write("%s\n" %task)
            dset = file['tasks'][task]
            f = open('%s_set' %task,'w')
            fpar = open('%s_par' %task,'w')
            for j in range(0,len(time)):
                f.write('%f %f \n' %(time[j],dset[j][0]))
            fpar.write("task='%s'\n" %task)
            fpar.write("file='%s_set'\n" %task)
            fpar.write('xr=%f\n' %maxtime)
            fpar.write("set ylabel '%s'\n" %task)
            f.close()
            fpar.close()

        for task in tasks2D:
            label=''
            x0=0
            y0=0
            ftasks2d.write("%s\n" %task)
            dset = file['tasks'][task]
            if(len(dset[0][0]) > 1): 
                y0=1
                label='y'
            else:
                x0=1
                label='x'
            f = open('%s_set' %task,'w')
            fpar = open('%s_par' %task,'w')
            for j in range(0,len(time)):
                for k in range(0,len(xaxis)):
                    f.write('%f %f %f\n' %( time[j], xaxis[k] ,dset[j][k*x0][k*y0] ))
                f.write('\n')
            fpar.write("task='%s'\n" %task)
            fpar.write("set ylabel '%s'\n" %label)
            fpar.write("file='%s_set'\n" %task)
            fpar.write('xr=%f\nyr=%f' %(maxtime,maxX))
            f.close()
            fpar.close()

        for task in tasks2D_PS:
            kxaxis = file['scales']['kx']
            kyaxis = file['scales']['ky']
            lx=len(kxaxis)
            ly=len(kyaxis)
            label=''
            x0=0
            y0=0
            ftasks2d.write("%s\n" %task)
            dset = file['tasks'][task]
            if(len(dset[0][0]) > 1): 
                y0=1
                label='ky'
            else:
                x0=1
                label='kx'
            f = open('%s_set' %task,'w')
            fpar = open('%s_par' %task,'w')
            for j in range(0,len(time)):
                for k in range(0,x0*lx + y0*ly):
                    temp=dset[j][((k+lx/2+1) % lx)*x0][((k+ly/2+1) % ly)*y0]
                    val=np.log10(temp*np.conj(temp))
                    f.write('%f %f %f\n' %( time[j], kxaxis[(k+lx/2+1)%lx] if x0 == 1 else kyaxis[(k+ly/2+1)%ly], val))
                f.write('\n')
            fpar.write("task='%s'\n" %task)
            fpar.write("set ylabel '%s'\n" %label)
            fpar.write("file='%s_set'\n" %task)
            fpar.write('xr=%f\n' %(maxtime))
            f.close()
            fpar.close()

    ftasks1d.close()
    ftasks2d.close()

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

