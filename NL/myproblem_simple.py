"""
Simulation script for 2D Poisson equation.

This script is an example of how to do 2D linear boundary value problems.
It is best ran serially, and produces a plot of the solution using the included
plotting helpers.

On a single process, this should take just a few seconds to run.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)

from dedalus import public as de
from dedalus.extras import plot_tools

from mpi4py import MPI


dt = 5.0e-2
simtime = 5
Lx, Ly = (2.*np.pi*1.,2.*np.pi*2.)
nx, ny = (96,96)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)


# Random Forcing

def forcing(sh, solv,domain):	
	arr = np.zeros(sh*1.5)
#	for x in range(0, len(arr)):
#		arr[x][4] = 1
	if(domain.distributor.rank == 0):
		arr[4][2] = 1
#	arr[4][2] = 1
#	return arr
#	return 100*np.random.standard_normal(sh)
	ret = np.random.uniform(-1,1,sh*1.5)/np.sqrt(dt)
	sq = np.square(ret)
	return arr
	#return ret


forcing_func = de.operators.GeneralFunction(domain,'g',forcing, args=[])
logger.info('Iteration: %s' %(forcing_func.layout.local_shape(1.5)))


# Poisson equation
problem = de.IVP(domain, variables=['phi'])

# Create Parameters
problem.parameters['rho'] = 0.05 
problem.parameters['L'] = 1
problem.parameters['ep'] = 0.25
problem.parameters['nu'] = 5
problem.parameters['mu'] = 5

problem.parameters['forcing_func'] = forcing_func

# Add equations
problem.add_equation("dt(phi) + nu*phi = sin(x+3*y)")
#problem.add_equation("dt(w) + 2*ep*rho*dy(T)   + nu*w = -dx(phi)*dy(w) + dy(phi)*dx(w)", condition="(nx != 0) or (ny != 0)")
#problem.add_equation("dt(w) + 2*ep*rho*dy(T)   + nu*w = -dx(phi)*dy(w) + dy(phi)*dx(w)", condition="(nx != 0) or (ny != 0)")
#problem.add_equation("dt(T) + rho*ep*dy(phi)/L + nu*T = -dx(phi)*dy(T) + dy(phi)*dx(T)", condition="(nx != 0) or (ny != 0)")

# Boundary conditions
#problem.add_bc("left(u) = left(sin(8*x))")
#problem.add_bc("right(uy) = 0")
#problem.add_equation("T=0", condition="(nx == 0) and (ny == 0)")
#problem.add_equation("phi=0", condition="(nx == 0) and (ny == 0)")

# Time pepper stepper
ts= de.timesteppers.RK443


# Build solver
solver = problem.build_solver(ts)

forcing_func.args = [forcing_func.layout.local_shape(1),solver,domain]
forcing_func.original_args = [forcing_func.layout.local_shape(1),solver,domain]

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
phi = solver.state['phi']
phi['g'] = 0


# Integration Parameters
solver.stop_sim_time = simtime
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
snapshots.add_system(solver.state)

while solver.ok:
	solver.step(dt)
	if solver.iteration  % 10 == 0:
		logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

# Plot solution
if (domain.distributor.rank == 0 or 1 == 1):
	u = solver.state['phi']
	r=int(96*1.5)
	size = domain.distributor.size
	div = int(r/size)
	u.require_grid_space()
	data=np.zeros(r*div)
	logger.info('div %f size %s' %(div, u.data.shape))
	for x in range(0,r):
		for y in range(0,div):
			data[div*x+ y]=u.data[x][y]
	buf= np.zeros(r*r) 
	domain.distributor.comm_cart.Gather(data,buf)
	if(domain.distributor.rank == 0):
		buf2 = np.zeros((r,r))
		for proc in range(0,size):
			for y in range(0,r):
				for z in range(0,div):
					buf2[y][proc*div + z]=buf[proc*r*div + z + y*div]
		u.data=buf2
		sq=np.square(buf2)
		logger.info('last int val %f' %(sq.sum()/144./144.))
		plot_tools.plot_bot_2d(u)
		plt.savefig('forcing_%s.png' %domain.distributor.size)
	
