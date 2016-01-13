"""

Simulation script for a model that captures the toroidal ITG mode.

It is a special case of the Busse annulus. 

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time

import logging

logger = logging.getLogger(__name__)

from dedalus import public as de
from dedalus.extras import plot_tools
from dedalus.extras import flow_tools

from mpi4py import MPI


# Simulation parameters
shape = (128,128)
Lx, Ly = (10. * 2. * np.pi, 10. * 2. * np.pi)

a=1
k=1
fmag = 0
muZF = 0
nu = 1e-4
dt = 1e-3
sim_time=500
mod_print=10
data_dt =  sim_time/200.#0.1 #either print out 100 data files
data_iter = np.inf# or if things go slow, ever 100 iteration
linear=False
correlated=True


logger.info('Shape:%s   Lx:%.2f   Ly:%.2f' %(shape,Lx,Ly))
logger.info('a:%.2e   k:%.5e  muZF:%.5e' %(a,k,muZF))
logger.info('nu:%.2e   dt:%.2e   sim_time:%.d' %(nu,dt,sim_time))
logger.info('Linear:%s  Correlated:%s' %(linear,correlated))

# Initialize some variables
nx, ny = shape
lastF = 0
lastIt =-1
seed = np.random.randint(0,high=2**32-1)
logger.info('Simulation random seed: %d' %seed)
np.random.seed([seed])

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Poisson equation
problem = de.IVP(domain, variables=['phi','u','v','w','n','wx','wy','nx0','ny0','Lw','Ln','phi1','u1','v1','w1','n1','wx1','wy1','nx1','ny1','Lw1','Ln1'])

# Create Parameters
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['k'] = k
problem.parameters['a'] = a
problem.parameters['nu'] = nu
problem.parameters['muZF'] = muZF


# Add equations
problem.add_equation("dx(dx(phi)) + dy(dy(phi)) - w = 0")
problem.add_equation("dx(dx(phi1)) + dy(dy(phi1)) - w1 = 0")


problem.add_equation("u - dx(phi) = 0")
problem.add_equation("v + dy(phi) = 0")
problem.add_equation("wx - dx(w) = 0")
problem.add_equation("wy - dy(w) = 0")
problem.add_equation("nx0 - dx(n) = 0")
problem.add_equation("ny0 - dy(n) = 0")
problem.add_equation("dx(wx) + dy(wy) - Lw = 0")
problem.add_equation("dx(nx0) + dy(ny0) - Ln = 0")

problem.add_equation("u1 - dx(phi1) = 0")
problem.add_equation("v1 + dy(phi1) = 0")
problem.add_equation("wx1 - dx(w1) = 0")
problem.add_equation("wy1 - dy(w1) = 0")
problem.add_equation("nx1 - dx(n1) = 0")
problem.add_equation("ny1 - dy(n1) = 0")
problem.add_equation("dx(wx1) + dy(wy1) - Lw1 = 0")
problem.add_equation("dx(nx1) + dy(ny1) - Ln1 = 0")
# handle zonal flows differently if needed
if(linear):
	problem.add_equation("dt(w) + nu*dy(dy(Lw)) + nu*dx(dx(Lw)) - a*(phi - n)       = 0", condition="(ny != 0)")
	problem.add_equation("dt(n) + nu*dy(dy(Ln)) + nu*dx(dx(Ln)) - a*(phi - n) - k*v = 0", condition="(ny != 0)")
	problem.add_equation("dt(w) + muZF*w = 0", condition="(nx != 0) and (ny == 0)")
	problem.add_equation("dt(n) + muZF*n = 0", condition="(nx != 0) and (ny == 0)")
else:
	problem.add_equation("dt(w) + nu*dy(dy(Lw)) + nu*dx(dx(Lw)) - a*(phi - n)       = -u*wy - v*wx0", condition="(ny != 0)")
	problem.add_equation("dt(n) + nu*dy(dy(Ln)) + nu*dx(dx(Ln)) - a*(phi - n) - k*v = -u*ny0 - v*nx0", condition="(ny != 0)")
	problem.add_equation("dt(w) + muZF*w = -u*wy - v*wx", condition="(nx != 0) and (ny == 0)")
	problem.add_equation("dt(n) + muZF*n = -u*ny0 - v*nx0", condition="(nx != 0) and (ny == 0)")
	problem.add_equation("dt(w1) + nu*dy(dy(Lw1)) + nu*dx(dx(Lw1)) - a*(phi1 - n1)        = -u*wy1 - v*wx1", condition="(ny != 0) or (nx != 0)")
	problem.add_equation("dt(n1) + nu*dy(dy(Ln1)) + nu*dx(dx(Ln1)) - a*(phi1 - n1) - k*v1 = -u*ny1 - v*nx1", condition="(ny != 0) or (nx != 0)")

# Gauge conditions
problem.add_equation("phi = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("n = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("phi1 = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("n1 = 0", condition="(nx == 0) and (ny == 0)")


# Time stepper
ts= de.timesteppers.RK443 #443


# Build solver
solver = problem.build_solver(ts)

# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
phi = solver.state['phi']
w = solver.state['w']
n = solver.state['n']
phi1 = solver.state['phi1']
w1 = solver.state['w1']
n1 = solver.state['n1']
sh=domain.local_grid_shape()
phi['g'] = 0.01*np.random.uniform(-1,1,sh)
n['g'] = 0.01*np.random.uniform(-1,1,sh)
q = de.operators.differentiate(phi,x=2) + de.operators.differentiate(phi,y=2) - phi + de.operators.integrate(phi,'y')/Ly
w['c'] = q.evaluate()['c']
phi1['c'] = phi['c']
w1['c'] = w['c']
n1['c'] = n['c']
#phi['g'] = 0 
#w['g'] = 0.01*np.random.uniform(-1,1,sh)/np.sqrt(dt)


# Integration Parameters
solver.stop_sim_time = sim_time
solver.stop_wall_time = np.inf #30 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=data_dt, iter=data_iter, max_writes=50)
snapshots.add_task("phi",name='phi ZF')
snapshots.add_task("w",name='w ZF')
snapshots.add_task("n",name='n ZF')
snapshots.add_task("phi1",name='phi')
snapshots.add_task("w1",name='w')
snapshots.add_task("n1",name='n')
snapshots.add_task("phi",layout='c',name='phi2')
snapshots.add_task("w",layout='c',name='w2')
snapshots.add_task("n",layout='c',name='n2')
#snapshots.add_system(solver.state)

# ZF data file
snapshot_ZF = solver.evaluator.add_file_handler('ZF_data', sim_dt=data_dt, iter=data_iter)
snapshot_ZF.add_task("integ(u,'y')/Ly",name='ZF')
snapshot_ZF.add_task("integ(v,'x')/Lx",name='RS')
snapshot_ZF.add_task("integ(phi,'x')/Lx",layout='c',name='phix_ft')
snapshot_ZF.add_task("integ(phi,'x')/Lx",name='phix')
snapshot_ZF.add_task("integ(w**2,'x','y')/(Ly*Lx)",name='Wsq')
snapshot_ZF.add_task("integ((w-integ(w,'y')/Ly)**2,'x','y')/(Ly*Lx)",name='Cov')
snapshot_ZF.add_task("integ((integ(w,'y')/Ly)**2,'x')/Lx",name='ZFmag')
snapshot_ZF.add_task("integ(v*n,'x','y')/(Ly*Lx)",name='chi')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=1.5, max_change=1.5, min_change=0.5, max_dt=0.02)
CFL.add_velocities(('u1','v1'))

average_time=0.0
try:
	logger.info('Starting loop')
	start_time = time.time()
	last_time = start_time
	while solver.ok and dt > 1e-5:
		dt = CFL.compute_dt()
		solver.step(dt)
		if solver.iteration  % mod_print == 0:
			temp = time.time()
			logger.info('Iteration: %i, Time: %e, dt: %.2e, looptime: %.2f s' %(solver.iteration, solver.sim_time, dt, temp-last_time))
			average_time += temp-last_time 
			last_time = temp
except:
	logger.error('Exception raised, triggering end of main loop.')
	raise
finally:
	end_time = time.time()
	logger.info('Iterations: %i' %solver.iteration)
	logger.info('Sim end time: %f' %solver.sim_time)
	logger.info('Average log time: %.2f sec' %(average_time*mod_print/solver.iteration))
	logger.info('Run time: %.2f sec' %(end_time - start_time))
	logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

# Quickly plot final solution
u = solver.state['phi']
r = int(shape[0]*1.5)
size = domain.distributor.size
div = int(r/size)
u.require_grid_space()
data=np.zeros(r*div)
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
	plot_tools.plot_bot_2d(u)
	plt.savefig('plots/ITG_%s.png' %domain.distributor.size)
