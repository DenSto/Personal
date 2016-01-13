"""

Simulation script for a model that captures the toroidal ITG mode.

QL model

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
shape = (64,64)
Lx, Ly = (10. * 2. * np.pi, 10. * 2. * np.pi)

param = open('param','r')
rho=float(param.readline().split()[0])

logger.info('test %f' %rho)
rho=0.5
ep=0.25
R_L=45 
fmag = 0*5e-1
mu = 1
muZF = 0
nu = 0.00
dt = 1e-3
sim_time=20
mod_print=10
data_dt = sim_time/200.#0.1
linear=False
correlated=True
correct_physics=True



# Initialize some variables
nx, ny = shape
lastF = 0
lastIt =-1

# Random Forcing
def forcing(solv,domain):	
	global lastIt
	global lastF
	if(solv.iteration != lastIt or not correlated):
		global dt
		sh=domain.local_grid_shape()
		#logger.info('Iteration: %d' %solv.iteration)
		lastF = np.random.uniform(-1,1,sh*1.5)/np.sqrt(dt)
		#lastF = np.random.standard_normal(sh*1.5)/np.sqrt(dt)
		lastIt=solv.iteration
	return lastF	
		

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
forcing_func = de.operators.GeneralFunction(domain,'g',forcing, args=[])

# Poisson equation
problem = de.IVP(domain, variables=['phi','w','T', 'UZF','TZF', 'u','v','wx','wy','Tx','Ty'])

# Create Parameters
problem.parameters['L'] = 1/(rho*ep*R_L)
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['ep'] = 2*rho*ep
problem.parameters['mu'] = mu
problem.parameters['nu'] = nu
problem.parameters['fmag'] = fmag
problem.parameters['muZF'] = muZF

problem.parameters['forcing_func'] = forcing_func

# Add equations
if(correct_physics):
    problem.add_equation("dx(dx(phi)) + dy(dy(phi)) - w - phi = 0", condition="(ny != 0)")
    problem.add_equation("dx(dx(phi)) - w = 0", condition="(ny == 0)")
else:
    problem.add_equation("dx(dx(phi)) + dy(dy(phi)) - w - phi = 0")


problem.add_equation("u - dx(phi) = 0")
problem.add_equation("v + dy(phi) = 0")
problem.add_equation("Tx - dx(T) = 0")
problem.add_equation("Ty - dy(T) = 0")
problem.add_equation("wx - dx(w) = 0")
problem.add_equation("wy - dy(w) = 0")

# handle zonal flows differently if needed
if(linear):
	problem.add_equation("dt(w) + ep*Ty  + mu*w - nu*dy(wy) - nu*dx(wx) = fmag*forcing_func", condition="(ny != 0)")
	problem.add_equation("dt(T) - v/L    + mu*T - nu*dy(Ty) - nu*dx(Tx) = fmag*forcing_func", condition="(ny != 0)")
	problem.add_equation("dt(UZF) + muZF*UZF = 0", condition="(ny == 0)")
	problem.add_equation("dt(TZF) + muZF*TZF = 0", condition="(ny == 0)")
else:
	problem.add_equation("dt(w) + ep*Ty  + mu*w - nu*dy(wy) - nu*dx(wx) = -UZF*wy - v*dx(dx(UZF)) + fmag*forcing_func", condition="(ny != 0)")
	problem.add_equation("dt(T) - v/L    + mu*T - nu*dy(Ty) - nu*dx(Tx) = -UZF*Ty - v*dx(TZF) + fmag*forcing_func", condition="(ny != 0)")
	problem.add_equation("dt(UZF) + muZF*UZF = -dx(integ(u*v,'y'))/Ly", condition="(ny == 0)")
	problem.add_equation("dt(TZF) + muZF*TZF = -dx(integ(u*T,'y'))/Ly", condition="(ny == 0)")

# Gauge conditions
problem.add_equation("T = 0", condition="(ny == 0)")
problem.add_equation("phi = 0", condition="(ny == 0)")
problem.add_equation("TZF = 0", condition="(ny != 0)")
problem.add_equation("UZF = 0", condition="(ny != 0)")


# Time stepper
ts= de.timesteppers.RK443 #443


# Build solver
solver = problem.build_solver(ts)
forcing_func.args = [solver,domain]
forcing_func.original_args = [solver,domain]


# Initial conditions
x = domain.grid(0)
y = domain.grid(1)
w = solver.state['w']
T = solver.state['T']
phi = solver.state['phi']
w['g'] = 0
sh=domain.local_grid_shape()
#logger.info('Iteration: %d' %solv.iteration)
w['g'] = 0.01*np.random.uniform(-1,1,sh)/np.sqrt(dt)
T['g'] = 0.01*np.random.uniform(-1,1,sh)/np.sqrt(dt)


# Integration Parameters
solver.stop_sim_time = sim_time
solver.stop_wall_time = np.inf #30 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=data_dt, max_writes=50)
snapshots.add_task("phi",name='phi')
snapshots.add_task("w",name='w')
snapshots.add_task("T",name='T')
snapshots.add_task("TZF",name='TZF')
snapshots.add_task("UZF",name='UZF')
#snapshots.add_system(solver.state)

# ZF data file
snapshot_ZF = solver.evaluator.add_file_handler('ZF_data', sim_dt=data_dt)
snapshot_ZF.add_task("UZF",name='ZF')
snapshot_ZF.add_task("integ(v*T,'x','y')/(Ly*Lx)",name='chi')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=1.5, max_change=1.5, min_change=0.5, max_dt=0.02)
CFL.add_velocities(('u','v'))

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
	plt.savefig('ITG_%s.png' %domain.distributor.size)
