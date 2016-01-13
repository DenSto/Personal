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

from dedalus import public as de
from dedalus.extras import plot_tools


#Create test forcing
def forcing(solver):
	return f(solver.sim_time)

# Create bases and domain
x_basis = de.Fourier('x', 128, interval=(0, 2*np.pi))
y_basis = de.Fourier('y', 128, interval=(0, 2*np.pi))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Poisson equation
problem = de.IVP(domain, variables=['u','forcing_func'])
problem.add_equation("dt(u) = forcing_func")
#problem.add_bc("left(u) = left(sin(8*x))")
#problem.add_bc("right(uy) = 0")

forcing_func = de.operators.GeneralFunction(domain,'g',forcing,args=[])
problem.parameters['forcing_func'] = forcing_func

# Build solver
solver = problem.build_solver()
solver.solve()

# Plot solution
u = solver.state['u']
u.require_grid_space()
plot_tools.plot_bot_2d(u)
plt.savefig('forced.png')
