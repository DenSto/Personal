rm frames/*.png
mpiexec -np 4 python3 merge.py snapshots/
mpiexec -np 4 python3 plot_2d_series.py snapshots/*.h5
