mkdir plots
rm -rf snapshots
rm -rf ZF_data
mpiexec -np 4 python3 myproblem.py | tee /dev/tty > log
./plot_ZF.sh
./plot_phi.sh
