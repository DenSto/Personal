rm *.ps
rm *.png
mpiexec -np 4 python3 merge.py ZF_data/
mpiexec -np 4 python3 plot_ZF.py ZF_data/*.h5
./gnuplots.sh
