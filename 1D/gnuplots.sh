#!/bin/bash
FILE=tasks_1D
while read line;do
	echo 'Plotting '${line}
	gnuplot ${line}_par plot1d.mac
done < $FILE

FILE=tasks_2D
while read line;do
	echo 'Plotting '${line}
	gnuplot ${line}_par plot2d.mac
done < $FILE
mv *.ps plots/
mv *.png plots/
