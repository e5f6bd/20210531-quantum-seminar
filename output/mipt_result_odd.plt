set xrange [0.4:0.6]
set yrange [-0.15:0.01]
set tics font "Arial, 20"
set lmargin at screen 0.1
set rmargin at screen 0.95
data_12="tmp_odd_n12.tsv"
data_16="tmp_odd_n16.tsv"
plot data_12 using 1:2:($3/sqrt(1000)) with errorlines notitle, data_16 using 1:2:($3/sqrt(1000)) with errorlines notitle
pause -1
