set xrange [0.2:0.4]
set yrange [-0.15:0.01]
set tics font "Arial, 20"
set lmargin at screen 0.1
set rmargin at screen 0.95
data_12="20210924_130948_pt_n12_d48_a2_r1000.tsv"
data_16="20210922_023619_pt_n16_d64_a2_r1000.tsv"
plot data_12 using 1:2:($3/sqrt(1000)) with errorlines notitle, data_16 using 1:2:($3/sqrt(1000)) with errorlines notitle
pause -1
