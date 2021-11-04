# set xrange [0.2:0.4]
set xrange [0.4:0.6]
set yrange [:0.01]
# plot ARG1 with errorlines title ARG1, ARG2 with errorlines title ARG2
plot ARG1 using 1:2:($3/sqrt(1000)) with errorlines title ARG1, ARG2 using 1:2:($3/sqrt(1000)) with errorlines title ARG2
pause -1
