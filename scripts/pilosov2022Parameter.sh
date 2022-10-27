#!/bin/bash

# Figs 1 and 2
mud examples -ns --seed 21 --save-path figs comparison 

# Fig 3
mud examples -ns --seed 21 --save-path figs contours 

# Fig 4
mud examples -ns --seed 21 --save-path figs wme-covariance

# Fig 5
echo "{\"A\": [[1, 1]], \"b\": [[0]], \"y\": [[1]], \"mean_i\": [[0.25], [0.25]], \"cov_i\": [[1.0, -0.5], [-0.5, 0.5]], \"cov_o\": [[0.5]], \"alpha\": 1.0}" > fig5-data.json
mud examples -ns --seed 21 --save-path figs contours -f fig5-data.json -p comparison

# Fig 6
mud examples -ns --seed 12 --save-path figs high-dim-linear

# Generate Poisson Dataset by solving fenics.
mud examples --seed 21 --save-path poisson_data poisson-generate 1000 500

# Fig 7
mud examples -ns --seed 21 --save-path figs poisson-solve -p response poisson_data/s1000_n500_d2_res

# Fig 8
mud examples -ns --seed 21 --save-path figs poisson-solve -p qoi poisson_data/s1000_n500_d2_res

# Fig 9
mud examples -ns --seed 21 --save-path poisson-trials poisson_data/s1000_n500_d2_res

# Fig 10 - ADCIRC grid plots - wind speed multiplier and inlet bathymetry
mud examples -ns --seed 21 --save-path figs adcirc-solve data/adcirc-si \
	-p mesh -mv wind_speed_mult_0
mud examples -ns --seed 21 --save-path figs adcirc-solve data/adcirc-si \
	-p mesh -mv DP -mz '[[-72.5, 0.1], [40.85, 0.04]]' -mc -10

# Fig 11 - Full time series with each window of data marked
mud -ns --save-path figs examples --seed 21 adcirc-solve notebooks/adcirc-si.pickle -p full_ts -ly 1.8 \
	-t1 "2018-01-11 01:00:00" -t2 "2018-01-11 07:00:00" -lx "2018-01-10 14:00:00" \
	-t1 "2018-01-04 11:00:00" -t2 "2018-01-04 14:00:00" -lx "2018-01-04 00:00:00" \
	-t1 "2018-01-07 00:00:00" -t2 "2018-01-09 00:00:00" -lx "2018-01-07 20:00:00"

# Fig 12 - Updated distributions for T1 ADCIRC time window using 1 and 2 principal components
mud -ns --save-path figs examples --seed 21 adcirc-solve notebooks/adcirc-si.pickle -p updated_dist \
	-n 1 -t1 "2018-01-11 01:00:00" -t2 "2018-01-11 07:00:00" -p1 65 -p2 1600
mud -ns --save-path figs examples --seed 21 adcirc-solve notebooks/adcirc-si.pickle -p updated_dist \
	-n 2 -t1 "2018-01-11 01:00:00" -t2 "2018-01-11 07:00:00" 

# Fig 13 - Updated distributions for T2 ADCIRC time window using 1 and 2 principal components
mud -ns --save-path figs examples --seed 21 adcirc-solve notebooks/adcirc-si.pickle -p updated_dist \
	-n 1 -t1 "2018-01-04 11:00:00" -t2 "2018-01-04 14:00:00" -p1 40 -p2 4000
mud -ns --save-path figs examples --seed 21 adcirc-solve notebooks/adcirc-si.pickle -p updated_dist \
	-n 2 -t1 "2018-01-04 11:00:00" -t2 "2018-01-04 14:00:00" -p1 35 -p2 10000

# Fig 14 - Updated distributions for T3 ADCIRC time window using 1 and 2 principal components
mud -ns --save-path figs examples --seed 21 adcirc-solve notebooks/adcirc-si.pickle -p updated_dist \
	-n 1 -t1 "2018-01-07 00:00:00" -t2 "2018-01-09 00:00:00" -p1 30 -p2 3000
mud -ns --save-path figs examples --seed 21 adcirc-solve notebooks/adcirc-si.pickle -p updated_dist \
	-n 2 -t1 "2018-01-07 00:00:00" -t2 "2018-01-09 00:00:00" -p1 160 -p2 14000
