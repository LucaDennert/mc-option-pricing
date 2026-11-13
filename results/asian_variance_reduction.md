Arithmetic Asian call, S0=K=100, T=1, 12 monthly fixings, sigma=20%, N = 65,536 paths.

| method                          |   price |   std error |   time (s) |   variance reduction |   efficiency gain (var x time) |
|:--------------------------------|--------:|------------:|-----------:|---------------------:|-------------------------------:|
| plain MC                        |  5.3135 |    3.13e-02 |      0.053 |                    1 |                              1 |
| antithetic                      |  5.3327 |    2.35e-02 |      0.044 |                    2 |                              2 |
| geometric-Asian control variate |  5.3256 |    8.41e-04 |      0.051 |                 1389 |                           1452 |
| antithetic + control variate    |  5.3238 |    8.34e-04 |      0.057 |                 1413 |                           1327 |
| Sobol + Brownian bridge (RQMC)  |  5.3245 |    7.54e-04 |      0.045 |                 1726 |                           2062 |
| Sobol + Brownian bridge + CV    |  5.3247 |    2.04e-04 |      0.045 |                23585 |                          27930 |
