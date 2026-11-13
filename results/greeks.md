ATM 1Y call and digital, S0=K=100, sigma=20%, 131k antithetic paths; bumps use common random numbers.

| Greek         |   closed form |   MC bump & revalue (CRN) | MC single-run estimator                                 |
|:--------------|--------------:|--------------------------:|:--------------------------------------------------------|
| call delta    |        0.5735 |                    0.5735 | pathwise 0.5747 ± 0.0026                                |
| call gamma    |        0.0194 |                    0.0193 | -                                                       |
| call vega     |       38.7152 |                   38.7797 | -                                                       |
| call rho      |       48.5223 |                   48.5223 | -                                                       |
| digital delta |        0.0194 |                    0.0196 | likelihood ratio 0.0193 ± 0.0001 (pathwise = 0, biased) |
