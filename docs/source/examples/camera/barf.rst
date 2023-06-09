BARF
====================

In this example we showcase how to plug the nerfacc library into the *official* codebase 
of `BARF <https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/>`_. See 
`our forked repo <https://github.com/liruilong940607/barf/tree/90440d975fc76b3559126992b2fbce27dd02456f>`_
for details.


Benchmark: NeRF-Synthetic Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA GeForce RTX 2080 Ti. 

+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| PSNR                  | Lego  | Mic   |Materials| Chair |Hotdog | Ficus | Drums | Ship  | MEAN  |
|                       |       |       |         |       |       |       |       |       |       |
+=======================+=======+=======+=========+=======+=======+=======+=======+=======+=======+
| BARF                  | 31.16 | 23.87 | 26.28   | 34.48 | 28.4  | 27.86 | 31.07 | 27.55 | 28.83 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| training time         | 9.5hrs| 9.5hrs| 9.2hrs  | 9.3hrs|12.3hrs| 9.3hrs| 9.3hrs| 9.5hrs| 9.8hrs|
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| camera errors (R)     | 0.105 | 0.047 | 0.085   | 0.226 | 0.071 | 0.846 | 0.068 | 0.089 | 0.192 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| camera errors (T)     | 0.0043| 0.0021| 0.0040  | 0.0120| 0.0026| 0.0272| 0.0025| 0.0044| 0.0074|
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| Ours (occ)            | 32.25 | 24.77 | 27.73   | 35.84 | 29.98 | 28.83 | 32.84 | 28.62 | 30.11 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| training time         | 1.5hrs| 2.0hrs| 2.0hrs  | 2.3hrs| 2.2hrs| 1.9hrs| 2.2hrs| 2.3hrs| 2.0hrs|
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| camera errors (R)     | 0.081 | 0.036 | 0.056   | 0.171 | 0.058 | 0.039 | 0.039 | 0.079 | 0.070 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| camera errors (T)     | 0.0038| 0.0019| 0.0031  | 0.0106| 0.0021| 0.0013| 0.0014| 0.0041| 0.0035|
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
