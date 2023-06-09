.. _`TensoRF Example`:

TensoRF
====================

In this example we showcase how to plug the nerfacc library into the *official* codebase 
of `TensoRF <https://apchenstu.github.io/TensoRF/>`_. See 
`our forked repo <https://github.com/liruilong940607/tensorf/tree/f2d350873c54f249e64b6e745919b6a94bf54f1d>`_
for details.


Benchmark: NeRF-Synthetic Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA GeForce RTX 2080 Ti. 

+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| PSNR                  | Lego  | Mic   |Materials| Chair |Hotdog | Ficus | Drums | Ship  | MEAN  |
|                       |       |       |         |       |       |       |       |       |       |
+=======================+=======+=======+=========+=======+=======+=======+=======+=======+=======+
| TensoRF               | 35.14 | 25.70 | 33.69   | 37.03 | 36.04 | 29.77 | 34.35 | 30.12 | 32.73 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| training time         | 504s  | 522s  | 633s    | 648s  | 584s  | 824s  | 464s  | 759s  | 617s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| Ours (occ)            | 35.05 | 25.70 | 33.54   | 36.99 | 35.62 | 29.76 | 34.08 | 29.39 | 32.52 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| training time         | 310s  | 312s  | 463s    | 433s  | 363s  | 750s  | 303s  | 468s  | 425s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+


Benchmark: Tanks&Temples Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA GeForce RTX 2080 Ti. 

+-----------------------+-------+-------------+--------+-------+-------+
| PSNR                  | Barn  | Caterpillar | Family | Truck | MEAN  |
|                       |       |             |        |       |       |
+=======================+=======+=============+========+=======+=======+
| TensoRF               | 26.88 | 25.48       | 33.48  | 26.59 | 28.11 |
+-----------------------+-------+-------------+--------+-------+-------+
| training time         | 24min | 19min       | 15min  | 18min | 19min |
+-----------------------+-------+-------------+--------+-------+-------+
| Ours (occ)            | 26.74 | 25.64       | 33.16  | 26.70 | 28.06 |
+-----------------------+-------+-------------+--------+-------+-------+
| training time         | 19min | 15min       | 11min  | 13min | 14min |
+-----------------------+-------+-------------+--------+-------+-------+
