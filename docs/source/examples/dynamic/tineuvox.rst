.. _`TiNeuVox Example`:

TiNeuVox
====================

In this example we showcase how to plug the nerfacc library into the *official* codebase 
of `TiNeuVox <https://jaminfong.cn/tineuvox/>`_. See 
`our forked repo <https://github.com/liruilong940607/tineuvox/tree/0999858745577ff32e5226c51c5c78b8315546c8>`_
for details.


Benchmark: D-NeRF Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA GeForce RTX 2080 Ti. 

+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| PSNR                 | bouncing | hell    | hook  | jumping | lego  | mutant | standup | trex  | MEAN  |
|                      | balls    | warrior |       | jacks   |       |        |         |       |       |
+======================+==========+=========+=======+=========+=======+========+=========+=======+=======+
| TiNeuVox             | 39.37    | 27.05   | 29.61 | 32.92   | 24.32 | 31.47  | 33.59   | 30.01 | 31.04 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Training Time        | 832s     | 829s    | 833s  | 841s    | 824s  | 833s   | 827s    | 840s  | 833s  |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Ours (occ)           | 40.56    | 27.17   | 31.35 | 33.44   | 25.17 | 34.05  | 35.35   | 32.29 | 32.42 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Training Time        | 378s     | 302s    | 342s  | 325s    | 355s  | 360s   | 346s    | 362s  | 346s  |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+


Benchmark: HyperNeRF Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA GeForce RTX 2080 Ti. 

+----------------------+----------+---------+-------+-------------+-------+
| PSNR                 | 3dprinter| broom   |chicken| peel-banana | MEAN  |
|                      |          |         |       |             |       |
+======================+==========+=========+=======+=============+=======+
| TiNeuVox             | 22.77    | 21.30   | 28.29 | 24.50       | 24.22 |
+----------------------+----------+---------+-------+-------------+-------+
| Training Time        | 3253s    | 2811s   | 3933s | 2705s       | 3175s |
+----------------------+----------+---------+-------+-------------+-------+
| Ours (occ)           | 22.72    | 21.27   | 28.27 | 24.54       | 24.20 |
+----------------------+----------+---------+-------+-------------+-------+
| Training Time        | 2265s    | 2221s   | 2157s | 2101s       | 2186s |
+----------------------+----------+---------+-------+-------------+-------+
| Ours (prop)          | 22.75    | 21.17   | 28.27 | 24.97       | 24.29 |
+----------------------+----------+---------+-------+-------------+-------+
| Training Time        | 2307s    | 2281s   | 2267s | 2510s       | 2341s |
+----------------------+----------+---------+-------+-------------+-------+
