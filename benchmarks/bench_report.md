# EnochModel1 基准对比

环境: baseline: python 3.12.12, numpy 2.5.2, blas scipy-openblas, 模型 dtype float64, 参数 210,073 | optimized: python 3.12.12, numpy 2.5.2, blas scipy-openblas, 模型 dtype float64, 参数 210,073 | optimized_fp32: python 3.12.12, numpy 2.5.2, blas scipy-openblas, 模型 dtype float32, 参数 210,073 | optimized_t16: python 3.12.12, numpy 2.5.2, blas scipy-openblas, 模型 dtype float64, 参数 210,073

| 负载 | baseline | optimized | optimized_fp32 | optimized_t16 | 加速比 |
| --- | --- | --- | --- | --- | --- |
| ms/token(贪心解码) | 1.71 | 0.16 | 0.14 | 0.17 | **10.95x** / **12.56x** / **10.09x** |
| ms/token(4路采样) | 1.09 | 0.12 | 0.09 | 0.13 | **9.21x** / **11.98x** / **8.43x** |
| ms/步(语料LM) | 43.03 | 21.38 | 11.44 | 42.16 | **2.01x** / **3.76x** / **1.02x** |
| ms/步(RL含rollout) | 244.49 | 97.83 | 49.98 | 133.97 | **2.50x** / **4.89x** / **1.82x** |
| ms/步(fwd+bwd) | 40.56 | 25.87 | 11.77 | 40.11 | **1.57x** / **3.45x** / **1.01x** |

| 资源 | baseline | optimized | optimized_fp32 | optimized_t16 |
| --- | --- | --- | --- | --- |
| 峰值 RSS (MiB) | 113.8 | 99.6 | 71.2 | 101.5 |
| 参数体积 (B) | 1,680,584 | 1,680,584 | 840,292 | 1,680,584 |
| checkpoint 体积 (B) | 1,687,842 | 1,687,842 | 1,687,842 | 1,687,842 |
| 全负载总耗时 | 2.57s | 1.26s | 0.63s | 2.07s |
