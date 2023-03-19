1.
验证函数为verification_sample.py:
```
python verification_sample.py
```
2.
主要调用修改在：
https://github.com/ssxxx1a/classifier_free/blob/2f28344b34513a7170317541a0b36d5e63f0c0ed/diffusion.py#L304

3.
https://github.com/ssxxx1a/classifier_free/blob/2f28344b34513a7170317541a0b36d5e63f0c0ed/diffusion.py#L187
以下目前使用的只有2中的 compare_cond_uncond_diff 和calc_diff。

4.
res_mean.jpg 即使用1/10概率叠加后的差值图
x从0-1000，为迭代的次序，对应timestep为999->0，
y为uncondition_epsilon和带有condition的累加值的差值。

5.
res_classifier.jpg 即，使用分类器得到的差值图。
