实现情况，目前堪比开源库，我感觉还差个 PQ 量化就可以开源了。
回头没架构优化我就开 PQ 了。
先说数据集实力：
--- SIFT 评测结果 ---
  Build 时延: 380306.578000 ms
  Build 距离计算次数: 6405623149
  平均 Search 时延 (T_avg): 0.361298 ms
  平均 Search 距离计算次数: 2481.338400
  检索精度 (delta): 0.992870
  [成功] 精度满足 >= 0.99 约束。

--- GLOVE 评测结果 ---
  Build 时延: 900444.260800 ms
  Build 距离计算次数: 16065900995
  平均 Search 时延 (T_avg): 3.393075 ms
  平均 Search 距离计算次数: 23492.766600
  检索精度 (delta): 0.991590
  [成功] 精度满足 >= 0.99 约束。

再说核心过程：
读了论文 https://arxiv.org/abs/1603.09320 实现了初版代码。
之后看了 https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h 
select_neighbor 的代码和论文写的不一样，这里后面大概率是加了启发式优化。
实现了之后快了不少。
之后通过 https://arxiv.org/abs/2505.15636 加了自适应，第一次变快了。
（话说咱是不是可以仿照这个的思路写一篇，虽然好像没投上，但这篇论文我看了一下
它估计也是知道自己的出发点没啥创新，但我试了真的有用）
我看了 SHG 等启发式跳层策略，过于复杂（当然后续我基本确认没用）
之后读了 https://arxiv.org/abs/2412.01940 去掉了层级。

之后仿照 https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h
加了 SIMD 和多线程，多写了一点点，但我觉得很值得，因为快了不少。

接下来我找到了 https://ann-benchmarks.com/index.html
发现 PQ 量化是不错的策略，但太复杂先搁置了。

思考了一下 IVF 结合的策略，进行了实验，即选取第一次 search 的前十作为入口点，发现没用，
因此得到入口点不重要的理论。

也就是关键在于优化图的结构，此时我看到了 https://arxiv.org/abs/1810.07355，这篇

ONNG 的关键性优化引发了我的思考，我没有选择参考 https://github.com/yahoojapan/NGT/blob/main/lib/NGT/GraphReconstructor.h

中按照角度删边的逻辑，而是结合了那个 beam-search 的思路，采用了代码所示的删边策略，最终它也没让我失望，这是核心优化。

接下来我打算沿着 NSG 的图结构思路看看有没有搞头。

