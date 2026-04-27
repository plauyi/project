
# DCN 模型速读（Deep & Cross Network）

DCN 主要解决 CTR/推荐里“特征交叉”问题：

- DNN 能学高阶非线性交互，但对显式低阶交叉不够直接、样本效率偏低。  
- LR/FM 擅长低阶交叉，但表达能力有限。  
- DCN 用 Cross Network 显式构造交叉项 + Deep Network 学隐式高阶模式，兼顾两者。  

## 1) 核心结构

输入向量设为 `x0`（通常是 embedding 拼接后的 dense 向量）。

### Cross Layer（经典 DCN）

第 `l` 层：

$$
x_{l+1}=x_0(w_l^\top x_l)+b_l+x_l
$$

- `w_l` 是 `d` 维向量（`d` 为输入维度）  
- `w_l^T x_l` 是标量  
- `x_0 * scalar` 让每层都“回到原始输入”做显式交叉  

直觉：每加一层，能显式提升交叉多项式阶数（到 `l+1` 阶）。

### Deep Tower

普通 MLP：`Linear + Act + Dropout/BN ...`

### 融合

常见是 `concat([cross_out, deep_out]) -> final linear -> sigmoid`

## 2) 为什么有效

- 显式交叉：比纯 MLP 更容易学到“字段 A × 字段 B”这类组合。  
- 参数高效：经典 cross 每层参数量 `O(d)`，比全连接交叉便宜。  
- 稳定性好：残差项 `+ x_l` 有利训练。  

## 3) DCN v2（面试常问）

经典 DCN 的 `w` 是向量，表达能力有限。DCN v2 常见改进：

$$
x_{l+1}=x_0\odot(W_lx_l+b_l)+x_l
$$

- `W_l` 为矩阵，能力更强（可学更复杂交叉）  
- 但参数更大，常配低秩分解降成本（如 `W≈UV`）  

一些实现还支持：

- `stacked`：cross 后接 deep  
- `parallel`：cross 与 deep 并行再融合（实践中常见）  

## 4) 跟 FM / DeepFM / xDeepFM 区别（高频）

- FM：二阶交叉，强归纳偏置，简单稳。  
- DeepFM：FM（二阶显式）+ DNN（高阶隐式）。  
- DCN：Cross（可到更高阶显式）+ DNN。  
- xDeepFM（CIN）：更细粒度向量级显式交叉，结构更复杂。  

一句话：  
DCN 是“显式高阶交叉 + 深层隐式交互”的折中方案。

## 5) 训练与工程要点

- 输入前先做 embedding（类别特征）+ 数值标准化。  
- Cross 层一般不宜过深（常见 2~6 层）。  
- Deep 分支可配 BN/Dropout，Cross 分支通常较轻。  
- 优化器常用 Adam/AdamW。  

线上注意：

- embedding 热点与内存  
- 特征一致性（训练/推理 schema）  
- 冷启动 fallback  

## 6) 面试高频问题 + 参考回答

### Q1: DCN 为什么要每层都乘 x0，不用 xl 自乘？

答：乘 x0 是一种结构化显式交叉约束，能稳定地构造多项式交叉项，同时参数量低、泛化更稳；如果自由自乘，表达更强但更难训、易过拟合。

### Q2: Cross 层越深越好吗？

答：通常不是。层数增加会提升交叉阶数，但收益递减，且训练不稳/过拟合风险上升。工业里常 2~6 层，通过离线 AUC + 线上指标调参。

### Q3: DCN 和 DeepFM 怎么选？

答：

- 特征稀疏且二阶关系明显：DeepFM 常是稳妥 baseline。  
- 需要更强显式高阶交叉且参数预算可控：DCN/DCN-v2 常更优。  

最终靠离线+在线实验决定。

### Q4: DCN v2 为什么更强？

答：v1 的向量核等价于 rank-1 形式，交叉表达受限；v2 用矩阵核（可配低秩）提升表示能力，能拟合更复杂特征关系。

### Q5: 如何解释线上效果不升反降？

答：优先查四类：

1. 训练推理特征不一致（最常见）  
2. 交叉层过深过拟合  
3. 负采样/样本分布偏移  
4. 校准问题（AUC升但 pCTR 偏）  

再做 ablation：去掉 cross / 调浅 cross / 调正则与dropout。

### Q6: DCN 的复杂度？

答：经典 cross 每层 `O(d)` 参数、`O(d)` 乘加；v2 矩阵版每层约 `O(d^2)`，常用低秩降到 `O(dr)`。

## 7) 可直接背的项目表述（面试）

“我在 CTR 项目里用了 DCN（Cross + Deep 并行）。Cross 分支显式学习多阶特征交叉，Deep 分支学习隐式非线性。相比纯 MLP，DCN 在相同 embedding 规模下通常能更快学到关键字段组合。调参上我主要控制 cross 层数、deep 宽度和正则，结合离线 AUC 与线上 CTR/CVR 校准共同决策是否上线。”
