# Predictive Latent Correction for Frozen Humanoid Controllers

### Research Plan v3 — August 2026

## Executive thesis

我们研究一个非常具体的问题：

> **给定一个已经很强、完全冻结的 humanoid whole-body controller，以及它即将执行的 motion intent，能否在 tracking failure 真正发生之前预测其物理执行偏差，并利用 controller-native latent 中有限、受约束的 residual intervention 提前修正？**

核心不是 tokenization。

核心不是重新做 domain randomization。

核心也不是证明 history 有信息。

真正的科学问题是：

[
\boxed{
\text{Does explicit prediction of future execution
provide control value beyond direct history-conditioned adaptation?}
}
]

整个项目只保留三条主 claim：

### C1 — Correctability

存在幅度受限、保持原 motion intent 的 latent correction：

[
\delta z
]

能够挽救一部分 frozen SONIC 原本会失败的执行。

### C2 — Forecastability

在 frozen controller 已经拥有的 observation history 基础上，未来 execution degradation / failure 可以提前预测，而且预测提前量大于实际 correction pipeline 所需要的时间。

### C3 — Predictive control value

在**相同输入、相近模型容量、相同训练数据和相同 residual authority**下，

[
\text{predictive corrector}

>

\text{direct history-conditioned adapter}.
]

C3 才是整篇论文最重要的 claim。

---

# 1. 首先修正 v2 对 SONIC history 的理解

这个 TODO 现在可以关闭。

当前 SONIC release config 明确设置：

[
\texttt{actor_prop_history_length}=10
]

以及

[
\texttt{actor_actions_history_length}=10.
]

policy observations 中 gravity direction、base angular velocity、joint positions、joint velocities 和 actions 都按这个 history length 堆叠。

同时 release 使用的是：

[
\texttt{universal_token/all_mlp_v1}
]

其 G1 dynamic decoder 是 MLP：

[
[\text{token},\text{proprioception history}]
\rightarrow
\text{joint action}.
]

它并不是一个我们需要提取 LSTM/GRU hidden state 才能公平比较的 recurrent policy。

因此不应该再写：

> “相对 SONIC recurrent state 多提供 AUPRC +0.05。”

正确的实验是：

[
P(\rho\mid z)
]

vs.

[
P(\rho\mid z,o_t)
]

vs.

[
P(\rho\mid z,x_t^{SONIC})
]

vs.

[
P(\rho\mid z,h_t^{long})
]

其中

[
x_t^{SONIC}
]

就是 SONIC 真正看到的完整 10-frame observation stack。

如果更长 history 没有帮助，并不意味着项目失败。

甚至如果 frozen SONIC hidden feature 本身已经非常容易 linear-probe 出 failure risk，也不意味着项目失败。

因为 SONIC 的训练目标不是：

[
\text{predict future failure}.
]

最终真正的检验仍然是：

[
\boxed{
\text{explicit prediction}
\stackrel{?}{>}
\text{direct adapter using exactly the same information}.
}
]

---

# 2. Novelty 防守需要重写

v2 对 RMA、Residual RL、ABS 的定位是正确的，但还不够。

## RMA

RMA 从 history 中估计环境/动力学 extrinsics，再让 policy 根据估计结果适应。

我们的差异不是：

> “我们用了 history。”

而是：

[
\text{history}
\rightarrow
\boxed{\text{future execution consequence}}
\rightarrow
\text{correction}.
]

也就是说我们预测的是：

> 如果继续执行当前 intent，接下来会发生什么？

而不是：

> 当前环境参数大概是什么？

---

## ABS / BAS

ABS 已经使用 policy-conditioned reach-avoid value 预测危险并切换 recovery policy；BAS 又加入未知 friction、payload 等物理条件的在线估计。

所以我们的 novelty 绝不能写成：

> “预测危险然后恢复。”

真正区别是：

**ABS/BAS**

[
\text{risk}
\rightarrow
\text{policy switch / recovery policy}
]

而我们：

[
\text{given motion intent}
\rightarrow
\text{predicted tracking execution}
\rightarrow
\delta z
\rightarrow
\text{same frozen controller}.
]

我们试图保留原 behavior，而不是离开原 behavior。

---

## ASAP

ASAP 已经学习 delta action model 来描述真实机器人和 simulation physics mismatch，并利用它重新 fine-tune policy。

区别是我们：

* 不需要真实数据建立 mismatch model；
* 不 fine-tune SONIC；
* 不把 dynamics correction 烘焙回 base controller；
* 研究的是 state- and intent-conditioned **future execution failure**。

---

## MOSAIC

这是目前最直接的危险工作之一。

MOSAIC 已经针对 generalist humanoid tracker 做 rapid residual adaptation，并通过 additive residual module 提升真实 teleoperation robustness。

因此我们不能再把：

> “frozen/generalist controller + residual”

当贡献。

我们的 delta 必须明确写成：

> **Prediction precedes correction. The residual is selected because of a forecasted future execution consequence, not merely from present tracking error or interface context.**

---

## RobotDancing / CoorDex

RobotDancing 已经说明 residual action 能显著改善 humanoid long-horizon tracking；CoorDex 也利用 frozen latent priors + residual RL 完成 humanoid loco-manipulation。

所以：

[
\boxed{\text{residual interface itself is not novelty}.}
]

这反而让我们的实验设计更干净：

> **Why prediction?**

必须成为整篇论文唯一真正需要回答的问题。

---

# 3. 删除主路线中的 execution-token / pairwise metric-learning

我会比 v2 更进一步。

不仅 token 可选。

我建议第一篇里连

[
\mathcal L_{\text{exec-structure}}
]

都先彻底拿掉。

原因有三个。

第一，它需要昂贵的 pairwise counterfactual comparison。

第二，它会把 representation-learning 和 control 两个 research questions 混在一起。

第三，即使完全没有所谓 execution-equivalent token，我们的 C1–C3 仍然可以成立。

因此主模型只学习：

[
p_\theta
(
Y_{t:t+H}^{exec},
\rho_{t:t+H}
\mid
x_t^{SONIC},
z_t^S
).
]

如果之后发现 predictor latent 自然形成有意义的 execution geometry，我们再把它作为 analysis。

只有在后续 language experiment 确实需要 discrete interface 时才研究 tokenization。

---

# 4. 不再叫“World Model”，先叫 Predictive Execution Model

“world model”容易让 reviewer 期待：

-完整 state transition；

* long-horizon imagination；
* planning；
* observation generation。

我们其实不需要这些。

因此核心模块叫：

## Predictive Execution Model — PEM

输入：

[
x_t^{SONIC},
\quad
z_t^S.
]

输出只预测与 failure/correction 有关的低维物理量：

[
\hat y_{t:t+H}
==============

[
\Delta q,
\Delta\dot q,
\Delta R_{root},
\Delta\omega_{root},
c^{mismatch},
v^{foot}_{slip},
\rho
].
]

并不试图生成完整世界。

形式：

[
\hat y_{t:t+H}
==============

F_\theta(x_t^{SONIC},z_t^S).
]

这让模型：

* 小；
* 快；
* 单 5090 容易训练；
* inference 可以进入 supervisory loop；
* claim 也更准确。

---

# 5. Baseline 必须重新整理

v2 的 Baseline B：

[
\delta z=f(h,z)
]

和所谓 RMA-style Baseline C 实际上太接近。

两者都是：

[
\text{history}
\rightarrow
\text{latent residual}.
]

这样比较没有科学意义。

新的 baseline ladder：

| 方法                                    | 输入                                      |                           是否预测未来 | 是否改 base controller |
| ------------------------------------- | --------------------------------------- | -------------------------------: | ------------------: |
| **A Frozen SONIC**                    | SONIC input                             |                               No |                  No |
| **B Error-feedback residual**         | current tracking error + intent         |                               No |                  No |
| **C Direct history adapter**          | exact SONIC history + intent            |                               No |                  No |
| **D System-ID adapter**               | history → estimated dynamics → residual | No explicit execution prediction |                  No |
| **E Predictive Execution Correction** | exact same history + intent             |                          **Yes** |                  No |
| **F Additional-DR fine-tune**         | —                                       |                               No |             **Yes** |

Baseline C 是真正的核心对手。

它就是：

[
\delta z_t
==========

g_\phi(x_t^{SONIC},z_t^S).
]

而我们的方法：

[
\hat y_{t:t+H}
==============

F_\theta(x_t^{SONIC},z_t^S),
]

[
\delta z_t
==========

G_\psi(x_t^{SONIC},z_t^S,\hat y_{t:t+H}).
]

关键比较：

[
\boxed{E;vs.;C}.
]

---

# 6. 还必须增加一个 matched-capacity prediction ablation

否则即使 E 胜过 C，reviewer 仍然可以说：

> “你只是多了网络容量和辅助 supervision。”

所以至少加入：

### C+

与 E 使用相同 backbone / parameter count，但不给 correction head predicted future execution：

[
\delta z=G(x,z,\text{dummy auxiliary features}).
]

或者训练相同 prediction auxiliary loss，但 correction head 不读取预测结果。

然后需要满足：

[
E>C^+.
]

另外做一个非常重要的 mediator test：

### Prediction shuffle

evaluation 时保持：

[
x_t,z_t
]

不变，

但把：

[
\hat y_t
]

在 batch 内随机置换。

如果 control performance 明显下降，说明 corrector 真正在使用 future-execution information。

如果完全不下降：

> predictor 只是 regularizer，根本不是 mechanism。

这会是非常强的 ablation。

---

# 7. Phase 0 应进一步压缩成三个最便宜的问题

## G0 — Latent controllability

不要再使用：

> “pre-quant no-op rate <20%”

作为核心标准。

因为 no-op rate 完全依赖 residual amplitude。

我们真正需要的是：

[
\boxed{
\text{controllability–distortion curve}
}
]

扫描：

[
|\delta z|
==========

\epsilon_1,\epsilon_2,\ldots
]

比较 pre-quant 和 post-quant：

[
\Delta a(\epsilon),
]

[
\Delta Y(\epsilon),
]

[
\Delta MPJPE(\epsilon),
]

[
P(Q(z+\delta z)=Q(z)).
]

SONIC 当前代码正式支持 pre-quantization、post-quantization 和 replacement residual injection，因此这个扫描可以直接建立在 release interface 上。

选出的不是：

> no-op 最少的 channel，

而是：

> **在最小 intent distortion 下拥有最大有效 physical authority 的 channel。**

---

# 8. Oracle 搜索必须改成 constrained rescue

原来的：

[
\text{“没有摔倒”}
]

远远不够。

否则 oracle 可以：

> 停下来；

> 大幅减速；

> 完全不跟 reference；

> 通过 latent 把原动作改成另一件事。

这是假 headroom。

定义 constrained rescue：

[
\text{Rescue}
=============

\begin{cases}
1,
&
\text{no termination}
\
&
\text{motion progress}\ge90%\text{ nominal}
\
&
\Delta MPJPE\le10%
\
&
|\delta z|\le\epsilon_{\max}
\
0,&\text{otherwise}.
\end{cases}
]

Oracle objective：

[
J=
w_sJ_{\text{survival}}
----------------------

## w_tJ_{\text{tracking}}

## w_vJ_{\text{speed loss}}

## w_z|\delta z|^2

w_jJ_{\text{jerk}}.
]

G1 的真正问题变成：

> **在不明显改变原 motion 的情况下，有多少 baseline failure 在 latent channel 上实际上可救？**

---

# 9. Oracle 搜索降维，否则 single 5090 很浪费

不要直接搜索：

[
\delta z_{t:t+H}
\in
\mathbb R^{64H}.
]

第一阶段使用低维 correction basis：

[
\delta z_t=Uc_t,
\qquad
U\in\mathbb R^{64\times r},
]

其中：

[
r\in{8,16}.
]

(U) 可以来自：

* SONIC latent PCA；
* decoder local Jacobian 的 dominant directions；
* random orthogonal basis 作为 control。

Correction 在短窗口内 piecewise constant：

[
\delta z_{t:t+K_c}
==================

\delta z.
]

先证明：

[
\exists\delta z
]

再考虑复杂 temporal residual。

### Phase-0 oracle规模

从 L0 中选择约：

[
N_f=64\text{--}128
]

个代表性 near-failure states。

每个：

[
64\text{--}128
]

candidate，

2–3 轮 CEM。

rollout horizon：

[
0.4\text{--}0.8s.
]

全部 vectorized。

这比对数千个 failure 做完整 MPPI 便宜几个数量级，却足够回答 C1。

---

# 10. Gate G1 不再只报告一个 rescue%

需要画：

[
\boxed{
R_{\text{rescue}}
(B,\epsilon)
}
]

其中：

* (B)：oracle search budget；
* (\epsilon)：允许 residual authority。

这样我们能看到：

> 需要多强 correction 才能救？

以及：

> 增加 search compute 是否仍持续带来 headroom？

如果只有：

[
4096\text{ candidates}
]

和巨大 residual 才能 rescue，

虽然数学上有 headroom，工程上仍然没有价值。

### Revised G1

在 modest search budget + bounded latent authority 下：

[
R_{\text{rescue}}\ge40%
]

→ 主线继续。

[
20%-40%
]

→ 只研究可救 failure family。

[
<20%
]

→ 主方向停止。

我不会预注册 50% 这个过于武断的数字。

---

# 11. Counterfactual dataset 必须分开 state effect 和 dynamics effect

v2 的：

[
\mathcal G_i
============

{
u_i,(s_i^{(1)},Y_i^{(1)}),\ldots
}
]

会同时改变 state 和 dynamics。

这会让 causal interpretation 重新混乱。

应该建立两个 factorial。

## Dynamics counterfactual

保持：

[
u,s
]

完全相同，

只改变：

[
\xi.
]

即：

[
Y(u,s,\xi_1),
\dots,
Y(u,s,\xi_K).
]

回答：

> 同一个状态和 intent，在不同 physics 下会怎样？

## State counterfactual

保持：

[
u,\xi
]

相同，

只改变：

[
s.
]

即：

[
Y(u,s_1,\xi),
\dots,
Y(u,s_J,\xi).
]

回答：

> 同一个 intent 在不同 balance/contact phase 下会怎样？

只有后续才做完整 crossed design：

[
Y(u,s_j,\xi_k).
]

这才真正对应：

[
d_{\text{exec}}(\cdot\mid s).
]

---

# 12. 预测问题应定义成 horizon-conditioned event prediction

不要简单给每帧一个：

[
\text{fall/not fall}.
]

定义：

[
y_t^{(H)}
=========

\mathbf 1
[
\text{failure occurs within }(t,t+H]
].
]

使用：

[
H\in
{200,400,600}\text{ ms}.
]

然后分别报告：

[
\operatorname{AUPRC}(H).
]

这样我们可以真正得到：

> 200 ms horizon 很容易；

> 600 ms horizon 很难；

> 哪个 horizon 仍然足够可控？

---

# 13. Lead time 不应该硬编码成 300 ms

SONIC controller 本身运行 50 Hz，并已有低延迟 checkpoint。

corrector 实际可以跑：

[
10\text{--}25\text{ Hz}
]

取决于我们的 measured inference latency。

所以 Gate 应该定义成：

[
\boxed{
\frac{
T_{\text{lead}}
}{
T_{\text{sense}}
+
T_{\text{predict}}
+
T_{\text{correct}}
+
T_{\text{actuation}}
}
\ge3
}
]

而不是先规定：

[
T_{\text{lead}}\ge300\text{ ms}.
]

如果整个 correction pipeline 最终只需要 40–60 ms，

150–200 ms lead time 可能已经足够有实际价值。

先测 latency，再冻结 gate。

---

# 14. Predictability Gate 也需要重写

v2 的 AUPRC ≥0.60 在正类 30% 时不算离谱，但单独使用不够。

Primary：

[
\text{AUPRC}
]

[
\text{Brier score}
]

[
\text{calibration slope/intercept}
]

以及：

[
\text{lead-time recall at fixed precision}.
]

例如：

> 在 precision = 80% 的 operational threshold 下，有多少 failure 能至少提前 (3T_{\rm pipeline}) 被发现？

这个 metric 与 controller 最直接相关。

ECE 可以继续报告，但不要让：

[
ECE\le0.05
]

成为唯一 calibration kill-switch，因为 ECE 对 binning 和样本量相当敏感。

---

# 15. 最重要的预测对照

所有 predictor 使用完全相同 train split：

### P0 — intent only

[
F(z_t)
]

回答：

> 只是因为这条 motion 本来就难吗？

### P1 — current physical state

[
F(z_t,o_t)
]

### P2 — exact SONIC information

[
F(z_t,x_t^{SONIC})
]

这是主 baseline。

### P3 — extended history

[
F(z_t,h_t^{long})
]

### P4 — privileged dynamics oracle

[
F(z_t,x_t,\xi)
]

只作为 upper bound。

这比所谓：

> “vs recurrent state”

科学得多。

---

# 16. Corrector 不要一开始做 flow matching

先用最简单的：

[
\delta z
========

\operatorname{MLP}(\cdot).
]

而且 direct adapter 和 predictive adapter 最好都从**同一批 oracle corrections**蒸馏。

这样：

[
\text{data},
\text{supervision},
\text{optimization}
]

基本一致。

Difference 只剩：

[
\text{explicit foresight}.
]

只有 oracle correction distribution 明确显示：

[
p(\delta z^*|x,z)
]

是多峰的，

比如 cluster analysis / conditional covariance 显示明显多个 recovery mode，

才引入：

[
\text{flow matching}.
]

否则 flow 是不必要复杂度。

---

# 17. 一个非常重要的新 Gate：Prediction Utilization

即使 predictive model 准确，也可能 corrector 完全不用它。

增加：

## G3b — Mediator Gate

比较正常模型：

[
G(x,z,\hat y)
]

与：

[
G(x,z,\operatorname{shuffle}(\hat y)).
]

以及：

[
G(x,z,0).
]

要求：

[
M_{\text{normal}}

>

M_{\text{shuffle}}
]

并且 effect 在多个 perturbation family 中一致。

如果 shuffle prediction 后 performance 完全不变：

> **不要再声称 prediction drives correction。**

论文应该退化成：

> auxiliary predictive supervision improves residual adaptation。

这是一个很重要的 claim boundary。

---

# 18. Revised decisive Gate G4

不再要求：

[
\text{Predictive}

>

\text{Reactive}

>

\text{RMA}

>

\text{SONIC}.
]

中间顺序没有必要成立。

唯一重要的是：

[
\boxed{
M_{\text{Predictive}}

>

\max(
M_{\text{Direct Adapter}},
M_{\text{Reactive}},
M_{\text{Frozen}}
)
}
]

在 paired stress evaluation 上：

### Primary

failure / termination：

相对 frozen SONIC：

[
\ge25%-30%\text{ relative reduction}.
]

相对 direct history adapter：

[
\ge10%-15%\text{ relative reduction}.
]

### Fidelity constraints

[
\Delta MPJPE\le5%.
]

[
\Delta\text{motion progress}\ge-10%.
]

同时报告：

[
|\delta z|,
]

intervention frequency，

energy，

jerk。

---

# 19. 统计设计必须从 independent trial 改成 paired trial

v2 写：

> 30% → 21%，每 condition ≈350 trials。

对于完全独立 Bernoulli trial，这个数量级基本合理。

问题在于我们的 trials **不独立**：

同一 motion、

同一 source actor、

同一 perturbation family

会产生明显 correlation。

因此不能：

[
350\text{ rollout windows}
]

直接当作：

[
n=350.
]

真正设计应该是：

对每个：

[
(m,s,\xi)
]

使用完全相同的 rollout seed 比较：

[
\pi_A
\quad vs\quad
\pi_B.
]

也就是 common-random-number paired evaluation。

统计单位优先设为：

[
\text{motion/source group}.
]

CI 使用：

[
\text{cluster bootstrap over source motions}.
]

binary rescue/failure 可使用 paired analysis。

这样比机械地跑：

[
350\times5
]

有效率得多，也更适合单卡。

最终 headline methods 再使用 5 independent training seeds；prediction dataset 本身没有必要机械复制 5 次。

---

# 20. Real robot Gate 需要降低“故意摔机器人”的成分

我不同意：

> “每 condition 30 次真实 fall trial。”

真实机器人实验不应该把：

[
\text{actual fall}
]

设计成需要大量发生的统计事件。

硬件阶段的 primary endpoint 改成：

[
\boxed{\text{safety-stop / recovery success}}
]

并使用：

* overhead tether；
* bounded push；
* bounded payload；
* conservative abort threshold。

Simulation 承担 powered statistical comparison。

Hardware 承担：

> mechanism exists outside simulation。

最重要的 hardware evidence 是完整时间序列：

[
\text{same intent}
]

[
\downarrow
]

[
\text{risk rises}
]

[
\downarrow
]

[
\delta z \text{ activates}
]

[
\downarrow
]

[
\text{tracking residual reverses}
]

[
\downarrow
]

[
\text{motion continues}.
]

而不是追求几十次真正摔倒。

---

# 21. Single-5090 实验顺序

整个计划应该围绕：

> **最早得到一个能够杀死项目的答案。**

## Phase 0A — SONIC instrumentation

使用 release checkpoint。

不重训 SONIC。

验证 baseline performance。

暴露：

[
z_t^S,
x_t^{SONIC},
a_t,
s_t,
contact,
termination.
]

---

## Phase 0B — Latent authority

约：

[
256\text{--}512
]

diverse motion windows。

pre/post quant amplitude sweep。

得到：

[
\text{controllability–distortion curves}.
]

---

## Phase 0C — Oracle headroom

只选：

[
64\text{--}128
]

representative near-failure states。

低秩 latent search。

得到：

[
R_{\text{rescue}}(B,\epsilon).
]

### Kill switch #1

如果 bounded correction 几乎没有 rescue headroom：

**停止。**

---

## Phase 0D — Minimal predictor

这一步其实会训练小网络。

但：

> 不训练任何 controller。

训练三个很小的 MLP/TCN：

[
P0,;P2,;P3.
]

模型控制在：

[
<1\text{--}5M params.
]

### Kill switch #2

如果 failure 只能在已经来不及 correction 时被预测：

**停止 predictive-control 主线。**

---

# 22. Phase 1 — Counterfactual execution dataset

只有两个 gate 都过以后才扩数据。

第一版完全不需要 BONES-SEED 全量。

建议：

[
2k\text{--}5k
]

motion windows，

每个约：

[
K=4\text{--}8
]

physics interventions。

优先覆盖：

locomotion、turn、stop、crouch、transition、jump、asymmetric whole-body motion。

保存严格的 paired counterfactual metadata。

---

# 23. Phase 2 — Predictive Execution Model

训练：

[
P0,
P1,
P2,
P3,
P4
]

五级信息 ablation。

主要回答：

[
\text{future physical execution 到底能预测到什么程度？}
]

同时绘制：

[
\boxed{
\text{prediction quality}
;vs;
\text{forecast horizon}
}
]

和：

[
\boxed{
\text{failure recall}
;vs;
\text{lead time}
}.
]

---

# 24. Phase 3 — Correction

生成 oracle corrections。

训练：

### Direct adapter

[
\delta z_C=g_C(x,z).
]

### Predictive adapter

[
\hat y=F(x,z)
]

[
\delta z_E=g_E(x,z,\hat y).
]

### Matched-capacity auxiliary baseline

相同参数量、相同 prediction loss，但 correction 不读取 (\hat y)。

最后做：

prediction shuffle test。

### Kill switch #3

如果：

[
E\simeq C
]

说明 explicit foresight 没有 marginal control value。

这仍然是一个非常有价值的 negative result，但主方法 story 到此结束。

---

# 25. Phase 4 — OOD

训练 perturbations：

* friction；
* COM；
* torque strength；
* push；
* observation noise。

测试：

* actuator delay；
* payload；
* damping change；
* 另一种未见 disturbance family。

不要把 floor compliance 写死为 mandatory；如果 simulator/contact model 不支持稳定、可信的 compliant surface，换一个更干净的 held-out dynamics intervention。

真正核心是：

[
\boxed{
\text{held-out perturbation type}
}
]

而不是具体一定要是哪一种。

---

# 26. Phase 5 — Hardware

只有 G1–G4 全过再做。

先：

nominal motion。

再：

motion transition。

再：

small push。

再：

payload。

最后才提高 disturbance。

hardware 用于支持：

[
\text{prediction}
\rightarrow
\text{intervention}
\rightarrow
\text{recovery}
]

的机制证据。

---

# 27. Token / Language 彻底移出主时间线

如果主论文已经有：

[
\text{correctability}
+
\text{forecastability}
+
\text{predictive advantage}
+
\text{OOD}
+
\text{hardware},
]

它已经是一篇完整论文。

只有后续我们需要：

[
\text{language}\rightarrow\text{intent}
]

接口时，才重新研究：

[
z^S_{1:H}
\rightarrow
u_{1:K}.
]

因此 token 不再有 Gate G7。

它变成另一个 project branch。

这避免一个本来很锋利的 control paper 被 foundation-model vocabulary story 稀释。

---

# 28. Revised Gate Table

| Gate    | 真正的问题                                           | 通过标准                                                                   | 失败后                             |
| ------- | ----------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------- |
| **G0**  | latent 有可控 residual authority 吗？                | 存在低 distortion 的有效 residual region                                     | 换 channel / substrate           |
| **G1**  | bounded residual 真能救吗？                          | constrained rescue ≥约40%；20–40%则缩 failure family                       | <20% 停题                         |
| **G2**  | failure 能及时预测吗？                                 | operational precision 下 lead time ≥3× measured correction latency      | monitor/analysis paper          |
| **G3**  | prediction 是否超越 command/current-state baseline？ | P2/P3 在 OOD 上有稳定 predictive value                                      | 简化 model                        |
| **G3b** | corrector 真使用 prediction 吗？                     | prediction shuffle 显著伤 performance                                     | 改 claim 为 auxiliary supervision |
| **G4**  | explicit foresight 是否胜 direct adapter？          | vs direct adapter 10–15% relative failure reduction；fidelity preserved | predictive claim 失败             |
| **G5**  | 未见 dynamics 类型仍成立？                              | improvement 保留大部分                                                      | claim 限于 in-distribution        |
| **G6**  | real robot 是否复现 mechanism？                      | 安全条件下出现重复的 risk→correction→recovery sequence                           | 保持 sim-only claim               |

这些百分比在 pilot 后应冻结一次，而不是现在假装它们有自然常数意义。

---

# 29. 最值得做的第一张 Figure

最终 Paper Figure 1 不应该是 architecture。

而应该是：

### Same Intent, Divergent Execution, Predictable Recovery

三条 rollout：

**Nominal**

[
z^S
\rightarrow
\text{successful execution}.
]

**Perturbed**

[
z^S
\rightarrow
\text{risk rises}
\rightarrow
\text{tracking divergence}
\rightarrow
\text{failure}.
]

**Perturbed + Predictive Correction**

[
z^S
\rightarrow
\text{risk rises}
\rightarrow
\delta z
\rightarrow
\text{execution returns}
\rightarrow
\text{success}.
]

下方再配：

[
P(\text{failure within }400ms)
]

[
|\delta z_t|
]

[
\text{root/contact error}
]

随时间变化。

这一张图实际上会把整篇论文说清楚。

---

# 30. 最关键的第二张 Figure

不是 token embedding。

而是：

[
\boxed{
\text{Rescue success}
;vs;
\text{available lead time}
}
]

比较：

* Frozen SONIC；
* error feedback；
* direct history adapter；
* predictive correction；
* oracle correction。

如果出现：

```text
lead time
  500 ms    predictive ≈ oracle
  300 ms    predictive clearly > reactive
  150 ms    gap shrinks
   50 ms    all learned methods fail
```

这几乎直接证明：

> **foresight has causal control value because earlier information creates recoverable control headroom.**

这是我现在认为这篇论文最强、最干净的 mechanism result。

---

# 31. 最终 paper claim

如果所有主 gate 成立，最终摘要中心应该是：

> Strong humanoid motion controllers already condition on short proprioceptive and action histories, yet they can still fail under unseen dynamics because their objective is immediate control rather than explicit prediction of the physical consequence of an intended motion. We show that many such failures retain bounded correction headroom in the native latent interface of a frozen generalist controller. A lightweight execution predictor forecasts reference-conditioned tracking degradation before failure, and a predictor-mediated residual corrector exploits this lead time to reduce failures under unseen dynamics. Crucially, it outperforms matched direct history-conditioned adapters with identical observations and residual authority, demonstrating that explicit foresight — rather than additional history or residual capacity alone — provides the improvement.

最重要的关键词不是：

**token**。

不是：

**world model**。

也不是：

**residual policy**。

而是：

[
\boxed{\textbf{foresight}}
]

以及它带来的：

[
\boxed{
\textbf{lead time}
\rightarrow
\textbf{correctable headroom}.
}
]

这才是整个研究最难被现有 RMA、ASAP、MOSAIC、ABS 和 residual-control 工作吞掉的 scientific delta。
