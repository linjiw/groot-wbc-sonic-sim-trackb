# Guidance, 2026-08-19: Feasibility-First Whole-Body Navigation on SONIC

Received research plan (verbatim, as given). Saved as the standing direction for what the
counterfactual dataset is *for*: not a dataset paper alone, but scene–route–motion feasibility
learning executed by a frozen SONIC controller, ending on a real G1.

---

## 核心判断

对，你们真正想完成的，不应该只是一个"数据集能训练小型 selector"的论文，而应该是：

> **用 physics-verified counterfactual data 学到场景—路线—身体行为之间的可行性关系，再由 SONIC 执行这些行为，最终完成未知环境中的 humanoid navigation and traversal。**

这一步会让论文从"一个有意思的数据构造原则"，变成一篇完整的 embodied learning 论文。

但这里要非常清醒地看到：**当前工作还只完成了可行性标签与行为反转的物理证明，没有完成真正的场景感知闭环。** 稿子自己也把 Claim 1–4 标成 established，而 Claim 5"learner 是否真的使用几何信息"和 Claim 6"能否泛化到未针对轨迹设计的场景"都还是 not started。当前 decisive experiment 也刻意被定义成一个小型 behaviour selector，而不是连续视觉—运动策略。

所以现在还缺的，不是"再跑一次 SONIC fine-tuning"，而是下面这条完整链路：

```text
RGB-D / local map + navigation goal + proprioception
                         ↓
      scene–route–motion feasibility model
                         ↓
        body-aware route and behaviour planner
                         ↓
  nominal / crouch / tuck / sidestep motion tokens
                         ↓
        SONIC whole-body tracking controller
                         ↓
   Isaac Lab → sim2sim → real Unitree G1
```

我最推荐把这套系统暂时称为：

> **Feasibility-First Whole-Body Navigation on SONIC**

而 SweepCF 是让这个系统可以被训练的核心数据。

---

## 一、先区分两个完全不同的"训练"

这里很容易把两件事混在一起。

### 1. 用你们的 motion data fine-tune SONIC

这训练的是：

pi_SONIC(a_t | s_t, motion reference/token)

它回答的是：

> 给定 crouch、arm tuck 或 walk reference，G1 能不能稳定地把它执行出来？

这部分主要解决 tracking、balance、progress loss、sim-to-real robustness。SONIC 官方训练管线已经支持将 G1 motion 转成 `motion_lib` PKL，从 release checkpoint 继续 PPO fine-tuning、进行 tracking evaluation，并导出 ONNX。

但是这类训练**并不会自动让机器人看见障碍物后决定 crouch**。因为 SONIC 的低层控制器接收的是 motion reference/token，而不是"为什么应该选择这个 motion"。

### 2. 用 counterfactual data 训练场景条件决策模型

这训练的是：

F_theta(o_t, r, m) = P(motion m 沿 route r 成功 | o_t)

其中：

* o_t：RGB-D、局部地图、机器人状态；
* r：候选路线或局部路径；
* m：walk、crouch、tuck、sidestep 等 motion candidate；
* 输出：成功概率、预计碰撞、progress loss、adaptation cost 和 uncertainty。

然后规划器选择：

(r*, m*) = argmin_{r,m} [ L(r) + lambda * D(m, m_0) + mu * C_theta(o, r, m) ]
subject to F_theta(o, r, m) >= tau.

换句话说，**真正使用 SweepCF 核心监督的，是这个 feasibility model，而不只是 SONIC tracker**。

我的明确建议是：

> **主实验中冻结 SONIC，训练上层 feasibility model；SONIC fine-tuning 作为第二组正交实验。**

这样才能回答两个不同的问题：

1. SweepCF 是否真的教会了机器人依据几何选择行为？
2. SweepCF 中的 adapted motions 是否也能改善 SONIC 对困难动作的执行能力？

如果一开始把视觉模型、导航、SONIC controller 全部联合训练，最后成功率提高时，谁也说不清到底是 counterfactual supervision 有效，还是 controller 被调强了。你们现在最宝贵的优势恰恰是"因果归因干净"，不要在最后一步把它丢掉。

---

## 二、当前数据还缺一个关键维度：**route counterfactual**

你们现在的数据是：

> 同一条路线、同一个 start/goal，比较 nominal motion 和 adapted motion。

这对于 traversal 非常好，因为它能够明确说明：

> 路线不变，几何变化后，身体行为必须改变。

而且当前 `local_crouch` 和 `local_arm_tuck` 特意保持 root XY、yaw、duration、gait phase、start 和 goal 不变。

但这还不能支持真正的 navigation，因为它没有教机器人：

> 同一个 start 和 goal，如果存在多条路线，应该走哪一条？

因此需要增加第二类 counterfactual family：

### Route × Behaviour Counterfactual Family

同一个 scene、start、goal，构造：

* 短路线，但有低横梁，需要 crouch；
* 长路线，但可以直接 walk；
* 窄路线，需要 arm tuck；
* 宽路线，但距离更长；
* 某条路线任何已有行为都无法通过，必须 stop/replan。

数据标签从现在的 Y[S, m] 扩展为 Y[S, r, m]。

这背后的逻辑很重要：

* **Traversal**：给定路线，选择身体行为；
* **Navigation**：同时选择路线和身体行为；
* **Whole-body navigation**：路线长度和身体适应成本需要一起权衡。

因此有两个论文版本：

#### 最小而干净的版本

外部 planner 或 oracle 提供 waypoint/route，SweepCF policy 只决定 walk、crouch、tuck。

这应该被准确称为：**goal-directed route-conditioned traversal**

#### 更强的版本

模型同时评价 route 和 motion，机器人在不同走廊之间选择。

这才是真正的：**body-aware humanoid navigation**

我更推荐最终至少加入一个 forked-route benchmark：左边低但短，右边高但远。普通导航器只会看路径长度，而你们的系统会理解"身体需要付出什么代价"。

---

## 三、你们需要把 motion bank 从"论文分析样本"变成"可部署技能库"

这里有一个非常关键、也很容易被忽略的问题。

当前 matched pair 为了保持因果干净，要求 adapted motion 与 nominal motion 拥有相同 duration 和 forward schedule。但你们已经发现，crouched G1 无法按照直立行走同样的速度前进，随着 crouch 加深，endpoint lag 单调增加；目前较深的 crouch 即使避免了碰撞，也可能因为无法到达终点而被拒绝。

这意味着：

> **当前 causal clip 可以证明"为什么必须 crouch"，但不一定是适合真实 navigation 的 deployable crouch skill。**

最好的解决方式不是修改已经冻结的 counterfactual pair，而是明确维护两套 artifact。

### A. Causal reference

用于论文的数据集监督和 attribution：

* root path 相同；
* duration 相同；
* gait phase 相同；
* 只改变必要关节；
* 支持严格 matched comparison。

### B. Deployable skill

用于 SONIC fine-tuning 和真机执行：

* crouch 时允许适度 retiming；
* 根据 crouch depth 降低 forward velocity；
* transition 与当前 gait phase 对齐；
* 保留 start/goal 和局部 route，但不再强行保持逐帧 root XY 一致；
* 重新经过 SONIC trackability 和 physics verification。

这两者之间不是矛盾。前者回答"为什么行为必须改变"，后者回答"机器人怎样可靠地执行改变后的行为"。

### 最低限度的 motion bank

至少需要：

* nominal walk；
* local arm tuck；
* speed-consistent local crouch；
* stop / abstain；
* walk → adaptation → walk 的 phase-aligned transition。

更强的版本再加入：

* shallow / medium / deep 三档 crouch；
* left / right tuck；
* sidestep；
* turn-in-place 或 curved walk；
* step-over。

尤其是 transition 数据不能省。真实机器人不可能每次都从 motion frame 0 开始完美进入 crouch。你们需要训练并验证：

* 何时开始 adaptation；
* 从哪个 gait phase 进入；
* 何时恢复 nominal；
* 两个连续障碍之间如何切换；
* 选择改变时如何避免 action discontinuity。

---

## 四、现有 VLA 数据接口已经有骨架，但还没有真实训练数据

你们代码里已经做了一件很有价值的工作：定义了一个包含

* ego video；
* proprioceptive state；
* 64-dimensional `action.motion_token`；
* G1 reference qpos；
* left/right hand states；
* 40-step action horizon

的 SONIC/VLA 数据契约和 validator。

但目前 tiny fixture 自己明确写着：它只有 synthetic numeric rows 和 placeholder videos，**不能用于模型训练或性能评估**。

所以接下来真正要收集或渲染的是每个 episode 的同步数据：

| 数据层 | 必须包含的内容 |
|---|---|
| Perception | ego RGB、ego depth、相机 intrinsics/extrinsics、timestamp |
| Navigation | goal、local waypoint、route candidate、route progress |
| Proprioception | qpos、qvel、projected gravity、base velocity/orientation |
| Behaviour | candidate ID、motion token、reference qpos、adaptation amplitude |
| Temporal | adaptation onset、offset、gait phase、remaining horizon |
| Physics label | accepted/rejected、contact body、force/impulse、drift、progress |
| Counterfactual identity | family ID、scene ID、route ID、nominal ID、operator ID |
| Safety | no-feasible-candidate、abstain、stop/replan label |

特别要增加一个字段：`distance_to_constraint`，或者等价的 route-progress coordinate。否则模型可能知道要 crouch，却不知道**什么时候** crouch。

你们项目的核心一句话是"scene variation is not scene-conditioned behaviour"。同理，单纯把每个 episode 配上一个视频，并不自动产生 scene-conditioned timing。必须让视觉变化、障碍相对距离和 motion onset 之间存在监督关系。

---

## 五、Isaac Lab 里还需要一个真正的 closed-loop benchmark

目前的四格 counterfactual rollout 是固定 reference 的 physics evaluation。下一步需要将它包装成一个可训练、可重复运行的 Isaac Lab environment。

### Observation

主模型建议使用：

o_t = { depth history, local goal, route direction, proprioception, current motion phase }

RGB 可以作为更困难的视觉层；**第一版先让 depth 跑通**。因为你们要先验证几何条件决策，而不是把结果押在纹理识别和 sim-to-real appearance gap 上。

继续保留现有四个 observation tier 很合理：

1. privileged geometry；
2. depth；
3. RGB；
4. blind。

### Action

不要直接输出 29-DoF torque 或 joint target。最适合现有数据的是：

a_t = { motion ID, amplitude, activation onset, duration, target velocity, heading }

或者输出一段 motion-token chunk，由 SONIC 负责低层 tracking。

### Closed-loop execution

不能只在 episode 开始时选择一次。推荐：

1. 观察局部几何；
2. 给所有 route–motion candidates 打分；
3. 选择当前最小代价的可行方案；
4. 只执行前一小段；
5. 重新观察；
6. 继续、切换或 stop。

这会把当前 selector 扩展成 receding-horizon planner，同时避免一次视觉误判导致整段 motion 无法恢复。

### Task ladder

Isaac Lab benchmark 应当分为四级，而不是一上来做复杂迷宫：

* **T1：Single-constraint traversal** — 单个 ceiling 或 lateral wall，固定 route。验证机器人是否会在正确位置选择和触发 adaptation。
* **T2：Sequential traversal** — 同一条 route 上连续两个障碍：先 tuck；恢复 walk；再 crouch；最后到达 goal。这一层检验 transition 和 temporal reasoning。
* **T3：Route-choice navigation** — 同一个 start/goal，有两条或多条路线，比较距离与 adaptation cost。
* **T4：Procedural navigation course** — 未知 obstacle order、turn、width、height 和 route topology，真正做 goal-conditioned navigation。

### Sim-to-real randomization

至少需要覆盖：friction；actuator strength 和 latency；joint damping；payload / mass；camera extrinsics；depth noise 和 missing pixels；obstacle height/width bias；lighting/material；initial pose 和 heading；control/vision delay。

但这里不要把 randomization 当成"越多越好"。真正重要的是先测量真机误差可能来自哪里，然后让 randomization 对准那些误差，而不是把所有参数随便抖动。

---

## 六、训练应该分成三阶段，而不是一个 end-to-end 大训练

### Stage A：SONIC motion adaptation fine-tuning

将 verified deployable skills 转成 SONIC `motion_lib`，从 release checkpoint 开始 fine-tune。

训练集不能只有 crouch/tuck，否则很可能 catastrophic forgetting。应该混合：原始 locomotion rehearsal set；nominal motions；adapted motions；transition motions；perturbation/recovery motions。

关键指标：adapted-motion success rate；MPJPE；route progress；edit survival（reference adaptation 有多少真正被 controller 执行出来）；collision/contact；原始 locomotion holdout 上的 forgetting。

你们现有 paired learning-curve infrastructure 已经能够按 checkpoint 比较 success、progress、MPJPE 和 AUC，可以直接复用到这一阶段，而不是重新设计一套训练审计。

### Stage B：Counterfactual feasibility pretraining

对每个 (S, r, m) cell，训练 compatibility model。

L = L_success + lambda_1 * L_pairwise-rank + lambda_2 * L_cost + lambda_3 * L_uncertainty

其中最重要的是 pairwise ranking：

* easy scene：nominal 应高于 adapted；
* hard scene：adapted 应高于 nominal；
* no-feasible scene：全部低于 threshold，并触发 abstain。

这比直接训练"scene → crouch"分类器更好，因为模型学到的是：这个 motion 在这个 geometry 里是否可行。它可以自然扩展到新 motion、新 route 和新的 candidate bank。

### Stage C：Closed-loop adaptation

先通过 behaviour cloning 学会 selector/planner，再使用 Isaac Lab rollout 做：DAgger；offline-to-online fine-tuning；或小规模 PPO。

我不会把纯 RL 作为第一步。你们已经花了很大力气获得高质量 counterfactual labels，如果一开始用 sparse success reward 重新探索，相当于放弃了数据集最有价值的监督。

---

## 七、论文实验矩阵应该怎样设计

最重要的是保持每一组实验都能回答一个明确问题。

| Research question | 固定什么 | 比较什么 | 能支持的结论 |
|---|---|---|---|
| RQ1 数据是否有用 | frozen SONIC、相同模型、相同样本数 | decorated vs random vs SweepCF | counterfactual supervision 是否改善行为选择 |
| RQ2 模型是否真的看几何 | 相同 SweepCF 数据 | privileged vs depth vs RGB vs blind/shuffle | 信号来自场景而非 shortcut |
| RQ3 motion data 是否改善执行 | 相同上层 planner | frozen vs SweepCF-finetuned SONIC | 数据是否改善困难动作 trackability |
| RQ4 能否闭环组合 | 相同 controller | open-loop selection vs receding-horizon | 是否能处理 timing、transition 和连续障碍 |
| RQ5 能否导航 | 相同 scene/start/goal | shortest-path vs always-adapt vs body-aware planner | 是否联合考虑路线与身体代价 |
| RQ6 能否迁移 | 同一 ONNX/决策阈值 | Isaac Lab、MuJoCo sim2sim、real G1 | sim-to-real generalization |

主指标继续保留你们已有的：counterfactual choice accuracy；unsafe-choice rate；unnecessary-adaptation rate；success minus adaptation cost。

再增加 navigation/traversal 指标：goal success；collision-free success；route completion；time/path length；adaptation duration；adaptation onset error；excessive switching rate；abstention correctness；fall/emergency-stop rate；sim-to-real performance drop。

特别不要只报 success rate。一个永远 crouch、永远 tuck 的策略可能非常安全，却完全没有理解场景。你们已经准确意识到这一点，应当贯穿所有 end-to-end 实验。

---

## 八、数据规模需要怎样补

当前稿子已经诚实指出，apparatus 不再是主要瓶颈，真正缺的是 independent families 和 learning evidence；scene instances 也不能被冒充为独立 behaviour families。这个统计纪律要继续保留。

我建议把规模分成两种口径：

### 科学统计单位

用于论文置信区间和 generalization claim：

* 至少 24–30 independent counterfactual families；
* 8–12 个不同 nominal motions；
* 至少 overhead、lateral 两个 regime；
* 最好加入一个新的 regime，例如 sidestep/step-over；
* scene-first、nominal-held-out、operator-held-out 三种 split。

### 神经网络训练样本

用于视觉模型训练：

* 每个 independent family 可以生成大量 camera pose、texture、lighting、尺寸扰动和 dynamics variants；
* 可以得到数千个 training episodes；
* 但这些只能算 augmentation，不能被写成数千个独立 family。

换句话说，**模型训练需要规模，科学结论需要独立性**。这两种数字必须分开报告。

你们当前 30-scene frozen test set 仍然适合作为预注册的第一轮 decisive test。不要因为后面要做完整系统，就修改或取消这个实验。它应当成为 Stage 0：先证明 counterfactual supervision 在受控 selector 上成立，再把同样的原则推进到 closed-loop。

---

## 九、真机前还缺哪些工程闭环

### 1. SONIC streaming bridge

最稳妥的顺序是：

* **调试路径**：直接通过 ZMQ v1 发送 G1 `qpos/qvel` reference，先确认 candidate motion、phase 和 switching 能在 MuJoCo 与真机上被稳定跟踪。
* **最终路径**：将 candidate motion 编码成 64D SONIC motion tokens，通过 token streaming 或 manager interface 发送，同时将 target velocity、heading 和 navigation waypoint 交给高层 manager。

SONIC 官方 streaming interface 已经支持 joint-based reference 和 direct token streaming；但 navigation manager、motion-token stream 和 route command 如何同步，仍需要你们自己完成统一接口。

### 2. Isaac Lab → deployment observation parity

需要逐项核对：joint order；quaternion convention；frame convention；camera frame；projected gravity；token normalization；history buffer；action horizon；controller frequency；timestamp/latency。

这个 parity test 应该是自动测试，而不是靠视频"看起来差不多"。

### 3. C++ / ONNX deployment

当前 encoder、decoder、planner ONNX 和 observation config 已经存在，但你们自己的 deploy-readiness 文档仍列出了 C++ toolchain、TensorRT、ONNX Runtime 等构建缺口。在真机实验前必须完成：`deploy.sh sim`；`deploy.sh real`；同一模型 sim/real parity；watchdog；packet timeout；emergency stop；inference latency logging。

### 4. 真实感知

至少需要一台固定安装的 RGB-D camera，并完成：camera-to-base extrinsic calibration；depth scale calibration；timestamp synchronization；obstacle dimension ground truth。

外部 AprilTag/Vicon 可以用于评估，但不要作为 policy input，否则真机结果会变成 privileged perception demo。

### 5. 安全实验设施

真机初期建议使用：foam ceiling；flexible side walls；safety gantry；human spotter；hardware E-stop；uncertainty-triggered stop；conservative velocity。

而且第一批真机动作应优先从 `arm tuck / lateral passage` 开始。你们已经测到上肢 adaptation 的执行保真度明显高于下肢 crouch；先用更稳定的 regime 验证完整闭环，再推进低矮横梁，会更理性。

---

## 十、我建议的实际推进顺序

* **Gate 0：保留并完成现有 decisive experiment** — 按预注册运行 decorated / random / SweepCF selector comparison。它负责回答 Claim 5 的第一版。
* **Gate 1：完成真实数据管线** — 把 placeholder fixture 替换为：真实 ego RGB-D；proprioception；motion token；route progress；physics labels；actual SONIC loader smoke test。
* **Gate 2：完成 frozen-SONIC single-obstacle closed loop** — 先在 Isaac Lab 中做到：depth → choose walk/tuck/crouch → SONIC executes → reach goal。这是从 dataset paper 跨到 embodied system paper 的关键一跳。
* **Gate 3：完成 deployable motion-bank fine-tuning** — 重点解决：crouch retiming；phase-aligned transitions；progress loss；original locomotion forgetting。
* **Gate 4：增加 sequential traversal** — 同一 episode 内两次以上的 behaviour transition。
* **Gate 5：增加 route counterfactual 与 body-aware navigation** — 让机器人在短而困难、长而容易的路线之间选择。
* **Gate 6：sim2sim 和真机** — 先 fixed route、单障碍，再未知尺寸，最后 multi-obstacle route。

---

## 最终最合适的论文故事

完成上述闭环后，论文可以非常清楚地讲成三项贡献：

1. **SweepCF dataset**：通过 matched counterfactual construction 和 frozen SONIC physics verification，把 scene variation 转化为可归因的 behaviour supervision；
2. **Scene–route–motion feasibility learning**：从 RGB-D 和 goal 预测候选路线与 whole-body motion 的可行性，而不是只识别场景；
3. **SONIC-based embodied validation**：在 Isaac Lab 和真实 G1 上完成 unseen-geometry traversal 与 body-aware navigation。

最有力量的一句方法描述会变成：

> **We do not train a humanoid to map rooms directly to joint actions. We train it to predict which route–motion combinations physics will permit, then let SONIC execute the least costly feasible one.**

这既保留了你们目前最珍贵的"physics decides"与因果干净性，又真正抵达了老师所说的 navigation/traversal 和真机说服力。

最重要的战略判断是：**主论文先冻结 SONIC 来证明 SweepCF supervision 的价值；SONIC fine-tuning 用来证明困难技能可以被改善；route counterfactual 则把 traversal 提升为真正的 navigation。** 这三条线分开训练、最后闭环组合，会比直接做一个庞大的 end-to-end policy 更强，也更容易让审稿人相信。
