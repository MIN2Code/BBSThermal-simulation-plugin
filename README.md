# BBSThermal-simulation-plugin

**Local FDM thermal simulation & print optimization for Bambu Studio** — 本地运行的 3D 打印热仿真分析与打印参数优化工具，通过 Bambu Studio 的 Helio 集成入口直接在切片器内显示结果，**全程本地计算，无云端依赖**。

> 灵感来自 [Helio Additive](https://www.helioadditive.com/) 的 Assess/Enhance 工作流：本项目用开源技术栈复刻了"切片后 G-code → 体素级热历史仿真 → 热质量指数（TQI）热图 → 自动速度优化"的完整闭环，并逆向对齐了其 Bambu Studio 集成协议，使官方 Helio 界面可以直接消费本地引擎的结果。

## 功能

- **热仿真分析（Assess）**：对切片后的 G-code 做体素级热历史仿真，逐段计算热质量指数 TQI（−100 太冷/弱结合 → 0 理想 → +100 太热/下垂），3D 逐层热图 + 风险报告
- **打印速度优化（Enhance）**：迭代调速（冷层提速缩短层时、热层降速防热积累），带防回退保底，输出可直接打印的优化 G-code/.3mf
- **Bambu Studio 直连**：实现 Helio 集成协议（GraphQL），改两行配置即可让官方 Helio 界面消费本地引擎结果
- **校准芯片**：一次 15 分钟的标准打印 + 逐段掰断测试，自动拟合你机器与耗材的真实热参数（结合阈值/风扇效率/再热系数），生成耗材档案并自动套用
- **纯本地**：不联网、不上传任何数据

## 快速开始

```bat
git clone https://github.com/MIN2Code/BBSThermal-simulation-plugin.git
cd BBSThermal-simulation-plugin
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
run.bat                # 启动服务 → http://127.0.0.1:8760/
```

1. 浏览器打开 `http://127.0.0.1:8760/`，拖入切片输出的 **.gcode.3mf**（或纯 .gcode）
2. 点「开始热仿真」查看 TQI 热图与报告
3. （可选）点「优化打印速度」，下载优化后的 G-code

### 接入 Bambu Studio（在切片器内直接显示）

先启动本服务，然后**完全关闭 Bambu Studio**，双击运行：

```bat
bridge\patch_bs_config.bat
```

它会修改 `%APPDATA%\BambuStudio\BambuStudio.conf`（幂等，可重复运行）：

```
helio_api_china = http://127.0.0.1:8760/graphql/helio
helio_api_other = http://127.0.0.1:8760/graphql/helio
helio_pat_china / helio_pat_other = local-pat（本地引擎不校验 PAT，任意非空即可）
helio_enable = true
```

之后在 Bambu Studio 中正常切片、点击 Helio 按钮：热指数仿真与速度优化全部由本地引擎完成，结果显示在 Bambu Studio 原生界面。引擎未启动时仅 Helio 功能不可用，不影响切片。

也可以只用无界面的后处理脚本桥接（切片完成后自动提交分析并打开浏览器）：

```
python bridge\bambu_postprocess.py <切片输出的gcode路径>
```

## 工作原理

```
G-code ──解析──▶ 挤出段序列（位置/速度/风扇/时间轴）
          │
          ▼
体素热仿真（默认 1.5mm 网格，numba 加速）
  ├─ 沉积混入：珠体积按热容加权，温度经流量降额修正（打得快→有效熔温低）
  ├─ 热传导：仅已沉积体素之间（7 点差分，空气格不导热）
  ├─ 散热：外露表面对流（风扇顶面全额/侧面 40%）+ 首层热床换热 + 薄壁快冷
  └─ 喷嘴驻留加热：打印中的喷嘴是移动热源，慢速=停留久=烤暖台阶
          │
          ▼
每段沉积时采样台阶邻域最热材料温度 = 界面温度
（经接触再热系数 κ 修正 —— 新珠会重熔基面表层）
          │
          ▼
按材料窗口映射 TQI：cold_below 以下 −100（弱结合）→ 0 理想 → hot_above 以上 +100
```

速度优化器按层迭代：冷层提速（缩短层时 → 界面更暖）、热层降速，粗细结合仿真保证 30 万段的件在 2 分钟内完成。

## 校准

TQI 窗口与散热参数默认为手册近似值——**风险定位可靠，绝对值仅供参考**。校准页（`/calibration`）提供标定流程：

1. 下载校准芯片 G-code（30×30mm 薄壁塔，6 段 = {风扇 0/50/100%} × {慢速/快速}，约 15 分钟）
2. 打印后逐段掰断：沿层裂开 = 易断，掰不动 = 结实
3. 网页填表 → 引擎网格搜索拟合 → 生成耗材档案（结合阈值 T_bond、κ、η、风扇倍率），之后仿真与优化自动套用

拟带回自诊断：如果所有参数组合都无法解释掰断结果（常见于耗材受潮、风道改装），会如实提示而不是硬拟合。

## 性能参考

| 场景 | 耗时 |
|------|------|
| 解析 30 万段 / 568 层 | ~4s |
| 热仿真（1.5mm 体素，numba CPU） | ~21s |
| 3 轮速度优化（粗细结合） | ~70s |

> 可选 GPU 路径（`SimConfig(device="gpu")`，需 torch+CUDA）：实测在 Windows WDDM 下
> 因小核函数启动开销反而慢于 CPU，保留为实验性代码；Linux 或未来 CUDA Graphs
> 优化后可再启用（见 `requirements-gpu.txt`）。

## 项目结构

```
backend/
  gcode/        解析器（Slic3r 系/Cura/Bambu、圆弧、3MF 容器）
  thermal/      体素热仿真引擎（numba 内核）、TQI、优化器
  calibration/  校准芯片生成与拟合
  helio_api/    Helio GraphQL 协议本地仿真层（Bambu Studio 直连）
  profiles.py   耗材档案存取
  api/          REST API
frontend/       three.js 3D 查看器 + 校准页（无构建工具，three.js 本地 vendor）
bridge/         Bambu Studio 后处理桥接 + 配置修补脚本
tests/          28 项单元测试
samples/        合成测试 G-code
```

## 已知限制（诚实版）

- TQI 绝对值未经实物标定前仅供参考；风险区域的位置与相对排序可靠
- 时间估计未建模加减速，比真实打印偏快 ~20%
- 1.5mm 体素下逐层 TQI 有换层锯齿（体素调 1.0 缓解）
- 不模拟：桥接/悬垂细节、应力翘曲、湿料/堵头等机械故障
- 优化只调整打印速度（与 Helio 一致），不改变路径与挤出

## 测试

```bat
.venv\Scripts\python -m pytest tests\ -q
```

## 致谢

- [Helio Additive](https://www.helioadditive.com/) — 工作流与协议参考（本项目的集成层为独立实现，未使用其代码）
- [Bambu Studio](https://github.com/bambulab/BambuStudio) 与开源切片器社区
- 学术基础：FDM 过程仿真与层间结合相关公开研究

## License

MIT
