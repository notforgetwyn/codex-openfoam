# FoamDesk：基于 Python 的 OpenFOAM 桌面仿真客户端

## 1. 项目简介

FoamDesk 是一个基于 Python 开发的 OpenFOAM 桌面客户端软件，目标是在 WSL/Linux 图形环境中，用图形界面封装 OpenFOAM 的几何准备、网格生成、求解运行、日志诊断和结果可视化流程。

项目不是 Web 平台，也不是前后端分离系统。FoamDesk 与 OpenFOAM 运行在同一个 WSL/Linux 环境中，客户端通过本地 Linux 进程直接调用 OpenFOAM 命令。

## 2. 项目背景

OpenFOAM 是开源 CFD 流体仿真工具箱，功能强大，但原生使用方式主要依赖命令行、字典文件和目录结构。对非 CFD 专业用户来说，主要门槛包括：

- 命令行操作复杂，需要手动执行 `blockMesh`、`snappyHexMesh`、`icoFoam` 等命令。
- 配置文件分散，需要理解 `controlDict`、`fvSchemes`、`fvSolution`、`0/U`、`0/p` 等 OpenFOAM 文件。
- 前处理和后处理流程繁琐，几何、网格、边界条件、结果查看需要多步手动操作。
- 出错信息偏底层，新手难以快速定位问题。

FoamDesk 的目标是把这些步骤封装成一个可视化桌面客户端，让用户通过界面完成从几何准备到结果查看的基础 CFD 仿真流程。

## 3. 核心功能

当前项目已经实现或正在完善以下功能：

- 项目与 Case 管理：支持项目选择、Case 树展示、当前 Case 记忆和恢复。
- 绘制几何：支持通过 CSV/界面方式构造基础几何，并生成 OpenFOAM 所需文件。
- STL 导入：支持导入 STL 几何文件，并提供预览、平移、旋转、缩放等几何放置能力。
- 网格生成：支持生成 `snappyHexMeshDict`，执行 `blockMesh`、`snappyHexMesh`、`checkMesh` 等前处理命令。
- 求解器准备：支持配置 `0/U`、`0/p`、流体物性、边界条件等求解前置文件。
- 求解器选择：支持 `icoFoam`、`simpleFoam`、`pisoFoam` 等求解器入口。
- 仿真参数配置：支持配置 `controlDict` 相关参数，例如总时间、时间步长、写出间隔等。
- 一键仿真流水线：串联几何、网格、求解准备和 OpenFOAM 命令执行流程。
- 日志与诊断：展示 OpenFOAM 原始日志，并对常见错误给出中文诊断建议。
- 结果索引：识别时间步、字段文件、网格状态和结果目录。
- 结果可视化：支持压力、速度、表面云图、切片、速度箭头、流线箭头、残差曲线等基础后处理。
- 报告导出：支持导出结果图片和 Markdown 报告。

## 4. 技术栈

- 编程语言：Python 3
- GUI 框架：PySide6 / Qt
- OpenFOAM 调用：本地 Linux 进程、`bash -lc`、OpenFOAM 环境脚本
- 数值结果处理：NumPy、SciPy
- 二维绘图：Matplotlib
- 三维数据读取与处理：VTK
- 本地配置：JSON 文件
- 测试框架：pytest
- 代码质量：Python 模块化分层、单元测试、语法检查
- 运行环境：WSL/Linux 图形环境

## 5. 系统架构

```text
┌────────────────────────────────────────────────────────────┐
│                        UI 展示层                            │
│  项目选择 / 项目主页 / 绘制几何 / 网格生成 / 求解器准备       │
│  求解器选择 / 仿真参数 / 求解运行 / 环境检查 / 设置 / 结果     │
├────────────────────────────────────────────────────────────┤
│                       应用控制层                             │
│  MainWindow / StartupWindow / 页面切换 / 状态栏 / 任务入口    │
├────────────────────────────────────────────────────────────┤
│                       业务服务层                             │
│  ProjectService / GeometryImportService / CaseParameterService│
│  ResultIndexService / ResidualPlotService / ReportExportService│
├────────────────────────────────────────────────────────────┤
│                    OpenFOAM 集成层                           │
│  OpenFOAMEnvironmentService / 命令执行 / 日志解析 / 错误诊断   │
├────────────────────────────────────────────────────────────┤
│                    本地进程调用层                             │
│  QProcess / bash -lc / source OpenFOAM bashrc / 工作目录隔离   │
├────────────────────────────────────────────────────────────┤
│                    文件与数据存储层                           │
│  workspace/projects / case 目录 / JSON 配置 / 结果索引 / 日志  │
└────────────────────────────────────────────────────────────┘
```

核心设计原则：

- UI 不直接拼 OpenFOAM 命令，命令执行通过服务层封装。
- Case 文件、配置文件、日志、结果文件统一放入项目工作区。
- 后处理可视化优先使用 Python/Matplotlib/VTK，不依赖 ParaView 作为主界面。
- 每个 Case 独立管理，避免不同仿真任务互相覆盖。

## 6. 项目目录结构

```text
codex_project/
├── AGENTS.md                    # 项目协作规则和长期记忆
├── README.md                    # 项目说明文档
├── pyproject.toml               # Python 项目配置
├── run.sh                       # WSL/Linux 启动脚本
├── config/                      # 本地应用配置
├── assets/                      # 测试几何、示例资源、静态资源
├── docs/                        # 阶段文档、设计文档和说明文档
├── src/
│   └── foamdesk/
│       ├── app/                 # 应用启动和依赖组装
│       ├── domain/              # 领域模型和数据结构
│       ├── integrations/        # OpenFOAM 等外部集成
│       ├── services/            # 项目、几何、参数、结果、报告等服务
│       └── ui/                  # PySide6 界面代码
├── tests/                       # 单元测试
├── tools/                       # 辅助工具脚本
├── workspace/                   # 本地项目与 OpenFOAM case 工作区
├── 对话记录/                    # 开发过程对话记录
└── 未来扩展.md                  # 中长期功能规划
```

## 7. 环境要求

推荐环境：

- Windows + WSL2 Ubuntu
- WSL 内已安装并编译 OpenFOAM
- WSL 图形环境可用，例如 WSLg
- Python 3.10 或更高版本
- Git

当前项目默认 OpenFOAM 环境脚本路径示例：

```text
/home/shihuayue/openfoam/OpenFOAM-dev/etc/bashrc
```

当前默认项目路径：

```text
/home/shihuayue/codex_project
```

## 8. 安装与运行

进入项目目录：

```bash
cd /home/shihuayue/codex_project
```

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

启动程序：

```bash
./run.sh
```

如果需要手动启动，也可以使用：

```bash
PYTHONPATH=src python3 -m foamdesk.app.main
```

运行测试：

```bash
PYTHONPATH=src pytest -q
```

## 9. 使用流程

典型仿真流程如下：

1. 启动 FoamDesk。
2. 在项目选择窗口中新建或打开项目。
3. 进入主界面后，在 Case 树中选择当前 Case。
4. 在“绘制几何”页面绘制几何，或通过“几何/CAD -> 导入 STL”导入几何。
5. 在“网格生成”页面检查几何、计算域和 STL 位置。
6. 生成 `snappyHexMeshDict`。
7. 执行网格生成流程：`blockMesh -> snappyHexMesh -> checkMesh`。
8. 在“求解器准备”页面配置 `0/U`、`0/p`、边界条件和流体物性。
9. 在“求解器选择”页面选择求解器，例如 `icoFoam`、`simpleFoam` 或 `pisoFoam`。
10. 在“仿真参数”页面配置总时间、时间步长、写出间隔等参数。
11. 在“求解运行”页面启动仿真并查看实时日志。
12. 在“结果”页面刷新结果索引，选择字段和显示方式。
13. 打开 3D 独立窗口查看表面云图、切片、速度箭头、流线箭头等结果。
14. 导出 PNG 图片或 Markdown 报告。

## 10. 功能截图

当前仓库暂未统一整理截图目录。建议后续将截图放入：

```text
assets/screenshots/
```

推荐截图清单：

- 项目选择窗口
- 主界面与 Case 树
- 绘制几何页面
- 网格生成页面
- 求解器准备页面
- 仿真参数页面
- 求解运行日志页面
- 结果页与残差曲线
- 3D 结果窗口：Surface、Slice、Vector、Streamline

## 11. 项目亮点

- 面向初学者：把 OpenFOAM 的命令行和配置文件封装成图形界面。
- 原生 WSL 集成：客户端和 OpenFOAM 在同一个 Linux 环境中运行，避免 Windows 调 WSL 的复杂路径问题。
- 工程化流程：按项目、Case、几何、网格、求解、结果分层组织。
- 中文诊断：对 OpenFOAM 常见错误提供中文解释和修复建议。
- Python 内置后处理：使用 Matplotlib/VTK 在客户端内部展示结果，减少对外部 ParaView 的依赖。
- 可扩展架构：后续可以继续扩展更多求解器、边界条件、物理模型、报告模板和行业场景。

## 12. 后续计划

短期计划：

- 完善从绘制几何到完整仿真的一键流水线。
- 增强求解器选择与实际运行命令的绑定。
- 完善 `simpleFoam`、`pisoFoam` 对应的字段文件和字典模板。
- 增强边界条件配置表格，支持每个 patch 独立设置。
- 优化 3D 可视化效果，包括透明外壳、切面、速度箭头、流线箭头和等值面。

中期计划：

- 支持更多计算域模板，例如风洞、管道、弯管、扩张管道。
- 增强 STL 导入、几何变换、几何检查和网格质量诊断。
- 增加批量参数扫描和多 Case 对比。
- 增加更完整的报告导出能力。

长期计划：

- 支持湍流、传热、多相流、可压缩流等更复杂物理模型。
- 支持插件化求解器和行业模板。
- 支持更专业的后处理模板和工程指标计算。
- 支持企业级数据管理、版本追踪和仿真流程复用。

## 13. 许可证

当前项目尚未正式指定开源许可证。

建议后续根据项目目标选择：

- MIT License：适合个人学习、开源协作和宽松复用。
- Apache License 2.0：适合需要专利授权条款的工程项目。
- GPL：适合希望衍生项目继续保持开源的场景。

在正式发布前，应在仓库根目录添加 `LICENSE` 文件，并在本章节中更新许可证说明。