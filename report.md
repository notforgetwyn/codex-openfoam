# 项目交接文档

## 1. 项目基本信息

- 项目名称：FoamDesk / codex-openfoam
- 项目类型：基于 Python 的 OpenFOAM 桌面仿真客户端软件
- 项目目标：用桌面 GUI 封装 OpenFOAM 的项目管理、几何准备、网格生成、求解运行、日志诊断和结果可视化流程，让非 CFD 专业用户也能完成基础流体仿真。
- 使用场景：在 WSL/Linux 图形环境中，用户通过 FoamDesk 创建项目和 Case，导入或绘制几何，生成 OpenFOAM case 文件，调用本机 OpenFOAM 命令运行仿真，并在客户端内查看残差曲线、速度/压力场、表面云图、切片、速度箭头和流线箭头等结果。
- 当前开发阶段：MVP 后期到工程化增强阶段。项目已经从最小 OpenFOAM 命令调用，推进到“几何/网格/求解准备/仿真参数/结果展示”的多页面工作流。
- 当前已经完成到什么程度：已经具备 Python 桌面主界面、项目选择、Case 树、绘制几何、STL 导入、网格生成入口、求解器准备、求解器选择、仿真参数配置、求解运行日志、OpenFOAM 错误诊断、结果索引、残差曲线、3D 独立结果窗口和基础报告导出。当前仍不是成熟商业 CFD 软件，完整复杂工业仿真、复杂 CAD 内核、真实高质量网格交互编辑、所有求解器模板和完整湍流模型仍未完成。

当前仓库状态说明：截至本交接文档生成时，本地 `main` 相对 `origin/main` 显示 `ahead 1`，最近本地提交为 `6b6a976 docs: update project readme and visualization notes`。还有未跟踪运行日志和临时目录，例如 `foamdesk_debug.log`、`foamdesk_run.log`、`assets/test_geometries/stl folader/`，这些不应随意提交。

## 2. 项目技术栈

- 编程语言：Python 3.12 及以上。
- GUI 框架：PySide6 / Qt。
- 三维数据处理：VTK，用于读取 OpenFOAM 结果、STL、PolyData、流线等数据。
- 图形绘制：Matplotlib，用于残差曲线和当前 3D 独立窗口的 Matplotlib 3D 可视化。
- 数值处理：NumPy，部分插值和网格相关处理依赖 SciPy。
- OpenFOAM 集成：通过 WSL/Linux 本地进程直接调用 OpenFOAM 命令，依赖 `bash -lc` 和 OpenFOAM 环境脚本。
- 数据存储方式：本地文件系统、JSON 配置、OpenFOAM 标准 case 目录、CSV 结果、Markdown 报告。
- 测试框架：pytest。
- 构建方式：setuptools，配置在 `pyproject.toml`。
- 启动方式：`run.sh` 或 `PYTHONPATH=src python3 -m foamdesk.app.main`。
- 运行环境：Windows + WSL2 Ubuntu + WSLg 图形环境 + WSL 内已安装 OpenFOAM。
- 默认 OpenFOAM 环境脚本：`/home/shihuayue/openfoam/OpenFOAM-dev/etc/bashrc`。
- 默认项目目录：`/home/shihuayue/codex_project`。

当前 `pyproject.toml` 主要依赖：

```text
matplotlib>=3.6,<4.0
PySide6>=6.8,<7.0
PyYAML>=6.0,<7.0
vtk>=9.6,<10.0
pytest>=8.0,<9.0  # dev
black / mypy / ruff  # dev
```

## 3. 项目整体架构

### 3.1 分层设计

```text
┌────────────────────────────────────────────────────────────┐
│                        UI 展示层                            │
│  StartupWindow / MainWindow / 页面 Tab / 3D 独立窗口         │
│  项目主页 / 绘制几何 / 网格生成 / 求解器准备 / 结果展示       │
├────────────────────────────────────────────────────────────┤
│                       应用控制层                             │
│  QApplication / ApplicationContext / 页面切换 / QProcess     │
├────────────────────────────────────────────────────────────┤
│                       业务服务层                             │
│  ProjectService / GeometryImportService / CaseParameterService│
│  ResultIndexService / ResidualPlotService / ReportExportService│
├────────────────────────────────────────────────────────────┤
│                    OpenFOAM 集成层                           │
│  OpenFOAMEnvironmentDetector / OpenFOAM 命令执行 / 日志解析   │
├────────────────────────────────────────────────────────────┤
│                    本地文件与数据层                           │
│  workspace/projects / case 文件 / JSON 配置 / CSV / Markdown │
└────────────────────────────────────────────────────────────┘
```

### 3.2 各层职责

- UI 展示层负责显示页面、按钮、表格、输入框、日志面板、3D 图窗，不应直接承载复杂 OpenFOAM 文件生成逻辑。
- 应用控制层负责启动 Qt 应用、创建服务上下文、页面切换、进程状态管理和状态栏提示。
- 业务服务层负责项目、Case、几何导入、参数写入、结果索引、残差分组、报告导出等可测试逻辑。
- OpenFOAM 集成层负责检查 OpenFOAM 环境、调用 Linux 命令、解析日志和识别错误。
- 文件与数据层负责 `workspace/projects` 中的 OpenFOAM case、`.foamdesk` 元数据、`foamdesk_results`、配置文件和测试资源。

### 3.3 核心模块关系

- `foamdesk.app.main` 创建 `QApplication` 并启动项目选择/主窗口。
- `ApplicationContext` 负责集中创建服务实例，并注入到 UI。
- `StartupWindow` 使用 `ProjectService` 和 `AppSettingsService` 新建、打开、记忆项目。
- `MainWindow` 组织所有功能页面，通过服务层读写项目和 OpenFOAM case。
- `ProjectService` 负责项目、Case、计算域模板、边界条件、几何元数据。
- `GeometryImportService` 负责 STL 导入、变换、`snappyHexMeshDict` 生成。
- `OpenFoamCaseParameterService` 负责写 `controlDict`、`fvSchemes`、`fvSolution`、`transportProperties`、`0/U`、`0/p` 等文件。
- `OpenFoamLogMetricService` 和 `MetricExportService` 负责从求解日志中提取残差、Courant 数等指标并导出。
- `ResultIndexService` 和 `OpenFoamVtkService` 负责识别结果时间步、字段和 VTK 可视化数据。
- `ResidualPlotService` 将 `residuals.csv` 分组为可绘图曲线，当前已经处理 PISO 中同一时间步多次 `p` 修正的问题。

### 3.4 数据流和调用流程

典型数据流：

```text
用户选择项目
-> StartupWindow 调用 ProjectService
-> MainWindow 打开项目并加载 Case 树
-> 用户绘制/导入几何
-> GeometryImportService 保存 STL/几何元数据
-> 用户生成网格配置
-> ProjectService / GeometryImportService 写 system/blockMeshDict 和 snappyHexMeshDict
-> MainWindow 通过 QProcess 调用 blockMesh/snappyHexMesh/checkMesh
-> 用户配置求解器准备和仿真参数
-> OpenFoamCaseParameterService 写 0/U、0/p、constant/transportProperties、system/controlDict 等
-> MainWindow 调用 OpenFOAM 求解器
-> OpenFoamLogMetricService 解析日志
-> MetricExportService 写 foamdesk_results/residuals.csv 和 metrics.json
-> ResultIndexService / OpenFoamVtkService 读取结果字段和时间步
-> VtkViewerDialog 显示 Surface、Slice、Vector、Streamline 等视图
```

### 3.5 用户操作流程

1. 运行 `./run.sh`。
2. 在项目选择窗口中新建或打开项目。
3. 在主界面 Case 树中选择当前 Case。
4. 在“绘制几何”页面导入 CSV 或手动构造几何。
5. 在“几何/CAD”菜单中导入 STL，必要时调整位置、旋转、缩放。
6. 在“网格生成”页面选择计算域模板，生成 `snappyHexMeshDict`，执行 `blockMesh`、`snappyHexMesh`、`checkMesh`。
7. 在“求解器准备”页面设置物性、`0/U`、`0/p` 和 patch 边界条件。
8. 在“求解器选择”页面选择 `icoFoam`、`simpleFoam` 或 `pisoFoam`。
9. 在“仿真参数”页面配置 `controlDict` 相关参数。
10. 在“求解运行”页面启动仿真并查看日志。
11. 在“结果”页面刷新结果，选择字段、时间步和显示方式。
12. 在 3D 独立窗口查看云图、切片、矢量箭头、流线箭头，或导出 PNG/Markdown 报告。

### 3.6 目录结构

```text
codex_project/
├── AGENTS.md                       # 项目级协作规则和长期需求记忆
├── README.md                       # 项目说明文档
├── report.md                       # 本交接文档
├── pyproject.toml                  # Python 包、依赖、pytest、ruff、black 配置
├── run.sh                          # WSL/Linux 启动脚本
├── 未来扩展.md                     # 中长期需求和未来功能池
├── config/                         # 应用配置文件
├── assets/
│   └── test_geometries/            # 示例 STL、CSV 几何、计算域数据
├── src/
│   └── foamdesk/
│       ├── app/
│       │   ├── main.py             # 应用入口
│       │   └── bootstrap.py        # ApplicationContext 服务组装
│       ├── domain/
│       │   └── models.py           # dataclass 领域模型
│       ├── integrations/
│       │   └── openfoam/
│       │       └── environment.py  # OpenFOAM 环境检测
│       ├── services/               # 业务服务层
│       └── ui/                     # PySide6 UI 层
├── tests/
│   └── unit/                       # 单元测试
├── tools/                          # 辅助脚本
├── workspace/                      # 本地项目、Case 和 OpenFOAM 运行结果
└── 对话记录/                       # 迭代过程 Markdown 对话记录
```

重要说明：`workspace/` 是运行时工作区，里面包含实际 OpenFOAM case 和结果，不应把所有运行数据都当作功能代码提交。日志文件 `foamdesk_debug.log`、`foamdesk_run.log` 一般不提交。

## 4. 已完成的功能

### 模块 1：项目选择与项目管理

- 功能作用：启动时先进入项目选择窗口，支持创建、打开、刷新、删除项目，并记忆上次项目。
- 涉及文件：`src/foamdesk/ui/startup_window.py`、`src/foamdesk/services/project_service.py`、`src/foamdesk/services/settings_service.py`。
- 实现逻辑：`StartupWindow` 调用 `ProjectService` 管理 `workspace/projects` 下的项目目录，使用 `AppSettingsService` 记录 `last_project_path`。
- 当前状态：可运行，用户已经使用过项目选择和恢复上次项目。
- 存在问题：项目选择窗口样式和原生窗口边框曾多次调整；需要继续验证 WSLg 下窗口边框、缩放、关闭行为是否一致。

### 模块 2：主界面工作台与 Case 树

- 功能作用：提供类似 VS Code 的深色工作台，左侧 Case 树，右侧多个功能页面，下方日志/任务/问题面板。
- 涉及文件：`src/foamdesk/ui/main_window.py`、`src/foamdesk/ui/theme.py`。
- 实现逻辑：`MainWindow` 构建菜单栏、Case 树、Tab 页面和底部日志面板。Case 树只显示当前项目下的 Case，并用状态栏显示当前 Case。
- 当前状态：可运行，已经支持项目主页、绘制几何、网格生成、求解器准备、求解器选择、仿真参数、求解运行、环境检查、设置、结果等页面。
- 存在问题：`main_window.py` 文件体积很大，很多 UI 和业务编排仍集中在一个文件中，后续应逐步拆分，但不要一次性大重构。

### 模块 3：绘制几何

- 功能作用：用户可以通过界面或 CSV 方式构造基础几何，生成 OpenFOAM 几何/字典相关输入。
- 涉及文件：`src/foamdesk/ui/main_window.py`、`assets/test_geometries/*.csv`。
- 实现逻辑：绘制几何页面支持点、边、arc、blocks 等配置，并用 Matplotlib 预览几何。已经新增过放大预览按钮和 CSV 示例，如球体相关 CSV。
- 当前状态：基础可用。
- 存在问题：CSV 几何生成复杂形体能力有限；球体等曲面在 blockMesh 表达上仍是近似构造；专业几何建模能力不足。

### 模块 4：STL 导入与几何变换

- 功能作用：支持导入 STL 到 OpenFOAM case 的 `constant/triSurface`，并允许平移、旋转、缩放、预览。
- 涉及文件：`src/foamdesk/services/geometry_import_service.py`、`src/foamdesk/ui/main_window.py`、`assets/test_geometries/*.stl`。
- 实现逻辑：`GeometryImportService` 使用 `vtkSTLReader` 读取 STL 点面数据，应用 `StlTransform`，保存几何资产元数据，并为 `snappyHexMeshDict` 提供几何名和边界信息。
- 当前状态：基础可用，菜单“几何/CAD -> 导入 STL”已接入。
- 存在问题：不是完整 CAD 内核，不支持 STEP/IGES/CATIA/SolidWorks 真实解析；复杂 STL 修复、非流形检测、自动特征清理尚未完成。

### 模块 5：计算域模板与网格生成

- 功能作用：为不同场景选择计算域模板，生成 `blockMeshDict` 和 `snappyHexMeshDict`，执行网格生成和检查。
- 涉及文件：`src/foamdesk/services/project_service.py`、`src/foamdesk/services/geometry_import_service.py`、`src/foamdesk/ui/main_window.py`。
- 实现逻辑：`ComputationDomainTemplate` 描述计算域尺寸、网格数、patch 名、推荐 `locationInMesh`。UI 中提供“网格生成”页面，串联 `blockMesh`、`snappyHexMesh -overwrite`、`checkMesh`。
- 当前状态：基础可运行，已支持简单/中等/高级模板、管道/弯管等预览方向的增强。
- 存在问题：真实 OpenFOAM 计算域仍主要依赖 blockMesh 矩形背景网格；“视觉上的管道域”和“OpenFOAM 实际网格域”之间仍需要进一步统一，避免用户误解。

### 模块 6：求解器准备

- 功能作用：集中配置求解器运行前必要文件，包括 `0/U`、`0/p`、流体物性、patch 边界条件。
- 涉及文件：`src/foamdesk/ui/main_window.py`、`src/foamdesk/services/case_parameter_service.py`、`src/foamdesk/services/project_service.py`。
- 实现逻辑：UI 读取当前 patch，用户可配置入口速度、出口压力、壁面类型、流体类型、密度、运动黏度、动力黏度等，再写入 OpenFOAM case 文件。
- 当前状态：基础可用。
- 存在问题：patch 级边界条件还不够专业，复杂边界类型、对称面、周期边界、物体表面自动识别仍需完善。

### 模块 7：求解器选择与仿真参数

- 功能作用：允许用户选择 `icoFoam`、`simpleFoam`、`pisoFoam`，并配置 `controlDict`、`fvSchemes`、`fvSolution` 等基础参数。
- 涉及文件：`src/foamdesk/ui/main_window.py`、`src/foamdesk/services/case_parameter_service.py`。
- 实现逻辑：`SimulationParameters` 描述求解器名、总时间、时间步长、写出间隔、最大迭代、残差阈值、材料、湍流模型、数值格式和 fvSolution 预设；服务层写 OpenFOAM 字典。
- 当前状态：部分可用。
- 存在问题：求解器选择和一键流水线的绑定仍需继续验证；`simpleFoam`、`pisoFoam` 所需场文件和配置模板还不完整，湍流模型支持仍偏入口级。

### 模块 8：OpenFOAM 命令执行与日志诊断

- 功能作用：直接调用 WSL/Linux 本机 OpenFOAM 命令，实时展示 stdout/stderr，并对错误做中文诊断。
- 涉及文件：`src/foamdesk/ui/main_window.py`、`src/foamdesk/integrations/openfoam/environment.py`、`src/foamdesk/services/log_diagnostic_service.py`、`src/foamdesk/services/log_metric_service.py`。
- 实现逻辑：`MainWindow` 使用 `QProcess` 启动命令，通过 `bash -lc` source OpenFOAM 环境脚本后执行命令；日志输出进入底部面板；`OpenFoamLogDiagnosticService` 识别缺文件、patch 不匹配、OpenFOAM fatal error、Courant 数过大、发散等问题。
- 当前状态：基础可运行。
- 存在问题：部分退出码和 OpenFOAM 版本差异还未完全覆盖；错误诊断规则仍需随着用户真实 case 增加。

### 模块 9：结果索引、指标导出和残差曲线

- 功能作用：识别当前 case 的时间步、字段、网格和残差结果，并绘制收敛曲线。
- 涉及文件：`src/foamdesk/services/result_index_service.py`、`src/foamdesk/services/log_metric_service.py`、`src/foamdesk/services/metric_export_service.py`、`src/foamdesk/services/residual_plot_service.py`、`src/foamdesk/ui/main_window.py`。
- 实现逻辑：日志解析得到 `SolverMetrics`，导出 `foamdesk_results/residuals.csv` 和 `metrics.json`；`ResidualPlotService` 读取 CSV 并分组绘制。近期修复了 PISO 中同一时间步多次压力修正导致 `p` 曲线变成厚红带的问题，现在会拆成 `p corrector 1`、`p corrector 2`。
- 当前状态：可运行。
- 存在问题：残差解释、收敛判断和论文级图表美化还可继续增强；Matplotlib 中文字体缺失时应优先使用英文图例或配置中文字体。

### 模块 10：3D 结果可视化

- 功能作用：在独立窗口显示 Surface 表面云图、Slice 切片、Vector/Glyph 速度箭头、Streamline 流线箭头、压力点云、3D case 预览等。
- 涉及文件：`src/foamdesk/ui/main_window.py`、`src/foamdesk/services/openfoam_vtk_service.py`。
- 实现逻辑：`OpenFoamVtkService` 读取 OpenFOAM VTK 输出和字段；`VtkViewerDialog` 使用 Matplotlib 3D 绘制多个标签页；支持显示范围过滤、STL 附近过滤、动画播放/暂停、导出 PNG。近期把 Surface 外壳透明度调为 `alpha=0.36`，把速度流线改成稀疏小箭头，长度系数调到 `0.027`。
- 当前状态：基础可运行，用户可查看结果。
- 存在问题：当前不是高质量商业级 3D 渲染；Matplotlib 3D 交互和立体效果有限。之前尝试 Native VTK/PyVista 风格时遇到 WSLg/X11 `BadWindow` 等问题，当前已经回到较稳定的 Matplotlib 3D 独立窗口方向。

### 模块 11：报告导出

- 功能作用：导出 Markdown 报告和相关图片资产。
- 涉及文件：`src/foamdesk/services/report_export_service.py`、`src/foamdesk/ui/main_window.py`。
- 实现逻辑：汇总项目、Case、结果索引、残差数据、可视化能力说明和导出的 PNG 图片，生成 Markdown 报告。
- 当前状态：基础可用。
- 存在问题：报告还偏模板化，缺少自动工程结论、图表编号、单位表、边界条件摘要和论文格式输出。

### 模块 12：主题、字体和设置

- 功能作用：支持深色 VS Code 风格主题、背景色、字体、字号、OpenFOAM 环境脚本、工作区路径、新手教程开关等配置。
- 涉及文件：`src/foamdesk/ui/theme.py`、`src/foamdesk/services/settings_service.py`、`src/foamdesk/ui/main_window.py`。
- 实现逻辑：`AppSettingsService` 读写 JSON 设置，`build_stylesheet` 基于主题调色板生成 Qt 样式。
- 当前状态：基础可用。
- 存在问题：Qt stylesheet 曾出现 `Could not parse stylesheet` 警告，需要继续排查具体选择器或属性；窗口边框颜色受系统原生窗口管理器限制，不能完全由 Qt stylesheet 控制。

## 5. 已编写的代码说明

### 文件：`src/foamdesk/app/main.py`

- 作用：应用程序入口。
- 主要函数：`main()`。
- 关键逻辑：创建 `QApplication`，创建 `ApplicationContext`，启动项目选择窗口或主窗口。
- 依赖关系：依赖 `ApplicationContext`、`StartupWindow`、`MainWindow`。
- 注意事项：GUI 启动依赖 WSLg/Display 环境，不能在无图形环境中正常弹窗。

### 文件：`src/foamdesk/app/bootstrap.py`

- 作用：集中组装应用服务，避免 UI 到处手动创建服务。
- 主要类：`ApplicationContext`。
- 关键逻辑：创建设置服务、项目服务、OpenFOAM 环境探测、参数服务、结果服务、日志服务、报告服务等。
- 依赖关系：被 `main.py` 和 UI 层使用。
- 注意事项：新增服务时优先在这里注入，避免在 `MainWindow` 中直接 new 复杂服务。

### 文件：`src/foamdesk/domain/models.py`

- 作用：定义项目领域数据结构。
- 主要类：`AppSettings`、`SimulationProject`、`SimulationParameters`、`OpenFOAMEnvironmentStatus`、`OpenFoamDiagnostic`、`CaseResultIndex`、`SolverResidual`、`SolverMetrics`、`OpenFoamVtkCaseInfo`。
- 关键逻辑：只放 dataclass，不应放复杂业务逻辑。
- 依赖关系：被 services 和 UI 使用。
- 注意事项：修改字段时要同步测试、JSON 读写和 UI 表单。

### 文件：`src/foamdesk/integrations/openfoam/environment.py`

- 作用：检测 OpenFOAM 环境是否可用。
- 主要类：`OpenFOAMEnvironmentDetector`。
- 关键逻辑：通过 shell 命令和环境脚本检测 OpenFOAM 版本、可执行命令和 bash 路径。
- 依赖关系：被 `ApplicationContext` 和环境检查页面使用。
- 注意事项：不同 OpenFOAM 发行版环境脚本和版本命令可能不同，后续要兼容 Foundation/ESI 版本。

### 文件：`src/foamdesk/services/settings_service.py`

- 作用：管理本地应用设置。
- 主要类：`AppSettingsService`。
- 关键逻辑：读取/保存工作区路径、OpenFOAM 环境脚本、主题、背景色、字体、字号、上次项目路径等。
- 依赖关系：被启动窗口、设置页、项目服务使用。
- 注意事项：JSON 字段变更要保持向后兼容，避免老配置导致启动失败。

### 文件：`src/foamdesk/services/project_service.py`

- 作用：项目、Case、计算域模板、边界条件等核心项目管理。
- 主要类：`ProjectService`、`ComputationDomainTemplate`、`BoundaryConditionSettings`。
- 关键逻辑：创建项目目录，管理 case 路径，生成基础 OpenFOAM 文件，提供计算域模板，保存边界条件。
- 依赖关系：被 `StartupWindow`、`MainWindow`、参数服务等使用。
- 注意事项：不要让项目目录和 Case 目录混淆；用户很在意不同 Case 结果不能串数据。

### 文件：`src/foamdesk/services/geometry_import_service.py`

- 作用：处理 STL 导入、几何资产、snappyHexMesh 配置。
- 主要类：`GeometryAsset`、`StlTransform`、`SnappyHexMeshSettings`、`GeometryImportService`。
- 关键逻辑：读取 STL、应用变换、复制到 `constant/triSurface`，生成 `snappyHexMeshDict` 所需片段。
- 依赖关系：被几何/CAD 菜单、网格生成页面和 3D 预览使用。
- 注意事项：STL 坐标、缩放、平移、旋转直接影响 OpenFOAM 网格生成；要避免导入后覆盖用户原始文件。

### 文件：`src/foamdesk/services/case_parameter_service.py`

- 作用：生成/更新 OpenFOAM case 参数文件。
- 主要类：`OpenFoamCaseParameterService`。
- 关键逻辑：将 UI 表单里的 `SimulationParameters` 写成 `controlDict`、`fvSchemes`、`fvSolution`、`transportProperties`、`turbulenceProperties`、`0/U`、`0/p` 等文件。
- 依赖关系：被仿真参数页、求解器准备页、一键仿真流程使用。
- 注意事项：不同求解器对字段文件要求不同，`icoFoam`、`simpleFoam`、`pisoFoam` 不应共用完全相同模板。

### 文件：`src/foamdesk/services/log_diagnostic_service.py`

- 作用：把 OpenFOAM 原始错误转成中文诊断。
- 主要类：`OpenFoamLogDiagnosticService`。
- 关键逻辑：正则识别缺文件、patch 不匹配、FOAM FATAL、Courant 数过大、残差发散、`pFinal` 等字典缺项。
- 依赖关系：被求解运行流程调用。
- 注意事项：诊断规则要基于真实错误逐步扩展，避免误报。

### 文件：`src/foamdesk/services/log_metric_service.py`

- 作用：从求解日志中提取残差、时间、Courant 数、连续性误差等指标。
- 主要类：`OpenFoamLogMetricService`。
- 关键逻辑：解析 `Solving for Ux`、`Solving for p` 等 OpenFOAM 输出，生成 `SolverMetrics`。
- 依赖关系：和 `MetricExportService` 配合导出 CSV/JSON。
- 注意事项：PISO 算法每个时间步可能有多次 `p` 修正，不能简单当成一条 p 曲线。

### 文件：`src/foamdesk/services/metric_export_service.py`

- 作用：导出求解指标。
- 主要类：`MetricExportService`。
- 关键逻辑：将 `SolverMetrics` 写到 `foamdesk_results/residuals.csv` 和 `metrics.json`。
- 依赖关系：求解完成后由 UI 调用。
- 注意事项：导出的 CSV 是残差曲线和报告的输入，字段名和表头要稳定。

### 文件：`src/foamdesk/services/residual_plot_service.py`

- 作用：读取残差 CSV 并整理为绘图序列。
- 主要类：`ResidualPlotService`。
- 关键逻辑：按字段和同一时间步的重复次数分组；当前将重复压力修正拆成 `p corrector 1`、`p corrector 2`，避免曲线被连接成厚带。
- 依赖关系：被结果页残差图和报告导出使用。
- 注意事项：图例使用英文是为了避免 Matplotlib 当前字体缺少中文字形。

### 文件：`src/foamdesk/services/result_index_service.py`

- 作用：索引 OpenFOAM case 的结果状态。
- 主要类：`ResultIndexService`。
- 关键逻辑：扫描时间步目录、字段文件、`constant/polyMesh`，判断是否有网格和结果。
- 依赖关系：被结果页使用。
- 注意事项：OpenFOAM 时间目录排序要按数值而不是字符串。

### 文件：`src/foamdesk/services/openfoam_vtk_service.py`

- 作用：读取 OpenFOAM/VTK 后处理数据。
- 主要类：`OpenFoamVtkService`。
- 关键逻辑：检查 VTK 标记文件、读取时间步、字段数组、polydata/blocks。
- 依赖关系：被结果页和 3D 视图使用。
- 注意事项：VTK 和 OpenFOAM 输出格式差异大，异常处理必须保守。

### 文件：`src/foamdesk/services/report_export_service.py`

- 作用：生成 Markdown 报告。
- 主要类：`ReportExportService`。
- 关键逻辑：汇总项目信息、结果索引、残差、图片资产，生成 `.md` 报告。
- 依赖关系：被结果页导出报告按钮调用。
- 注意事项：报告内容目前偏基础，后续可以扩展论文/工程报告模板。

### 文件：`src/foamdesk/ui/startup_window.py`

- 作用：项目选择窗口。
- 主要类：`StartupWindow`。
- 关键逻辑：展示项目列表，支持搜索、新建、打开、刷新、删除，并把选中项目传回主程序。
- 依赖关系：依赖 `ProjectService` 和 `AppSettingsService`。
- 注意事项：窗口边框和缩放行为曾多次调整，保留原生窗口管理通常更稳定。

### 文件：`src/foamdesk/ui/main_window.py`

- 作用：主窗口和大多数 UI 编排逻辑。
- 主要类：`MainWindow`、`VtkViewerDialog`、`NativeVtkViewerDialog`、`WindowTitleBar`。
- 关键逻辑：构建菜单栏、Case 树、页面 Tab、底部日志、QProcess 命令执行、几何预览、网格生成、求解器准备、参数配置、结果展示、3D 独立窗口。
- 依赖关系：依赖几乎所有服务，是当前项目最大文件。
- 注意事项：不要一上来全量重写。若要维护，应按页面和功能逐步拆分，例如把结果页、几何页、求解器准备页拆成独立 Widget。

### 文件：`src/foamdesk/ui/theme.py`

- 作用：主题和 Qt stylesheet。
- 主要类/函数：`ThemePalette`、`build_stylesheet()`。
- 关键逻辑：提供 VS Code 风格深色主题和控件样式。
- 依赖关系：被 `MainWindow` 和 `StartupWindow` 使用。
- 注意事项：Qt stylesheet 语法不完全等同 CSS，新增样式后要观察 `Could not parse stylesheet` 警告。

### 文件：`tests/unit/*.py`

- 作用：单元测试。
- 主要测试：项目服务、环境检测、参数生成、日志诊断、日志指标、指标导出、结果索引、VTK 服务、残差绘图服务。
- 关键逻辑：用临时目录构造项目，验证服务层行为。
- 依赖关系：服务层修改后应优先补测试。
- 注意事项：UI 目前缺少系统化自动测试，GUI 变化主要靠人工验证。

## 6. 当前运行方式

### 6.1 依赖安装

推荐在 WSL 中执行：

```bash
cd /home/shihuayue/codex_project
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需开发测试依赖：

```bash
pip install -e '.[dev]'
```

### 6.2 启动程序

优先使用项目脚本：

```bash
cd /home/shihuayue/codex_project
./run.sh
```

等价手动启动：

```bash
cd /home/shihuayue/codex_project
PYTHONPATH=src python3 -m foamdesk.app.main
```

后台启动可用：

```bash
cd /home/shihuayue/codex_project
nohup ./run.sh > foamdesk_run.log 2>&1 &
```

### 6.3 验证命令

```bash
cd /home/shihuayue/codex_project
python3 -m compileall src/foamdesk
PYTHONPATH=src pytest -q
```

如只验证最近残差修复：

```bash
PYTHONPATH=src pytest -q tests/unit/test_residual_plot_service.py
```

### 6.4 环境要求

- 必须在 WSL/Linux 环境中运行。
- WSLg 或其它 X/Wayland 图形服务必须可用。
- `DISPLAY`、`WAYLAND_DISPLAY`、`XDG_RUNTIME_DIR` 异常为空时，Qt 窗口可能无法弹出。
- OpenFOAM 需要已安装并能通过环境脚本 source，例如 `/home/shihuayue/openfoam/OpenFOAM-dev/etc/bashrc`。
- 项目路径固定按 `/home/shihuayue/codex_project` 使用，避免误在 Windows `D:` 或其它目录开发。

### 6.5 已知无法运行场景

- WSL 图形环境未启动或 DISPLAY 为空时，程序可能启动后没有窗口。
- 使用 Native VTK 原生窗口尝试时曾出现 X11 `BadWindow (invalid Window parameter)` 和 `QThreadStorage` 警告，因此当前稳定方向是 Matplotlib 3D 独立窗口。
- Qt 样式表如果有不兼容语法，会输出 `Could not parse stylesheet of object MainWindow(...)`，通常不一定阻断启动，但需要排查样式。

## 7. 当前存在的问题和 Bug

### 问题 1：本地仓库领先远程 1 个提交

- 问题现象：`git status --short --branch` 显示 `main...origin/main [ahead 1]`。
- 可能原因：提交 `6b6a976 docs: update project readme and visualization notes` 本地已完成，但前面 push 多次遇到远端断开。
- 涉及文件：Git 仓库状态，不是单个代码文件。
- 已尝试的方法：执行过 `git push origin main`，出现 `send-pack: unexpected disconnect while reading sideband packet`。
- 建议解决方向：网络稳定后重试 `git push origin main`；推送前确认未跟踪日志和临时目录不会被提交。

### 问题 2：WSL 图形环境可能导致窗口不弹出

- 问题现象：用户点击启动程序但窗口不弹出；检查时曾出现 `DISPLAY=`、`WAYLAND_DISPLAY=`、`XDG_RUNTIME_DIR=` 为空。
- 可能原因：WSLg 没正常接入当前 shell，或者 WSL 图形服务状态异常。
- 涉及文件：`run.sh`、`src/foamdesk/app/main.py`，但根因多半是系统环境。
- 已尝试的方法：查看 `/mnt/wslg`、进程、`foamdesk_run.log`。
- 建议解决方向：重启 WSL：Windows PowerShell 执行 `wsl --shutdown`，重新打开 WSL 后再运行 `./run.sh`；也可检查 `/mnt/wslg/runtime-dir` 和图形环境变量。

### 问题 3：Qt stylesheet 解析警告

- 问题现象：控制台出现 `Could not parse stylesheet of object MainWindow(...)`。
- 可能原因：`theme.py` 或某些内联 `setStyleSheet` 中存在 Qt 不支持的 CSS 属性或选择器。
- 涉及文件：`src/foamdesk/ui/theme.py`、`src/foamdesk/ui/main_window.py`、`src/foamdesk/ui/startup_window.py`。
- 已尝试的方法：多次调整主题和窗口边框，但未系统定位具体 stylesheet 片段。
- 建议解决方向：临时注释分段 stylesheet，逐段启用定位问题；不要混用浏览器 CSS 语法。

### 问题 4：3D 可视化效果仍不够专业

- 问题现象：Surface、Streamline、Vector 等图能看，但不如商业 CFD 软件美观；流线箭头密度、长度、遮挡关系需要反复调。
- 可能原因：当前主要使用 Matplotlib 3D，交互和渲染能力有限。
- 涉及文件：`src/foamdesk/ui/main_window.py` 中 `VtkViewerDialog`。
- 已尝试的方法：曾尝试 Native VTK 窗口，后因 WSLg/X11 稳定性问题回退；当前做过透明外壳、短箭头、稀疏采样等调整。
- 建议解决方向：短期继续优化 Matplotlib 3D；中期可做可选 VTK/PyVista 后端，但不要替换稳定路径。

### 问题 5：完整 CFD 工程流程仍未闭环到商业级

- 问题现象：基础 case 能运行，但复杂 STL、复杂边界条件、多求解器、多物理模型仍不完整。
- 可能原因：OpenFOAM 完整工业仿真涉及大量文件模板、几何清理、网格质量控制和求解器差异。
- 涉及文件：`case_parameter_service.py`、`geometry_import_service.py`、`project_service.py`、`main_window.py`。
- 已尝试的方法：已实现基础模板、求解器入口、物性编辑、patch 表格、最小仿真链路。
- 建议解决方向：下一阶段先稳定一个标准场景，例如“矩形风洞 + 圆柱/小车 STL + simpleFoam 稳态绕流”，不要同时扩展所有物理模型。

### 问题 6：`main_window.py` 过大

- 问题现象：主窗口文件包含 UI 构建、业务编排、绘图、命令执行等大量代码。
- 可能原因：快速迭代阶段为了保持功能连贯，未及时拆分页面 Widget。
- 涉及文件：`src/foamdesk/ui/main_window.py`。
- 已尝试的方法：目前仍保持单文件主控。
- 建议解决方向：在功能稳定后逐步拆分，不要一次性重构；优先抽出 `results_tab.py`、`geometry_tab.py`、`solver_prepare_tab.py`。

### 问题 7：运行时文件和临时目录容易污染 Git

- 问题现象：当前有 `foamdesk_debug.log`、`foamdesk_run.log`、`assets/test_geometries/stl folader/` 未跟踪。
- 可能原因：运行程序和测试导入 STL 时产生。
- 涉及文件：`.gitignore`、工作区。
- 已尝试的方法：前面提交时手动排除。
- 建议解决方向：更新 `.gitignore`，明确忽略 `foamdesk_*.log` 和临时 STL 文件夹。

## 8. 未完成任务 / TODO

### 高优先级

1. 打通并稳定一个完整标准仿真场景：项目 -> Case -> 几何/STL -> 网格 -> 求解器准备 -> 参数 -> 运行 -> 结果图。
2. 解决 WSLg 图形启动可靠性问题，至少在启动失败时给出明确诊断文档。
3. 修复或定位 Qt stylesheet 解析警告。
4. 把求解器选择和实际运行命令严格绑定，保证 `icoFoam`、`simpleFoam`、`pisoFoam` 不混用错误模板。
5. 完善 `simpleFoam` 稳态绕流模板，包括 `0/U`、`0/p`、`0/k`、`0/epsilon`、`0/nut`、`turbulenceProperties`、`fvSolution`、`fvSchemes`。
6. 增强 patch 边界条件表格，支持入口、出口、壁面、物体表面、对称面等逐项配置。
7. 更新 `.gitignore`，避免日志和临时目录误提交。
8. 把本地领先的 1 个 commit 成功 push 到 GitHub，保持本地和远程一致。

### 中优先级

1. 拆分 `main_window.py` 中结果页、几何页、求解器准备页的 UI 代码。
2. 优化 3D 可视化：半透明外壳、切片、等值面、速度矢量、流线箭头、颜色条范围、单位显示、最大/最小值显示。
3. 增加更多计算域模板，例如直管、弯管、风洞、扩张管、收缩管。
4. 完善 STL 导入后的几何检查：是否在计算域内、是否和边界重叠、`locationInMesh` 是否合理。
5. 增加报告导出内容：边界条件表、物性表、网格质量摘要、残差结论、图表说明。
6. 增加 GUI 自动测试或最小界面 smoke test。
7. 增强结果动画播放逻辑，默认从第 0 步播放，并避免重复创建新窗口。

### 低优先级

1. 支持 STEP、IGES、CATIA、SolidWorks 等 CAD 格式，实际应依赖外部 CAD/mesh 库，不建议短期硬做。
2. 支持高质量 Native VTK/PyVista 渲染后端，但必须解决 WSLg 稳定性问题后再启用。
3. 支持多相流、传热、可压缩流、动网格、多物理场耦合。
4. 支持 DOE、参数扫描、多目标优化、ROM 降阶模型。
5. 支持企业级账号权限、团队协作、数据溯源和版本管理。
6. 支持 Linux 打包、Nuitka/PyInstaller、发行版适配和安装器。

## 9. 下一阶段开发建议

### 推荐先做什么

建议下一个 AI 先做“标准稳态绕流场景闭环”，不要继续散点式加 UI。

推荐目标：

```text
一个项目 + 一个 Case
-> 导入 simple_center_cube.stl 或 medium_cylinder_obstacle.stl
-> 选择矩形风洞计算域
-> 生成 blockMeshDict 和 snappyHexMeshDict
-> 执行 blockMesh/snappyHexMesh/checkMesh
-> 自动生成 simpleFoam 所需 0 文件和 constant/system 文件
-> 运行 simpleFoam
-> 结果页显示 p、U、mag(U)、残差和 3D Surface/Slice/Vector
```

### 不建议先做什么

- 不建议先接入复杂 CAD 内核。
- 不建议马上重写整个 `MainWindow`。
- 不建议继续增加大量新按钮但不打通真实 OpenFOAM 文件链路。
- 不建议优先做多相流、传热、燃烧、HPC、GPU、AI 优化等高级功能。
- 不建议再次强行切换到 Native VTK，除非先解决 WSLg `BadWindow` 稳定性。

### 哪些代码不要随便重构

- `ProjectService` 的项目/Case 路径逻辑不要随便改，容易导致 Case 串数据。
- `GeometryImportService` 的 STL 存储和变换逻辑不要随便改，容易影响 `snappyHexMeshDict`。
- `CaseParameterService` 生成 OpenFOAM 文件的逻辑不要大改，应先补测试再调整。
- `ResidualPlotService` 对 PISO 压力修正的分组不要回退，否则残差 p 曲线会再次变成厚带。
- `VtkViewerDialog` 当前是稳定可用路径，不要直接删除替换。

### 哪些模块可以继续扩展

- `case_parameter_service.py`：扩展求解器模板、湍流字段、fvSolution/fvSchemes。
- `project_service.py`：扩展计算域模板和边界条件模型。
- `geometry_import_service.py`：扩展 STL 检查、位置编辑、导入前后预览。
- `openfoam_vtk_service.py`：扩展字段读取、时间步、数组转换。
- `report_export_service.py`：扩展工程报告内容。
- `tests/unit/`：给每个新增服务逻辑补测试。

### 如果要新增功能，应该改哪些文件

- 新增页面或按钮：先看 `src/foamdesk/ui/main_window.py` 对应 `_build_xxx_tab`。
- 新增项目/Case 行为：改 `src/foamdesk/services/project_service.py`。
- 新增 OpenFOAM 参数文件：改 `src/foamdesk/services/case_parameter_service.py`。
- 新增 STL/几何导入功能：改 `src/foamdesk/services/geometry_import_service.py`。
- 新增结果读取：改 `src/foamdesk/services/openfoam_vtk_service.py` 和 `result_index_service.py`。
- 新增残差/指标：改 `log_metric_service.py`、`metric_export_service.py`、`residual_plot_service.py`。
- 新增主题样式：改 `src/foamdesk/ui/theme.py`，并注意 Qt stylesheet 兼容。

### 如果要修 Bug，应该从哪里排查

- 程序不启动：先看 `foamdesk_run.log`、WSLg 环境变量、`run.sh`、`main.py`。
- 按钮没反应：在 `main_window.py` 找按钮文本和 connected handler。
- OpenFOAM 报错：看日志面板原始输出，再看 `log_diagnostic_service.py` 是否已有规则。
- 缺 OpenFOAM 文件：看 `case_parameter_service.py` 和当前 case 目录结构。
- patch 不匹配：看 `constant/polyMesh/boundary`、`0/U`、`0/p`。
- 结果串 Case：看 `_current_project`、`ProjectService`、Case 树选择和 `case_dir`。
- 3D 视图异常：看 `VtkViewerDialog`、`openfoam_vtk_service.py`、VTK 字段数组。
- 残差图异常：看 `foamdesk_results/residuals.csv` 和 `ResidualPlotService`。

## 10. 给接手 AI 的开发规则

接手本项目时，请先理解现有代码，不要一上来重写整个项目。当前项目已经有实际可运行的 PySide6 桌面端、OpenFOAM 命令链路、项目/Case 管理和结果可视化基础，直接推倒重来会丢失大量已验证的细节。

开发时请遵守以下规则：

1. 先读 `README.md`、`AGENTS.md`、`report.md`、`pyproject.toml`，再读代码。
2. 必须在 `/home/shihuayue/codex_project` 中开发，命令通过 WSL 执行。
3. 每次修改前先说明修改目标、涉及模块和边界。
4. 每次修改后说明改了哪些文件、验证了什么、还剩什么问题。
5. 保持现有目录结构和代码风格，不要无关重写。
6. 只改和当前任务相关的文件，不要顺手格式化整个大文件。
7. 新增业务逻辑优先放到 `services/`，UI 只做展示和编排。
8. 如果修改 OpenFOAM 文件生成逻辑，必须补或更新对应单元测试。
9. 如果信息不够，先根据现有代码做合理判断，必要时再问用户。
10. 优先保证项目能运行，再考虑视觉优化和架构重构。
11. 不要提交运行日志、临时 STL 文件夹、`.pytest_cache`、`__pycache__`、workspace 大量结果文件。
12. 每次提交前执行至少 `python3 -m compileall src/foamdesk`，相关服务修改还要跑对应 pytest。
13. Git commit 后需要说明当前 Sprint/阶段、本次 commit hash/message、下一阶段建议。
14. 用户是计算机背景，不默认理解 CFD，解释 OpenFOAM 概念时要用通俗例子。

## 11. 最终交接总结

FoamDesk 目前已经从空项目推进到一个可运行的 Python + PySide6 + OpenFOAM 桌面仿真客户端原型，具备项目/Case 管理、几何绘制与 STL 导入、网格生成入口、求解器准备、仿真参数、OpenFOAM 命令运行、日志诊断、残差曲线和基础 3D 结果可视化。当前最重要的问题不是继续堆新功能，而是稳定一个完整标准仿真闭环，尤其是求解器模板、边界条件、网格生成、结果字段和 UI 按钮之间的真实联动。下一个 AI 接手后，最应该先做的是：确认 WSL 图形环境可启动，推送本地领先的提交，清理 Git 忽略规则，然后围绕一个标准稳态绕流 Case 打通从几何到结果图的完整流程，并为关键服务补测试。