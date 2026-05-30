下面分完整可运行代码、分步解读、数据处理、流线绘制、渲染出图，基于 vtk 库，适配你用 foamToVTK -latestTime 导出的 .vtu + .vtp 文件。
前置准备
已执行 foamToVTK -latestTime，得到 VTK/volumes/xxx.vtu、VTK/surfaces/inlet.vtp
安装依赖：pip install vtk numpy
一、整体流程
读取全域体网格 + 速度场（.vtu）
读取入口面种子源（.vtp）
配置流线追踪器，基于速度场积分生成迹线 / 流线
数据映射（按速度大小上色）
渲染窗口显示结果
二、完整代码（稳态流线 / 迹线）
python
import vtk

# ===================== 1. 配置文件路径（改成你自己的路径） =====================
# 体网格+流场文件（volumes 下的 .vtu）
vtu_file = "./VTK/volumes/volume_1000.vtu"
# 入口面种子文件（surfaces 下的 inlet.vtp）
inlet_vtp = "./VTK/surfaces/inlet.vtp"

# ===================== 2. 读取 VTU 体网格与流场数据 =====================
# 2.1 新建 VTU 读取器
vtu_reader = vtk.vtkXMLUnstructuredGridReader()
vtu_reader.SetFileName(vtu_file)
vtu_reader.Update()  # 执行读取

# 拿到完整网格对象（包含网格拓扑 + U、p、VelocityMagnitude 等场量）
grid_data = vtu_reader.GetOutput()

# 可选：打印信息，确认数据是否读取成功
print("网格单元数：", grid_data.GetNumberOfCells())
print("网格顶点数：", grid_data.GetNumberOfPoints())
# 查看当前包含的场变量名
point_arrays = grid_data.GetPointData()
print("顶点场变量列表：")
for i in range(point_arrays.GetNumberOfArrays()):
    print(f"  {point_arrays.GetArrayName(i)}")

# ===================== 3. 读取入口面（流线种子发射源 VTP） =====================
seed_reader = vtk.vtkXMLPolyDataReader()
seed_reader.SetFileName(inlet_vtp)
seed_reader.Update()
seed_geometry = seed_reader.GetOutput()

# ===================== 4. 流线追踪核心：StreamTracer =====================
stream_tracer = vtk.vtkStreamTracer()

# 4.1 接入全域网格数据
stream_tracer.SetInputData(grid_data)

# 4.2 指定速度矢量场 U（流线计算的核心）
# 关联点数据里的 "U" 矢量
stream_tracer.SetInputArrayToProcess(
    0,
    0,
    0,
    vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
    "U"  # 变量名固定为 U，和 OpenFOAM 导出一致
)

# 4.3 设置种子源（从入口面发射流线）
stream_tracer.SetSourceData(seed_geometry)

# 4.4 流线计算参数
stream_tracer.SetMaximumPropagation(800)    # 流线最大延伸长度
stream_tracer.SetIntegrationStepUnit(vtk.vtkStreamTracer.LENGTH_UNIT)
stream_tracer.SetInitialIntegrationStep(0.05)# 积分步长，越小精度越高
stream_tracer.SetIntegrationDirectionToForward()  # 沿流动向前追踪

# 执行流线积分计算
stream_tracer.Update()
stream_lines = stream_tracer.GetOutput()

print("生成流线总数：", stream_lines.GetNumberOfLines())

# ===================== 5. 映射颜色：按速度大小 VelocityMagnitude 上色 =====================
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(stream_lines)

# 使用顶点标量场：VelocityMagnitude（速度幅值）着色
mapper.SetScalarModeToUsePointFieldData()
mapper.SelectColorArray("VelocityMagnitude")
mapper.SetScalarRange(
    stream_lines.GetPointData().GetArray("VelocityMagnitude").GetRange()
)

# ===================== 6. 渲染管线：窗口显示 =====================
# 流线图元
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetLineWidth(2)   # 线条粗细
actor.GetProperty().SetOpacity(1.0)  # 透明度

# 渲染器、窗口、交互器
renderer = vtk.vtkRenderer()
render_win = vtk.vtkRenderWindow()
render_win.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_win)

# 添加图元、背景色
renderer.AddActor(actor)
renderer.SetBackground(0.2, 0.2, 0.3)

# 启动渲染
render_win.Render()
interactor.Initialize()
interactor.Start()
三、逐段解读：读入后做了哪些关键处理
1. 读取 .vtu 之后
vtkXMLUnstructuredGridReader 解析 XML 格式网格文件
得到 vtkUnstructuredGrid 对象，内部整合两类数据：
几何拓扑：点坐标、面、体单元、网格连接关系（对应 OpenFOAM polyMesh）
场数据：顶点上的 U(速度矢量)、p(压力)、VelocityMagnitude(速度标量) 等
代码里打印数量、变量名，是为了校验读取是否正常。
2. 读取 .vtp 之后
读取入口面 inlet.vtp，得到面几何，作为流线种子源
含义：所有流线从这个面上的点开始发射，对应你仿真的流体入口。
3. 流线核心计算（最重要）
vtkStreamTracer 是 VTK 专门做流线 / 迹线的过滤器，内部逻辑：
从种子点出发，读取当前位置 速度矢量 U
按设定步长，沿速度方向积分，计算下一个空间位置
反复迭代，直到达到最大长度 / 流出计算域
把所有点位串联成线条，输出流线几何
稳态工况：只用单时间步 U → 得到流线 (Streamline)
瞬态工况：读取多时间步序列 → 得到迹线 (Pathline)
4. 颜色映射处理
选择 VelocityMagnitude（速度大小）作为着色变量
自动取全场最大 / 最小值作为色标范围，实现「蓝→红」速度渐变效果。
5. 渲染
组装 Mapper → Actor → Renderer → RenderWindow 标准 VTK 渲染管线，最终可视化展示。

结合 OpenFOAM 算例 + VTK（含 Python VTK / VTK 原生），分文件清单、完整操作步骤、代码示例，一步步实现流线 / 迹线绘制。
一、先明确：VTK 读取的核心文件
基于你现有的 OpenFOAM 算例，VTK 本身不直接读 OpenFOAM 原生格式，分两种方案：
方案 1（推荐）：OpenFOAM 先转 VTU/VTP（VTK 标准格式）
1. 源文件（OpenFOAM 原始数据）
网格：constant/polyMesh/ 整套文件（points/faces/owner/neighbour/boundary）
流场数据：目标时间步（如 1000/）下的 U（速度矢量）、可选 p（压力）
2. 转换后 VTK 目标文件（最终给 VTK 读取）
体网格 + 场量：.vtu（非结构化网格，主流）
边界面网格：.vtp
序列文件：.pvd（多时间步，瞬态用）
画流线必备：网格文件 + 速度场 U；上色额外读取标量场（速度幅值、压力）。
二、第一步：OpenFOAM 输出 VTK 格式文件
在算例目录执行，二选一：
方式 A：命令行批量转（稳态 / 瞬态通用）
1. 稳态仿真（只取最后一个时间步）
bash
# 将当前算例结果转为 VTU
foamToVTK -latestTime
执行后会在算例根目录生成 VTK/ 文件夹：
VTK/volumes/：体网格 + U/p 场量（核心）
VTK/surfaces/：各边界面网格（inlet/outlet/wall）

三、第二步：VTK 绘制流线（迹线）两种实操

场景 2：Python + VTK 编程实现（自主开发 / 后处理脚本）
环境准备
bash
pip install vtk numpy
完整代码：读取 VTU + 生成流线
功能：读取 OpenFOAM 转出的 VTU 网格、以入口面为种子、沿速度场生成流线并可视化
python
import vtk

# ===================== 1. 读取 VTU 网格与场数据 =====================
# 替换为你自己的 VTU 文件路径
vtu_path = "./VTK/volumes/volume_1000.vtu"

# 读取器
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtu_path)
reader.Update()
grid = reader.GetOutput()

# ===================== 2. 配置流线追踪器 (VTK StreamTracer) =====================
streamer = vtk.vtkStreamTracer()
streamer.SetInputData(grid)
# 指定速度矢量场名称（OpenFOAM转出后矢量名为 U）
streamer.SetInputArrayToProcess(
    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U"
)

# 流线基础参数
streamer.SetMaximumPropagation(500)    # 最大追踪长度
streamer.SetIntegrationStepUnit(2)
streamer.SetInitialIntegrationStep(0.1)
streamer.SetIntegrationDirectionToForward()

# ===================== 3. 设置种子点：从入口面发射 =====================
# 方式：读取边界面 vtp 作为种子源 (inlet.vtp)
seed_reader = vtk.vtkXMLPolyDataReader()
seed_reader.SetFileName("./VTK/surfaces/inlet.vtp")
seed_reader.Update()
seed_source = seed_reader.GetOutput()

streamer.SetSourceData(seed_source)
streamer.Update()

# ===================== 4. 渲染管线：显示流线 =====================
# 流线几何
stream_mapper = vtk.vtkPolyDataMapper()
stream_mapper.SetInputConnection(streamer.GetOutputPort())
# 按速度幅值着色
stream_mapper.SetScalarModeToUsePointFieldData()
stream_mapper.SelectColorArray("VelocityMagnitude")

stream_actor = vtk.vtkActor()
stream_actor.SetMapper(stream_mapper)
stream_actor.GetProperty().SetLineWidth(2)

# 渲染窗口
renderer = vtk.vtkRenderer()
render_win = vtk.vtkRenderWindow()
render_win.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_win)

renderer.AddActor(stream_actor)
renderer.SetBackground(0.1, 0.1, 0.2)

render_win.Render()
interactor.Start()