# 🚀 Python端物理计算架构 - 快速开始

## 📋 概述

**全新架构**：所有物理计算（轨道动力学、时间控制）都在Python端完成，Unity仅负责高保真渲染。

## 🎯 核心文件

### Python模块（新增）

| 文件 | 功能 | 说明 |
|------|------|------|
| `orbital_dynamics.py` | 轨道动力学 | 开普勒定律、位置/速度计算、预设轨道 |
| `simulation_time.py` | 时间控制 | 时间流速、帧率控制、暂停/恢复 |
| `simulation_main.py` | 仿真主控 | 完整的仿真循环、多卫星管理、实时更新 |
| `test_orbital_accuracy.py` | 测试套件 | 验证轨道计算准确性 |

### 原有文件（扩展）

| 文件 | 修改 |
|------|------|
| `unity_main.py` | 新增 `update_satellite_pose()`, `update_earth_rotation()` 等方法 |

### Unity脚本（简化）

| 文件 | 状态 |
|------|------|
| `SimulationServer.cs` | ✅ 保留（接收姿态命令） |
| `TimeController.cs` | ⚠️ 可选（不再需要，但保留不影响） |
| `SatelliteOrbit.cs` | ⚠️ 可选（不再需要，但保留不影响） |
| `SunSynchronizer.cs` | ✅ 保留（动态光照同步） |

## 🚀 快速开始（3步）

### 1️⃣ 启动Unity

打开Unity项目，点击 **▶ Play** 按钮启动服务器。

### 2️⃣ 运行Python仿真

```bash
python simulation_main.py
```

### 3️⃣ 选择示例

```
Python端物理仿真示例 - 选择一个示例运行
======================================================================
  1. 基础仿真 - 同步轨道
  2. 多卫星系统
  3. 定时拍摄照片
  4. 自定义极轨
  5. 交互式运行
  0. 运行所有示例（除交互式）

请选择 (0-5): 1
```

就这么简单！

## 📚 使用示例

### 示例1：基础仿真

```python
from unity_main import UnityClient
from simulation_main import SpaceSimulation
from orbital_dynamics import create_orbit_from_preset

client = UnityClient()
client.connect()

# 创建仿真（3600倍加速 = 1秒等于1小时）
sim = SpaceSimulation(client, time_scale=3600.0, target_fps=30.0)

# 添加地球同步轨道卫星
sim.add_satellite('Sat1', create_orbit_from_preset('GEO'))

# 运行10秒
sim.run(duration_seconds=10.0)

client.disconnect()
```

### 示例2：自定义轨道

```python
from orbital_dynamics import OrbitalDynamics

# 创建极轨卫星
polar_orbit = OrbitalDynamics(
    altitude_km=600,
    inclination_deg=90,  # 极轨
    initial_true_anomaly_deg=0
)

sim.add_satellite('Sat1', polar_orbit)
sim.run(duration_seconds=20.0)
```

### 示例3：定时拍摄

```python
def capture_every_hour(simulation):
    hours = simulation.sim_time.get_time_hours()
    if hours % 1.0 < 0.01:  # 每仿真1小时
        client.save_image("MainCamera", 1920, 1080, 
                         f"./capture_{int(hours)}h.png")

sim.run(duration_seconds=30.0, on_frame_callback=capture_every_hour)
```

## 🔧 预设轨道类型

```python
from orbital_dynamics import ORBIT_PRESETS

# 可用预设：
'ISS'         # 国际空间站 (408km, 周期92分钟)
'LEO'         # 低地球轨道 (400km, 赤道)
'LEO_POLAR'   # 低轨极轨 (600km, 90°倾角)
'GPS'         # GPS卫星 (20200km, 周期12小时)
'GEO'         # 地球同步轨道 (35786km, 周期24小时)
'MOLNIYA'     # 闪电轨道 (高椭圆轨道)
```

## 🧪 测试轨道计算

```bash
# 测试轨道动力学准确性
python test_orbital_accuracy.py

# 测试时间控制
python simulation_time.py

# 测试轨道计算
python orbital_dynamics.py
```

## 📐 关键参数说明

### 时间加速

```python
time_scale = 1.0      # 实时
time_scale = 60.0     # 1秒 = 1分钟
time_scale = 3600.0   # 1秒 = 1小时（推荐）
time_scale = 86400.0  # 1秒 = 1天（极快）
```

### 帧率设置

```python
target_fps = 60.0   # 高精度（但可能慢）
target_fps = 30.0   # 推荐
target_fps = 10.0   # 快速仿真（低精度）
```

### 场景缩放

```python
# 默认缩放：地球直径 = 1 Unity单位 = 12742 km
sim.scale_km_to_unity = 1.0 / 12742.0

# 如果你的Unity场景不同，需要调整
# 例如：地球直径 = 10 Unity单位
sim.scale_km_to_unity = 10.0 / 12742.0
```

## 🎮 交互式控制

```python
# 运行后可以动态控制
sim = SpaceSimulation(client, time_scale=3600.0)
sim.add_satellite('Sat1', create_orbit_from_preset('GEO'))

# 启动无限循环（按Ctrl+C停止）
sim.run(duration_seconds=None)

# 在另一个终端或回调中：
sim.sim_time.set_time_scale(7200.0)  # 加速到7200倍
sim.sim_time.pause()                  # 暂停
sim.sim_time.resume()                 # 恢复
```

## 📊 获取仿真信息

```python
info = sim.get_simulation_info()

print(f"仿真时间: {info['simulation_time_hours']:.2f} 小时")
print(f"地球旋转: {info['earth_rotation_deg']:.1f}°")

for sat_name, sat_info in info['satellites'].items():
    print(f"\n卫星 {sat_name}:")
    print(f"  位置: {sat_info['position_km']}")
    print(f"  速度: {sat_info['velocity_km_s']}")
    print(f"  轨道周期: {sat_info['orbit_info']['orbital_period_hours']:.2f}h")
```

## 🌟 新架构优势

✅ **完全Python控制** - 所有物理计算在Python端
✅ **易于调试** - Python print/logging，无需Unity Console
✅ **灵活扩展** - 轻松添加摄动、大气阻力等
✅ **RL友好** - 完美集成PyTorch/TensorFlow
✅ **Unity简单** - 场景配置极简，只需SimulationServer
✅ **物理准确** - 基于真实开普勒定律

## 🔄 架构对比

### 旧架构（Unity计算）
```
Python → [发送指令] → Unity
                      ↓
                   计算轨道位置
                      ↓
                   渲染场景
```

### 新架构（Python计算）
```
Python → [计算轨道] → [发送位置] → Unity → 渲染场景
       ↓
    RL/控制/分析
```

## 📖 详细文档

- **完整使用指南**: `Python端物理计算架构指南.md`
- **Unity配置**: `Unity场景配置指南.md`
- **功能总结**: `功能实现总结.md`
- **原README**: `readme-cn.md`

## 🐛 故障排查

### 卫星不动？
```python
# 检查连接
print(client.is_connected)  # 应该是 True

# 检查物体名称（大小写敏感）
# Unity中: "Sat1"
# Python中: sim.add_satellite('Sat1', orbit)
```

### 位置不对？
```python
# 调整缩放因子
sim.scale_km_to_unity = YOUR_SCALE
```

### 帧率太低？
```python
# 降低目标帧率
sim = SpaceSimulation(client, time_scale=3600.0, target_fps=10.0)
```

## 🎓 学习路径

1. **运行基础示例** → `python simulation_main.py` (选1)
2. **查看轨道计算** → `python orbital_dynamics.py`
3. **测试准确性** → `python test_orbital_accuracy.py`
4. **自定义轨道** → 修改 `simulation_main.py` 中的示例
5. **集成你的RL环境** → 使用 `SpaceSimulation` 类作为环境

## 💡 下一步

- 添加轨道摄动（J2、大气阻力）
- 实现轨道机动（霍曼转移）
- 多卫星编队飞行
- 与强化学习框架集成

## 📞 帮助

如有问题，请检查：
1. Unity是否在Play模式
2. 端口5000是否被占用
3. 场景中是否有对应的卫星物体
4. Python依赖是否安装（numpy, pillow）

---

**开始探索吧！** 🚀🌍🛰️
