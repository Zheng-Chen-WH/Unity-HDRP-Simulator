# Archive 文件夹说明

本文件夹包含已归档的Unity C#脚本，这些脚本在旧架构中使用，现已被Python端实现替代。

## 归档文件列表

### TimeController.cs
**功能**: Unity端时间控制和地球自转
**归档原因**: 时间控制已移至Python端（`simulation_time.py`）
**归档日期**: 2025年12月16日

**原功能**:
- 控制仿真时间流速
- 管理地球自转
- 计算旋转角度

**新实现**: Python端的 `simulation_time.py` 和 `SpaceSimulation` 类

---

### SatelliteOrbit.cs
**功能**: Unity端卫星轨道动力学计算
**归档原因**: 轨道计算已移至Python端（`orbital_dynamics.py`）
**归档日期**: 2025年12月16日

**原功能**:
- 基于开普勒定律的轨道计算
- 卫星位置更新
- 自动朝向地球
- 轨道可视化（Gizmos）

**新实现**: Python端的 `orbital_dynamics.py` 和 `SpaceSimulation` 类

---

## 架构变更说明

### 旧架构（Unity端计算）
```
Unity C#:
- TimeController.cs → 计算时间和地球旋转
- SatelliteOrbit.cs → 计算轨道和位置
- SimulationServer.cs → 接收Python指令

Python:
- unity_main.py → 发送简单控制指令
```

### 新架构（Python端计算）
```
Python:
- simulation_time.py → 时间控制 ✓
- orbital_dynamics.py → 轨道动力学 ✓
- simulation_main.py → 主仿真循环 ✓
- unity_main.py → 发送实时姿态数据

Unity C#:
- SimulationServer.cs → 接收姿态数据并渲染
```

## 为什么要归档？

1. **Python端完全控制**: 所有物理计算在Python中进行，便于调试和扩展
2. **简化Unity场景**: Unity只需专注于高质量渲染
3. **易于集成RL**: 强化学习环境完全在Python端
4. **代码复用性**: Python代码更容易在不同项目间复用

## 如何恢复使用？

如果需要恢复到Unity端计算的架构：

1. 将这两个文件移回 `Assets/Scripts/` 目录
2. 在Unity场景中添加相应组件：
   - 创建TimeController空物体，挂载TimeController.cs
   - 给卫星物体添加SatelliteOrbit.cs组件
3. 使用旧的Python客户端代码（`example_orbit_simulation.py`）

## 保留理由

保留这些文件的原因：
- 作为参考实现
- 用于性能对比测试
- 备用方案（如果Python端出现问题）
- 教学和演示用途

---

**注意**: Unity可能会在下次打开项目时重新生成.meta文件，这是正常现象。
