# Unity 太空仿真器 (HDRP)

本项目是一个基于 Unity HDRP (高清渲染管线) 的高保真太空仿真环境，专为航天器制导的端到端强化学习训练设计。采用 C/S 架构，Unity 负责渲染，Python 负责动力学解算与控制。

## 1. 架构概览

*   **服务端 (Unity)**: 负责高质量渲染 (HDRP)、物理光照模拟（阴影、反光）和场景管理。
*   **客户端 (Python)**: 负责轨道动力学解算、RL 智能体控制，并通过 TCP 向 Unity 发送指令。

## 2. 快速开始

### 前置要求
*   Unity 2022.3 LTS 或更高版本 (需支持 HDRP)
*   Python 3.8+
*   Python 依赖包: `numpy`, `pillow`

### 启动步骤
1.  **Unity 端**:
    *   在 Unity 中打开项目。
    *   打开场景 `Assets/SpaceScene`。
    *   点击顶部的 **播放 (▶)** 按钮启动仿真服务器。
2.  **Python 端**:
    *   运行客户端脚本:
        ```bash
        python unity_client.py
        ```

## 3. Python API 参考 (`unity_client.py`)

`UnityClient` 类提供了以下方法与仿真器交互：

*   **`connect()`**: 连接到 Unity 服务器。
*   **`add_object(name, prefab_name, position, rotation)`**: 在场景中实例化新物体。
*   **`set_object_pose(name, position, rotation)`**: 设置物体的位置 (x, y, z) 和旋转 (roll, pitch, yaw)。
*   **`get_image(camera_name, width, height)`**: 获取指定相机的图像。返回 numpy 数组。
*   **`switch_view(camera_name)`**: 切换 Unity 窗口中的主视角到指定相机。
*   **`reset()`**: 重置环境（例如将卫星移回原点）。
*   **`disconnect()`**: 断开连接。

## 4. Unity 核心概念与快捷键

对于从未接触过 Unity 的用户，以下技巧至关重要：

*   **场景视图导航**:
    *   **按住右键 + WASD**: 在场景中飞行漫游。
    *   **按住右键 + 移动鼠标**: 旋转视角。
    *   **滚轮**: 缩放。
*   **物体操作**:
    *   **Hierarchy 面板 (左侧)**: 场景中所有物体的列表。
    *   **Inspector 面板 (右侧)**: 选中物体的属性设置。
    *   **`F` 键**: 聚焦视角到选中物体（找不到物体时非常有用！）。
    *   **`Ctrl + Shift + F`**: 将选中物体（如相机）对齐到当前场景视图的角度。
*   **重要原则**:
    *   **永远不要**在 **播放模式 (Play Mode)** 下修改场景。运行时所做的修改在停止后会全部丢失。请务必遵循：**停止 (▶)** -> **修改** -> **保存 (Ctrl+S)**。

## 5. 当前功能特性
*   **HDRP 渲染**: 具备物理单位 (Lux) 的逼真太空光照与正确的阴影表现。
*   **双相机系统**:
    *   `MainCamera`: 安装在卫星上，用于 RL 观测。
    *   `ObserverCamera`: 外部上帝视角，用于调试和监控。
*   **TCP 通信**: 基于 JSON 的稳健通信协议，支持实时控制。
*   **自定义星球着色器**: 两套高质量地球渲染着色器，具备真实的光照效果。

## 6. 星球着色器系统

### 6.1 可用着色器

#### **Earth Shader** (`Assets/Shaders/Earth.shader`)
精简高效的着色器，优化性能：
*   **解耦纹理**: 独立的白天纹理、云层、夜晚灯光。
*   **程序化海洋高光**: 基于颜色亮度自动识别海洋区域。
*   **大气散射**: 蓝色边缘光晕效果。
*   **可调参数**:
    *   `Specular Power` (1-128): 控制高光锐度，数值越高光斑越集中。
    *   `Specular Intensity` (0-2): 海洋反射整体亮度。
    *   `Ocean Shininess` (0-1): 海洋反光强度。
    *   `Atmosphere Color/Power/Multiply`: 微调大气边缘效果。

#### **Reconfigurable Shader** (`Assets/Shaders/Reconfigurable Shader.shader`)
高级着色器，提供完整艺术控制：
*   **法线贴图**: 添加表面细节（山脉、地形起伏）。
*   **高光贴图**: 通过 alpha 通道精确标记海洋区域。
*   **细节纹理**: 额外的高频表面细节。
*   **智能海洋检测**: 优先使用高光贴图，无贴图时回退到亮度判断。
*   **包含 Earth Shader 所有特性**，并提供更高视觉保真度。

### 6.2 光照同步系统

两个着色器均使用 `SunSynchronizer.cs` 脚本自动同步 Unity 场景中的平行光源：

#### **配置步骤**
1.  **创建同步物体**:
    *   在 Hierarchy 面板右键 → `Create Empty`。
    *   重命名为 `SunSync`。
    *   点击 `Add Component` → 搜索 `SunSynchronizer`。

2.  **设置引用**:
    *   将场景中的 `Directional Light`（太阳）拖到 `Sun Light` 字段。
    *   将星球的材质球拖到 `Earth Material` 字段。

3.  **调整参数**:
    *   `Intensity Scale` (0.01-1): 缩放 HDRP 的高光照强度以适配着色器。
    *   从 0.1 开始，根据视觉效果调整。

#### **工作原理**
*   脚本每帧读取平行光的**方向**和**颜色**。
*   将这些值传递给着色器属性 `_SunDir` 和 `_SunColor`。
*   标记 `[ExecuteAlways]`: 在播放模式和编辑模式下均实时更新。
*   在场景视图中旋转太阳 → 星球光照实时响应。

#### **常见问题排查**
*   **星球太暗**: 增加 `Intensity Scale`（尝试 0.3-0.5）。
*   **高光过于刺眼**: 增加 `Specular Power`，减少 `Specular Intensity`。
*   **光照无响应**: 检查材质是否使用了 Earth 或 Reconfigurable 着色器。