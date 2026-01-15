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

## 3. Python API 参考 (`unity_main.py`)

`UnityClient` 类提供了以下方法与仿真器交互：

### 基础控制
*   **`connect()`**: 连接到 Unity 服务器。
*   **`disconnect()`**: 断开连接。
*   **`reset()`**: 重置环境（卫星位置和仿真时间）。

### 物体控制
*   **`add_object(name, prefab_name, position, rotation)`**: 在场景中实例化新物体。
*   **`set_object_pose(name, position, rotation)`**: 设置物体的位置 (x, y, z) 和旋转 (roll, pitch, yaw)。

### 时间控制（新增）
*   **`set_time_scale(time_scale)`**: 设置时间流速倍数（1=实时，3600=1秒等于1小时）。
*   **`get_time_scale()`**: 获取当前时间流速和仿真时间。

### 轨道控制（新增）
*   **`set_orbit(orbit_altitude_km)`**: 设置卫星轨道高度（千米），返回轨道周期。
    - 示例：400km（LEO），20200km（GPS），35786km（GEO同步）

### 相机与图像
*   **`switch_view(camera_name)`**: 切换 Unity 窗口中的主视角到指定相机。
*   **`get_image(camera_name, width, height)`**: 获取指定相机的图像。返回 numpy 数组。
*   **`save_image(camera_name, width, height, file_path)`**: 拍摄并保存图像到文件（新增）。

## 4. 当前功能特性
*   **HDRP 渲染**: 具备物理单位 (Lux) 的逼真太空光照与正确的阴影表现。
*   **物理真实的轨道动力学**: 
    *   基于开普勒定律的卫星轨道运动
    *   可配置轨道高度、倾角、方向
    *   自动计算轨道速度和周期
*   **同步的时间系统**:
    *   可调节时间流速（支持高达数千倍加速）
    *   地球自转与卫星轨道运动完全同步
    *   同步轨道卫星周期 = 地球自转周期 (24小时)
*   **双相机系统**:
    *   `MainCamera`: 安装在卫星上，用于 RL 观测和对地拍摄。
    *   `ObserverCamera`: 外部上帝视角，用于调试和监控。
*   **图像采集系统**:
    *   实时图像传输（Python端接收numpy数组）
    *   图像保存功能（PNG格式，可指定路径）
    *   支持自定义分辨率
*   **TCP 通信**: 基于 JSON 的稳健通信协议，支持实时控制。
*   **自定义星球着色器**: 两套高质量地球渲染着色器，具备真实的光照效果。

## 5. 星球着色器系统

### 5.1 可用着色器

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

### 5.2 光照同步系统

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

## 6. 轨道动力学与时间同步系统

### 6.1 时间控制器 (`TimeController.cs`)

控制仿真时间流速和地球自转，实现真实的昼夜更替效果。

#### **核心参数**
*   **Time Scale**: 时间加速倍数
    *   1 = 实时
    *   60 = 1分钟等于1小时  
    *   3600 = 1秒等于1小时（推荐用于快速演示）
*   **Earth Rotation Period**: 地球自转周期（默认86400秒，保持不变）
*   **Rotate Earth**: 选择旋转地球还是旋转太阳光源

#### **工作原理**
脚本根据设定的时间流速更新地球旋转或太阳位置：
*   旋转地球模式：地球绕Y轴旋转，太阳光源固定
*   旋转太阳模式：地球固定，太阳光源绕地球旋转

两种模式在视觉效果上等价，可根据场景需求选择。

### 6.2 卫星轨道系统 (`SatelliteOrbit.cs`)

基于开普勒定律实现物理真实的圆轨道运动。

#### **轨道参数**
*   **Orbit Altitude**: 轨道高度（千米，从地表算）
*   **Orbit Inclination**: 轨道倾角（0°=赤道，90°=极轨）
*   **Longitude of Ascending Node**: 升交点赤经（轨道旋转方向）
*   **Initial True Anomaly**: 卫星初始位置角度

#### **物理计算**
*   轨道半径：r = 地球半径 + 轨道高度
*   轨道速度：v = √(GM/r)
*   轨道周期：T = 2π√(r³/GM)

#### **常用轨道示例**
| 轨道类型 | 高度(km) | 周期 | 应用 |
|---------|---------|------|------|
| LEO低轨 | 400 | ~90分钟 | 国际空间站 |
| MEO中轨 | 20200 | ~12小时 | GPS卫星 |
| GEO同步 | 35786 | 24小时 | 通信卫星 |

#### **时间同步机制**
卫星轨道控制器自动引用 `TimeController` 的时间流速：
*   当时间加速3600倍时，卫星在1秒内完成1小时的轨道运动
*   同步轨道卫星绕地球一周时，地球也正好自转一周
*   完美模拟真实的相对运动关系

### 6.3 使用示例

#### Python 快速开始
```python
from unity_main import UnityClient

client = UnityClient()
client.connect()

# 设置3600倍时间加速（1秒=1小时）
client.set_time_scale(3600)

# 设置地球同步轨道（35786km）
response = client.set_orbit(35786)
print(f"轨道周期: {response['period']/3600:.2f} 小时")  # 输出: ~24小时

# 切换到卫星视角观测地球
client.switch_view("MainCamera")

# 等待10秒（仿真10小时），观察地球自转和光照变化
import time
time.sleep(10)

# 拍摄并保存地球照片
client.save_image(
    camera_name="MainCamera",
    width=1920,
    height=1080,
    file_path="./earth_observation.png"
)

client.disconnect()
```

#### 完整演示
运行提供的示例脚本：
```bash
python example_orbit_simulation.py
```

该脚本展示：
1. 时间加速设置
2. 同步轨道配置
3. 多时刻地球拍摄（对比光照变化）
4. 视角切换
5. 系统状态查询

详细配置步骤请参考 `Unity场景配置指南.md`。