# Unity场景配置指南

## 功能概述

已实现的三大核心功能：

1. **地球自转与光照变化** - 按真实时间流速模拟地球旋转，产生昼夜更替效果
2. **卫星轨道运动** - 基于开普勒定律的真实轨道动力学，与地球自转时间同步
3. **相机观测与图片保存** - 从卫星视角拍摄地球，支持实时观测和照片保存

## Unity场景配置步骤

### 第一步：添加TimeController（时间控制器）

1. 在Hierarchy面板中创建一个空物体：
   - 右键 → `Create Empty`
   - 重命名为 `TimeController`

2. 添加脚本组件：
   - 选中 `TimeController` 物体
   - 在Inspector面板点击 `Add Component`
   - 搜索并添加 `TimeController`

3. 配置参数：
   ```
   Time Scale: 3600
     (加速倍数，3600表示1秒=1小时，可根据需要调整)
   
   Earth Rotation Period: 86400
     (地球自转周期，秒，真实值保持86400不变)
   
   Earth Transform: [拖入场景中的Earth物体]
     (地球物体引用)
   
   Rotate Earth: ✓
     (勾选：旋转地球；不勾选：旋转太阳)
   
   Sun Transform: [拖入场景中的Directional Light]
     (仅当Rotate Earth不勾选时使用)
   ```

**工作原理：**
- 勾选"Rotate Earth"：地球绕Y轴旋转，太阳光源固定
- 不勾选"Rotate Earth"：地球固定，太阳光源绕地球旋转
- 两种方式效果相同，选择你喜欢的方式

### 第二步：配置SatelliteOrbit（卫星轨道）

1. 选中场景中的卫星物体（如 `Sat1`）

2. 添加脚本组件：
   - 在Inspector面板点击 `Add Component`
   - 搜索并添加 `SatelliteOrbit`

3. 配置参数：
   ```
   轨道参数：
   ├─ Orbit Altitude Km: 400
   │    (轨道高度，千米，从地表算起)
   │    常用值：400=LEO, 20200=GPS, 35786=GEO同步轨道
   │
   ├─ Orbit Inclination Deg: 0
   │    (轨道倾角，度，0=赤道轨道，90=极轨)
   │
   ├─ Longitude Of Ascending Node: 0
   │    (升交点赤经，轨道的旋转方向)
   │
   └─ Initial True Anomaly: 0
        (卫星初始位置角度)
   
   引用：
   ├─ Earth Transform: [拖入Earth物体]
   ├─ Time Controller: [拖入TimeController物体]
   │    (如果为空，会自动查找)
   └─ Look At Earth: ✓
        (让卫星相机始终指向地球)
   
   物理常数（保持默认）：
   ├─ Earth Radius Km: 6371
   └─ Earth GM: 398600.4418
   
   调试：
   └─ Show Orbit Gizmo: ✓
        (在Scene视图中显示轨道线)
   ```

**轨道参数说明：**
- **LEO低轨** (400km): 周期~90分钟，国际空间站
- **MEO中轨** (20200km): 周期~12小时，GPS卫星
- **GEO同步** (35786km): 周期=24小时，通信卫星

### 第三步：配置相机系统

场景中需要两个相机：

#### 1. MainCamera（卫星相机）
```
挂载位置：Sat1 → MainCamera（作为卫星的子物体）
用途：从卫星视角观测地球
配置：
  - Field of View: 60
  - Clipping Near: 0.01
  - Clipping Far: 100000
```

#### 2. ObserverCamera（外部观察相机）
```
位置：独立物体，放置在能看到整个系统的位置
用途：调试和监控整体运动
建议位置：
  - Position: (0, 5, -10)  // 根据场景缩放调整
  - Rotation: (20, 0, 0)
  - 初始设为disabled，由脚本控制切换
```

### 第四步：更新SimulationServer

1. 找到场景中的 `SimulationServer` 物体（或创建一个）

2. 确认已挂载更新后的 `SimulationServer.cs` 脚本

3. 配置端口：
   ```
   Port: 5000 (保持默认，需与Python客户端一致)
   ```

### 第五步：配置SunSynchronizer（光照同步）

1. 创建同步物体：
   - 右键 → `Create Empty`
   - 重命名为 `SunSync`

2. 添加脚本：
   - 添加 `SunSynchronizer` 组件

3. 配置：
   ```
   Sun Light: [拖入Directional Light]
   Materials: [拖入地球的材质球（可多个）]
     - Earth材质
     - Atmosphere材质（如果有）
   Intensity Scale: 0.1
     (根据HDRP光照强度调整，0.01-1之间)
   ```

## Python使用方法

### 基础使用

```python
from unity_main import UnityClient

client = UnityClient()
client.connect()

# 设置时间流速
client.set_time_scale(3600)  # 3600倍加速

# 设置卫星轨道
response = client.set_orbit(35786)  # 地球同步轨道
print(f"轨道周期: {response['period']/3600:.2f} 小时")

# 切换视角
client.switch_view("MainCamera")  # 卫星视角

# 保存图片
client.save_image(
    camera_name="MainCamera",
    width=1920,
    height=1080,
    file_path="./screenshots/earth.png"
)

client.disconnect()
```

### 运行完整示例

```bash
python example_orbit_simulation.py
```

这个示例会：
1. 设置时间加速
2. 配置同步轨道
3. 从卫星视角拍摄多张照片
4. 展示光照变化效果
5. 切换到外部视角观察整体

## 常见问题排查

### 问题1：地球旋转太快/太慢
- 调整 `TimeController.timeScale`
- 建议值：60-3600之间

### 问题2：卫星不在轨道上运动
- 检查 `SatelliteOrbit` 是否已添加到卫星物体上
- 确认 `Earth Transform` 和 `Time Controller` 引用正确
- 查看Console是否有错误信息

### 问题3：光照没有变化
- 确认 `SunSynchronizer` 已配置
- 检查材质是否使用了Earth.shader或Reconfigurable Shader
- 调整 `Intensity Scale` 参数

### 问题4：卫星轨道周期不对
- 检查场景缩放：`SatelliteOrbit.scaleKmToUnity`
- 确保地球物体的scale符合实际（建议直径=1单位）
- 调整脚本中的缩放因子

### 问题5：Python连接失败
- 确认Unity正在运行（Play模式）
- 检查端口5000未被占用
- 防火墙是否允许连接

### 问题6：拍摄的图片是黑色的
- 确认相机已启用
- 检查相机的Clipping Planes设置
- 确认场景中有光源

## 场景缩放建议

为了保证物理计算准确，建议：

```
地球直径 = 1 Unity单位 = 12742 km
这样：
  - 低轨400km ≈ 0.031 Unity单位
  - 同步轨道35786km ≈ 2.8 Unity单位
```

如果你的场景使用不同缩放，需要修改 `SatelliteOrbit.cs` 中的 `scaleKmToUnity` 计算。

## 进阶功能

### 自定义轨道倾角（极轨卫星）

```python
# 在Unity Inspector中设置：
Orbit Inclination Deg: 90  # 极轨
```

### 多卫星系统

为每个卫星添加 `SatelliteOrbit` 组件，设置不同的：
- `Initial True Anomaly` - 起始位置
- `Longitude Of Ascending Node` - 轨道方向
- `Orbit Altitude` - 高度

### 实时观测模式

在Unity Game窗口中始终显示MainCamera视角：
1. 将MainCamera设为唯一enabled的相机
2. Python端用 `get_image()` 获取图像数据
3. 不影响Game窗口显示

## 参考资料

- 地球同步轨道：35786 km，周期24小时
- 开普勒第三定律：T² = (4π²/GM) × r³
- 地球自转周期：23小时56分4秒（86164秒，恒星日）
- 太阳日：24小时（86400秒）
