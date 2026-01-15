"""
Python端完整仿真示例
所有物理计算在Python中完成，Unity只负责渲染
"""
import numpy as np
import time
from datetime import datetime
import os
from unity_main import UnityClient
from OrbitalDynamics.orbital_dynamics import OrbitalDynamics, EarthRotation, create_orbit_from_preset, ORBIT_PRESETS
from OrbitalDynamics.simulation_time import SimulationTime, FrameRateController

class Env:
    """太空仿真主控制器"""
    
    def __init__(self, unity_client: UnityClient, 
                 time_scale: float = 3600.0,
                 target_fps: float = 30.0):
        """
        初始化仿真
        
        Args:
            unity_client: Unity客户端连接
            time_scale: 时间流速倍数
            target_fps: 目标帧率
        """
        self.client = unity_client
        self.sim_time = SimulationTime(time_scale=time_scale)
        self.frame_controller = FrameRateController(target_fps=target_fps)
        
        # 地球自转
        self.earth_rotation = EarthRotation()
        
        # 卫星列表（可以有多个）
        self.satellites = {}
        
        # Unity场景缩放因子（根据实际场景调整）
        # 假设Unity中地球直径 = 1单位 = 1 km
        self.scale_km_to_unity = 1.0 / 1.0
        
        self.running = False
    
    def add_satellite(self, name: str, orbit: OrbitalDynamics):
        """
        添加卫星
        
        Args:
            name: 卫星在Unity中的名称
            orbit: 轨道动力学对象
        """
        self.satellites[name] = orbit
        print(f"[Sim] 添加卫星 '{name}'")
        info = orbit.get_orbital_info()
        print(f"      轨道高度: {info['altitude_km']:.0f} km")
        print(f"      轨道周期: {info['orbital_period_hours']:.2f} 小时")
    
    def remove_satellite(self, name: str):
        """移除卫星"""
        if name in self.satellites:
            del self.satellites[name]
            print(f"[Sim] 移除卫星 '{name}'")
    
    def update(self):
        """单帧更新"""
        # 更新仿真时间
        dt = self.sim_time.update()
        
        if dt <= 0:
            return
        
        # 更新地球自转
        self.earth_rotation.update(dt)
        roll, pitch, yaw = self.earth_rotation.get_rotation()
        self.client.update_earth_rotation("Earth", yaw)
        
        # 更新所有卫星
        for sat_name, orbit in self.satellites.items():
            # 更新轨道位置
            orbit.update(dt)
            position_km, velocity_km_s = orbit.get_position_velocity()
            
            # 转换为Unity坐标系（km -> Unity单位）
            position_unity = position_km * self.scale_km_to_unity
            
            # 计算朝向地球的旋转
            roll, pitch, yaw = orbit.get_look_at_earth_rotation(position_km)
            
            # 发送到Unity
            self.client.update_satellite_pose(
                sat_name,
                position_unity.tolist(),
                [roll, pitch, yaw]
            )
    
    def run(self, duration_seconds: float = None, 
            on_frame_callback=None,
            print_interval: float = 1.0):
        """
        运行仿真循环
        
        Args:
            duration_seconds: 运行时长（实际秒数），None表示无限运行
            on_frame_callback: 每帧回调函数，接收当前仿真对象
            print_interval: 打印状态的间隔（实际秒数）
        """
        self.running = True
        start_real_time = time.time()
        last_print_time = start_real_time
        frame_count = 0
        
        print("\n" + "="*60)
        print("仿真开始运行")
        print("="*60)
        
        try:
            while self.running:
                # 检查是否达到运行时长
                if duration_seconds and (time.time() - start_real_time) >= duration_seconds:
                    break
                
                # 更新一帧
                self.update()
                frame_count += 1
                
                # 调用回调
                if on_frame_callback:
                    on_frame_callback(self)
                
                # 打印状态
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    self._print_status(frame_count, current_time - start_real_time)
                    last_print_time = current_time
                    frame_count = 0
                
                # 控制帧率
                self.frame_controller.wait()
        
        except KeyboardInterrupt:
            print("\n\n[Sim] 收到中断信号，正在停止...")
        
        finally:
            self.running = False
            print("\n" + "="*60)
            print("仿真已停止")
            self._print_final_stats(time.time() - start_real_time)
            print("="*60)
    
    def stop(self):
        """停止仿真"""
        self.running = False
    
    def _print_status(self, frames, elapsed_real):
        """打印状态信息"""
        info = self.sim_time.get_info()
        fps = self.frame_controller.get_fps()
        
        print(f"\r[Sim] 仿真: {info['simulation_time_hours']:7.2f} h "
              f"({info['simulation_time_days']:.3f} days) | "
              f"实际: {elapsed_real:6.1f} s | "
              f"FPS: {fps:5.1f} | "
              f"地球: {self.earth_rotation.rotation_angle_deg:6.1f}°", 
              end='', flush=True)
    
    def _print_final_stats(self, total_real_time):
        """打印最终统计"""
        info = self.sim_time.get_info()
        print(f"仿真总时间: {info['simulation_time_hours']:.2f} 小时 "
              f"({info['simulation_time_days']:.3f} 天)")
        print(f"实际运行时间: {total_real_time:.2f} 秒")
        print(f"平均FPS: {self.frame_controller.get_fps():.1f}")
        print(f"时间加速比: {self.sim_time.time_scale:.0f}x")
    
    def get_simulation_info(self) -> dict:
        """获取仿真信息"""
        info = self.sim_time.get_info()
        info['earth_rotation_deg'] = self.earth_rotation.rotation_angle_deg
        info['satellites'] = {}
        
        for name, orbit in self.satellites.items():
            orbit_info = orbit.get_orbital_info()
            position, velocity = orbit.get_position_velocity()
            info['satellites'][name] = {
                'position_km': position.tolist(),
                'velocity_km_s': velocity.tolist(),
                'orbit_info': orbit_info
            }
        
        return info

client = UnityClient()
client.connect()

if not client.is_connected:
    print("无法连接到Unity！")

sim = Env(client, time_scale=3600.0, target_fps=60.0)

# 创建截图目录
screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots_auto")
os.makedirs(screenshot_dir, exist_ok=True)

# 创建自定义轨道
orbit = OrbitalDynamics(
    altitude_km=600,           # 600km高度
    inclination_deg=90,         # 极轨（90度倾角）
    longitude_ascending_node_deg=45,  # 轨道旋转45度
    initial_true_anomaly_deg=0
)

# 添加预设轨道卫星
# sim.add_satellite('Sat1', create_orbit_from_preset('GEO'))

# 添加自定义轨道卫星
sim.add_satellite('Sat1', orbit)

# 切换到主摄像机视角
client.switch_view("MainCamera")

# 从外部视角观察
client.switch_view("ObserverCamera")

# 定时拍摄的回调函数
last_capture_time = [0.0]  # 使用列表以便在闭包中修改
capture_interval_hours = 2.0  # 每仿真2小时拍一张

def capture_callback(simulation: Env):
    sim_hours = simulation.sim_time.get_time_hours()
    if sim_hours - last_capture_time[0] >= capture_interval_hours:
        timestamp = datetime.now().strftime('%H%M%S')
        file_path = os.path.join(screenshot_dir, 
                                f"earth_sim{sim_hours:.1f}h_{timestamp}.png")
        client.save_image("MainCamera", 1920, 1080, file_path)
        print(f"\n[Capture] 已保存照片: {file_path}")
        last_capture_time[0] = sim_hours

sim.run(duration_seconds=30.0, 
               on_frame_callback=capture_callback,
               print_interval=2.0)

# 打印最终信息
info = sim.get_simulation_info()
print(f"\n最终状态:")
print(f"  地球旋转: {info['earth_rotation_deg']:.1f}°")
print(f"  卫星位置: {info['satellites']['Sat1']['position_km']}")

client.disconnect()

