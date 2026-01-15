"""
仿真时间控制模块
管理仿真时间流速和实际时间的关系
"""

import time
from typing import Optional


class SimulationTime:
    """仿真时间控制器"""
    
    def __init__(self, time_scale: float = 1.0, start_time: float = 0.0):
        """
        初始化时间控制器
        
        Args:
            time_scale: 时间流速倍数（1.0 = 实时，3600.0 = 1秒=1小时）
            start_time: 仿真起始时间（秒）
        """
        self.time_scale = time_scale
        self.simulation_time = start_time  # 仿真时间（秒）
        self.real_time_start = time.time()  # 实际启动时间
        self.last_update_time = self.real_time_start  # 上次更新的实际时间
        self.paused = False
        self.pause_time = 0.0
    
    def update(self) -> float:
        """
        更新仿真时间
        
        Returns:
            delta_time: 仿真时间增量（秒）
        """
        if self.paused:
            return 0.0
        
        current_real_time = time.time()
        real_delta_time = current_real_time - self.last_update_time
        self.last_update_time = current_real_time
        
        # 计算仿真时间增量
        sim_delta_time = real_delta_time * self.time_scale
        self.simulation_time += sim_delta_time
        
        return sim_delta_time
    
    def get_time(self) -> float:
        """获取当前仿真时间（秒）"""
        return self.simulation_time
    
    def get_time_hours(self) -> float:
        """获取当前仿真时间（小时）"""
        return self.simulation_time / 3600.0
    
    def get_time_days(self) -> float:
        """获取当前仿真时间（天）"""
        return self.simulation_time / 86400.0
    
    def set_time_scale(self, scale: float):
        """设置时间流速"""
        self.time_scale = max(0.0, scale)
    
    def get_time_scale(self) -> float:
        """获取当前时间流速"""
        return self.time_scale
    
    def pause(self):
        """暂停仿真"""
        if not self.paused:
            self.paused = True
            self.pause_time = time.time()
    
    def resume(self):
        """恢复仿真"""
        if self.paused:
            self.paused = False
            # 调整last_update_time以跳过暂停期间
            pause_duration = time.time() - self.pause_time
            self.last_update_time += pause_duration
    
    def reset(self, start_time: float = 0.0):
        """重置仿真时间"""
        self.simulation_time = start_time
        self.real_time_start = time.time()
        self.last_update_time = self.real_time_start
        self.paused = False
    
    def get_elapsed_real_time(self) -> float:
        """获取已经过的实际时间（秒）"""
        return time.time() - self.real_time_start
    
    def get_info(self) -> dict:
        """获取时间信息"""
        return {
            'simulation_time_s': self.simulation_time,
            'simulation_time_hours': self.get_time_hours(),
            'simulation_time_days': self.get_time_days(),
            'time_scale': self.time_scale,
            'real_elapsed_time_s': self.get_elapsed_real_time(),
            'paused': self.paused
        }


class FrameRateController:
    """帧率控制器，用于控制仿真循环的更新频率"""
    
    def __init__(self, target_fps: float = 60.0):
        """
        初始化帧率控制器
        
        Args:
            target_fps: 目标帧率（每秒帧数）
        """
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps if target_fps > 0 else 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_sample_interval = 1.0  # 每秒采样一次FPS
        self.fps_sample_time = time.time()
        self.current_fps = 0.0
    
    def wait(self):
        """等待直到达到目标帧时间"""
        if self.target_frame_time <= 0:
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed < self.target_frame_time:
            time.sleep(self.target_frame_time - elapsed)
        
        self.last_frame_time = time.time()
        self.frame_count += 1
        
        # 计算实际FPS
        time_since_sample = self.last_frame_time - self.fps_sample_time
        if time_since_sample >= self.fps_sample_interval:
            self.current_fps = self.frame_count / time_since_sample
            self.frame_count = 0
            self.fps_sample_time = self.last_frame_time
    
    def get_fps(self) -> float:
        """获取当前实际帧率"""
        return self.current_fps
    
    def set_target_fps(self, fps: float):
        """设置目标帧率"""
        self.target_fps = fps
        self.target_frame_time = 1.0 / fps if fps > 0 else 0


if __name__ == "__main__":
    # 测试代码
    print("=== 仿真时间控制测试 ===\n")
    
    # 创建时间控制器（100倍加速）
    sim_time = SimulationTime(time_scale=100.0)
    frame_controller = FrameRateController(target_fps=10.0)  # 10 FPS for testing
    
    print("开始仿真（100倍加速，目标10 FPS）")
    print("运行5秒实际时间...\n")
    
    start_real = time.time()
    while time.time() - start_real < 5.0:
        # 更新仿真时间
        dt = sim_time.update()
        
        # 每秒打印一次
        if int(sim_time.get_time()) % 100 == 0 and dt > 0:
            info = sim_time.get_info()
            print(f"仿真时间: {info['simulation_time_s']:.1f} s "
                  f"({info['simulation_time_hours']:.2f} hours) | "
                  f"实际时间: {info['real_elapsed_time_s']:.1f} s | "
                  f"FPS: {frame_controller.get_fps():.1f}")
        
        # 控制帧率
        frame_controller.wait()
    
    print("\n=== 暂停/恢复测试 ===\n")
    
    sim_time.reset()
    print("仿真已重置")
    
    # 运行2秒
    start_real = time.time()
    while time.time() - start_real < 2.0:
        dt = sim_time.update()
        time.sleep(0.1)
    
    print(f"运行2秒后，仿真时间: {sim_time.get_time():.1f} s")
    
    # 暂停
    sim_time.pause()
    print("仿真已暂停")
    time.sleep(2.0)
    print(f"暂停2秒后，仿真时间: {sim_time.get_time():.1f} s (应该没变)")
    
    # 恢复
    sim_time.resume()
    print("仿真已恢复")
    start_real = time.time()
    while time.time() - start_real < 2.0:
        dt = sim_time.update()
        time.sleep(0.1)
    
    print(f"恢复并运行2秒后，仿真时间: {sim_time.get_time():.1f} s")
    
    print("\n测试完成！")
