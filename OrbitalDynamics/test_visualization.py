"""
椭圆轨道可视化验证脚本
通过图形化方式验证轨道计算的正确性
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from orbital_dynamics import OrbitalDynamics
from orbit_config import Earth
import os

# 不使用中文字体，避免字体加载问题

print("="*70)
print("椭圆轨道可视化测试")
print("="*70)

# 创建一个Molniya轨道 (高偏心率椭圆轨道)
print("\n创建Molniya轨道:")
print("  近地点高度: 500 km")
print("  远地点高度: 39,354 km")
print("  偏心率: 0.74")
print("  倾角: 63.4°")

orbit = OrbitalDynamics(altitude_km=500, eccentricity=0.74, inclination_deg=63.4)

print(f"\n轨道参数:")
print(f"  半长轴: {orbit.semi_major_axis_km:.2f} km")
print(f"  轨道周期: {orbit.orbital_period_s/3600:.2f} 小时")

# 模拟一个完整周期
n_steps = 500
dt = orbit.orbital_period_s / n_steps

positions = []
velocities = []
times = []
energies = []
angular_momenta = []

print(f"\n正在模拟轨道运动 ({n_steps} 步)...")

for i in range(n_steps):
    pos, vel = orbit.get_position_velocity()
    positions.append(pos.copy())
    velocities.append(vel.copy())
    times.append(orbit.time_since_periapsis_s)
    
    # 计算能量
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    energy = v**2 / 2 - Earth["GM_km3_s2"] / r
    energies.append(energy)
    
    # 计算角动量
    h = np.linalg.norm(np.cross(pos, vel))
    angular_momenta.append(h)
    
    orbit.update(dt)

positions = np.array(positions)
velocities = np.array(velocities)
times = np.array(times) / 3600  # 转换为小时
energies = np.array(energies)
angular_momenta = np.array(angular_momenta)

print("模拟完成!")

# 计算统计信息
print(f"\n统计信息:")
print(f"  能量平均值: {np.mean(energies):.6f} km²/s²")
print(f"  能量标准差: {np.std(energies):.12f} km²/s²")
print(f"  能量相对变化: {np.std(energies)/abs(np.mean(energies))*100:.12f}%")
print(f"  角动量平均值: {np.mean(angular_momenta):.6f} km²/s")
print(f"  角动量标准差: {np.std(angular_momenta):.12f} km²/s")
print(f"  角动量相对变化: {np.std(angular_momenta)/np.mean(angular_momenta)*100:.12f}%")

# 创建图形
print("\n正在生成图像...")
try:
    fig = plt.figure(figsize=(16, 12))
except Exception as e:
    print(f"创建图形失败: {e}")
    raise

# 1. 3D轨道图
print("  创建3D轨道图...")
try:
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1.5, label='Orbit')
    ax1.scatter([0], [0], [0], color='green', s=200, marker='o', label='Earth')
    ax1.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                color='red', s=100, marker='o', label='Start')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('Elliptical Orbit 3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    print("  3D轨道图完成")
except Exception as e:
    print(f"  3D轨道图失败: {e}")
    import traceback
    traceback.print_exc()

# 2. XY平面投影
print("  创建XY平面投影...")
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5)
ax2.scatter([0], [0], color='green', s=200, marker='o', label='Earth')
ax2.scatter([positions[0, 0]], [positions[0, 1]], 
            color='red', s=100, marker='o', label='Start')
theta = np.linspace(0, 2*np.pi, 100)
earth_r = Earth["radius_km"]
ax2.plot(earth_r*np.cos(theta), earth_r*np.sin(theta), 'g--', alpha=0.3, label='Earth Surface')
ax2.set_xlabel('X (km)')
ax2.set_ylabel('Y (km)')
ax2.set_title('Orbit XY Plane Projection')
ax2.axis('equal')
ax2.legend()
ax2.grid(True)
print("  XY平面投影完成")

# 3. 距离变化
print("  创建距离变化图...")
ax3 = fig.add_subplot(2, 3, 3)
radii = np.linalg.norm(positions, axis=1)
ax3.plot(times, radii, 'b-', linewidth=1.5)
ax3.axhline(y=Earth["radius_km"] + 500, color='r', linestyle='--', 
            alpha=0.7, label='Periapsis (500 km)')
ax3.axhline(y=Earth["radius_km"] + 39354, color='g', linestyle='--', 
            alpha=0.7, label='Apoapsis (39,354 km)')
ax3.set_xlabel('Time (hours)')
ax3.set_ylabel('Distance from Earth Center (km)')
ax3.set_title('Satellite Distance vs Time')
ax3.legend()
ax3.grid(True)
print("  距离变化图完成")

# 4. 速度变化
print("  创建速度变化图...")
ax4 = fig.add_subplot(2, 3, 4)
speeds = np.linalg.norm(velocities, axis=1)
ax4.plot(times, speeds, 'r-', linewidth=1.5)
ax4.set_xlabel('Time (hours)')
ax4.set_ylabel('Speed (km/s)')
ax4.set_title('Satellite Speed vs Time')
ax4.grid(True)
print("  速度变化图完成")

# 5. 能量变化
print("  创建能量变化图...")
ax5 = fig.add_subplot(2, 3, 5)
energy_mean = np.mean(energies)
energy_deviation = (energies - energy_mean) / abs(energy_mean) * 100
ax5.plot(times, energy_deviation, 'g-', linewidth=1.5)
ax5.set_xlabel('Time (hours)')
ax5.set_ylabel('Energy Deviation (%)')
ax5.set_title('Energy Conservation Test')
ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax5.grid(True)
print("  能量变化图完成")

# 6. 角动量变化
print("  创建角动量变化图...")
ax6 = fig.add_subplot(2, 3, 6)
h_mean = np.mean(angular_momenta)
h_deviation = (angular_momenta - h_mean) / h_mean * 100
ax6.plot(times, h_deviation, 'm-', linewidth=1.5)
ax6.set_xlabel('Time (hours)')
ax6.set_ylabel('Angular Momentum Deviation (%)')
ax6.set_title('Angular Momentum Conservation Test')
ax6.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax6.grid(True)
print("  角动量变化图完成")

print("  完成所有子图绘制")

print("\n正在调整布局...")
plt.tight_layout()
print("布局调整完成")

# 保存图像
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'orbital_validation.png')
print(f"\n正在保存图像到: {save_path}")

try:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图像已成功保存!")
    
    # 验证文件是否存在
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"文件大小: {file_size/1024:.2f} KB")
    else:
        print("警告: 文件未找到!")
except Exception as e:
    print(f"保存图像失败: {e}")
    import traceback
    traceback.print_exc()

print("\n尝试显示图像...")
try:
    plt.show()
except Exception as e:
    print(f"显示图像失败（这是正常的，因为使用了Agg后端）: {e}")

print("\n" + "="*70)
print("可视化完成!")
print("="*70)
