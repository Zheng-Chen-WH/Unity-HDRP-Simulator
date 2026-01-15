# -*- coding: utf-8 -*-
"""
椭圆轨道计算测试脚本
测试基本物理规律的正确性
"""

import sys
# Windows下设置UTF-8输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("测试脚本开始")
print("="*70)

import numpy as np
from orbital_dynamics import OrbitalDynamics, create_orbit_from_preset
from orbit_config import Earth

print("\n[模块导入成功]")

# 测试1: 能量守恒
print("\n" + "="*70)
print("[测试1: 能量守恒]")
print("="*70)

orbit = OrbitalDynamics(altitude_km=26554, eccentricity=0.74, inclination_deg=63.4)

energies = []
angular_momenta = []
dt = orbit.orbital_period_s / 50

for i in range(50):
    pos, vel = orbit.get_position_velocity()
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    energy = v**2 / 2 - Earth["GM_km3_s2"] / r
    energies.append(energy)
    h = np.linalg.norm(np.cross(pos, vel))
    angular_momenta.append(h)
    orbit.update(dt)

energy_std = np.std(energies)
energy_rel_var = energy_std / abs(np.mean(energies)) * 100
h_std = np.std(angular_momenta)
h_rel_var = h_std / np.mean(angular_momenta) * 100

print(f"能量相对变化: {energy_rel_var:.9f}%")
print(f"角动量相对变化: {h_rel_var:.9f}%")

if energy_rel_var < 1e-6 and h_rel_var < 1e-6:
    print("[PASS] 能量和角动量守恒良好")

# 测试2: 开普勒第三定律
print("\n" + "="*70)
print("[测试2: 开普勒第三定律]")
print("="*70)

test_altitudes = [400, 1000, 5000, 20000, 35786]

print(f"{'高度(km)':<12} {'半长轴(km)':<14} {'周期(h)':<12} {'T^2/a^3':<20}")
print("-"*65)

ratios = []
for alt in test_altitudes:
    orbit = OrbitalDynamics(altitude_km=alt, eccentricity=0)
    a = orbit.semi_major_axis_km
    T = orbit.orbital_period_s
    ratio = T**2 / a**3
    ratios.append(ratio)
    print(f"{alt:<12.0f} {a:<14.3f} {T/3600:<12.3f} {ratio:<20.9f}")

theory_ratio = 4 * np.pi**2 / Earth["GM_km3_s2"]
mean_ratio = np.mean(ratios)
error = abs(mean_ratio - theory_ratio) / theory_ratio * 100

print(f"\n理论值: {theory_ratio:.9f}")
print(f"平均值: {mean_ratio:.9f}")
print(f"误差: {error:.9f}%")

if error < 1e-6:
    print("[PASS] 完美符合开普勒第三定律")

# 测试3: 维里定理
print("\n" + "="*70)
print("[测试3: 维里定理]")
print("="*70)

test_cases = [
    ("圆轨道", {"altitude_km": 400, "eccentricity": 0.0}),
    ("椭圆轨道", {"altitude_km": 26554, "eccentricity": 0.74}),
]

for name, params in test_cases:
    orbit = OrbitalDynamics(**params)
    pos, vel = orbit.get_position_velocity()
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    
    energy_actual = v**2 / 2 - Earth["GM_km3_s2"] / r
    energy_theory = -Earth["GM_km3_s2"] / (2 * orbit.semi_major_axis_km)
    error = abs(energy_actual - energy_theory) / abs(energy_theory) * 100
    
    print(f"\n{name}:")
    print(f"  实际能量: {energy_actual:.6f} km^2/s^2")
    print(f"  理论能量: {energy_theory:.6f} km^2/s^2")
    print(f"  误差: {error:.9f}%")

# 测试4: 圆轨道特殊性质
print("\n" + "="*70)
print("[测试4: 圆轨道特殊性质]")
print("="*70)

orbit_circle = OrbitalDynamics(altitude_km=500, eccentricity=0.0)

radii = []
speeds = []
dt = orbit_circle.orbital_period_s / 20

for i in range(20):
    pos, vel = orbit_circle.get_position_velocity()
    radii.append(np.linalg.norm(pos))
    speeds.append(np.linalg.norm(vel))
    orbit_circle.update(dt)

r_mean = np.mean(radii)
r_std = np.std(radii)
v_mean = np.mean(speeds)
v_std = np.std(speeds)

print(f"半径: 均值={r_mean:.6f} km, 标准差={r_std:.9f} km")
print(f"速度: 均值={v_mean:.6f} km/s, 标准差={v_std:.9f} km/s")
print(f"半径变化: {r_std/r_mean*100:.12f}%")
print(f"速度变化: {v_std/v_mean*100:.12f}%")

if r_std/r_mean < 1e-10 and v_std/v_mean < 1e-10:
    print("[PASS] 圆轨道半径和速度保持常数")

print("\n" + "="*70)
print("所有测试完成!")
print("="*70)
