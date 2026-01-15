# -*- coding: utf-8 -*-
print("测试开始")

try:
    import numpy as np
    print("numpy导入成功")
    
    from orbital_dynamics import OrbitalDynamics
    print("OrbitalDynamics导入成功")
    
    from orbit_config import Earth
    print("orbit_config导入成功")
    
    # 创建轨道
    orbit = OrbitalDynamics(altitude_km=400, eccentricity=0.0)
    print(f"轨道创建成功: a={orbit.semi_major_axis_km:.1f} km")
    
    # 获取位置
    pos, vel = orbit.get_position_velocity()
    print(f"位置: {pos}")
    print(f"速度: {vel}")
    
    print("\n测试成功完成！")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
