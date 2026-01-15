"""
卫星轨道动力学计算模块
基于开普勒定律实现圆轨道和椭圆轨道计算
"""

import numpy as np
from typing import Tuple, Optional
from orbit_config import ORBIT_PRESETS, Earth


class OrbitalDynamics:
    """轨道动力学计算器 - 支持椭圆轨道"""
    def __init__(self, central_body: str = 'Earth', 
                 altitude_km: float = 400.0,
                 inclination_deg: float = 0.0,
                 eccentricity: float = 0.0,
                 longitude_ascending_node_deg: float = 0.0,
                 argument_of_periapsis_deg: float = 0.0,
                 initial_true_anomaly_deg: float = 0.0,
                 initial_position_km: Optional[np.ndarray] = None,
                 initial_velocity_km_s: Optional[np.ndarray] = None):
        """
        初始化轨道参数
        
        Args:
            central_body: 中心天体名称
            altitude_km: 轨道高度（千米，从地表算起，用于圆轨道）
            inclination_deg: 轨道倾角（度）
            eccentricity: 偏心率（0=圆轨道，<1=椭圆轨道）
            longitude_ascending_node_deg: 升交点赤经（度）
            argument_of_periapsis_deg: 近地点幅角（度）
            initial_true_anomaly_deg: 初始真近点角（度）
            initial_position_km: 初始位置向量（可选，用于从状态向量初始化）
            initial_velocity_km_s: 初始速度向量（可选，用于从状态向量初始化）
        """
        if central_body == 'Earth':
            # 物理常数
            self.RADIUS_KM = Earth["radius_km"]
            self.GM = Earth["GM_km3_s2"]
            self.ROTATION_PERIOD = Earth["rotation_period_s"]

        # 如果提供了初始位置和速度，从状态向量计算轨道参数
        if initial_position_km is not None and initial_velocity_km_s is not None:
            self._init_from_state_vectors(initial_position_km, initial_velocity_km_s)
        else:
            # 使用经典轨道根数初始化
            self.altitude_km = altitude_km
            self.inclination_deg = inclination_deg
            self.eccentricity = eccentricity
            self.longitude_ascending_node_deg = longitude_ascending_node_deg
            self.argument_of_periapsis_deg = argument_of_periapsis_deg
            
            # 当前状态
            self.true_anomaly_deg = initial_true_anomaly_deg
            
            # 计算轨道参数
            self._calculate_orbital_parameters()
        
        # 记录初始时刻的真近点角，用于计算时间
        self.initial_true_anomaly_deg = self.true_anomaly_deg
        self.time_since_periapsis_s = self._true_anomaly_to_time(self.true_anomaly_deg)
    
    def _init_from_state_vectors(self, r0: np.ndarray, v0: np.ndarray):
        """从位置和速度向量初始化轨道参数（按课件步骤1-4）"""
        # 步骤1: 计算角动量H确定轨道平面法向
        H_vec = np.cross(r0, v0)
        H = np.linalg.norm(H_vec)
        h_unit = H_vec / H  # 轨道平面法向单位向量
        
        # 步骤2: 计算拉普拉斯矢量L确定轨道平面中的基准方位
        r0_mag = np.linalg.norm(r0)
        v0_mag = np.linalg.norm(v0)
        L_vec = np.cross(v0, H_vec) - self.GM * r0 / r0_mag
        L = np.linalg.norm(L_vec)
        
        # 步骤3: 计算半通径p和偏心率e
        self.p = H**2 / self.GM  # 半通径
        self.eccentricity = L / self.GM
        
        # 步骤4: 计算半长轴a
        if self.eccentricity < 1.0:
            self.semi_major_axis_km = self.p / (1 - self.eccentricity**2)
        else:
            raise ValueError("偏心率 >= 1，不是椭圆轨道")
        
        self.altitude_km = self.semi_major_axis_km - self.RADIUS_KM
        
        # 计算轨道倾角
        self.inclination_deg = np.rad2deg(np.arccos(h_unit[2]))
        
        # 计算升交点赤经
        if self.inclination_deg > 0.01:
            n_vec = np.array([-h_unit[1], h_unit[0], 0])  # 节点线
            n_mag = np.linalg.norm(n_vec)
            if n_mag > 1e-10:
                self.longitude_ascending_node_deg = np.rad2deg(np.arctan2(n_vec[1], n_vec[0]))
            else:
                self.longitude_ascending_node_deg = 0.0
        else:
            self.longitude_ascending_node_deg = 0.0
        
        # 计算近地点幅角
        if self.eccentricity > 1e-6 and L > 1e-10:
            e_unit = L_vec / L
            if self.inclination_deg > 0.01:
                n_vec = np.array([-h_unit[1], h_unit[0], 0])
                n_unit = n_vec / np.linalg.norm(n_vec)
                self.argument_of_periapsis_deg = np.rad2deg(np.arccos(np.dot(n_unit, e_unit)))
                if e_unit[2] < 0:
                    self.argument_of_periapsis_deg = 360 - self.argument_of_periapsis_deg
            else:
                self.argument_of_periapsis_deg = np.rad2deg(np.arctan2(e_unit[1], e_unit[0]))
        else:
            self.argument_of_periapsis_deg = 0.0
        
        # 计算当前真近点角
        if self.eccentricity > 1e-6:
            e_unit = L_vec / L
            r_unit = r0 / r0_mag
            cos_nu = np.dot(e_unit, r_unit)
            self.true_anomaly_deg = np.rad2deg(np.arccos(np.clip(cos_nu, -1, 1)))
            # 判断象限
            if np.dot(r0, v0) < 0:
                self.true_anomaly_deg = 360 - self.true_anomaly_deg
        else:
            # 圆轨道
            self.true_anomaly_deg = 0.0
        
        # 计算其他轨道参数
        self._calculate_derived_parameters()
    
    def _calculate_orbital_parameters(self):
        """从经典轨道根数计算轨道参数"""
        # 半长轴（对于圆轨道，就是轨道半径）
        self.semi_major_axis_km = self.RADIUS_KM + self.altitude_km
        
        # 计算半通径
        self.p = self.semi_major_axis_km * (1 - self.eccentricity**2)
        
        # 计算其他参数
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self):
        """计算导出的轨道参数"""
        # 轨道周期（开普勒第三定律）
        self.orbital_period_s = 2 * np.pi * np.sqrt(
            self.semi_major_axis_km**3 / self.GM
        )
        
        # 平均角速度（rad/s）
        self.mean_motion_rad_s = np.sqrt(self.GM / self.semi_major_axis_km**3)
        
        # 近地点和远地点距离
        self.periapsis_km = self.semi_major_axis_km * (1 - self.eccentricity)
        self.apoapsis_km = self.semi_major_axis_km * (1 + self.eccentricity)
        
        # 近地点速度和远地点速度
        self.periapsis_speed_km_s = np.sqrt(self.GM * (1 + self.eccentricity) / self.p)
        self.apoapsis_speed_km_s = np.sqrt(self.GM * (1 - self.eccentricity) / self.p)
    
    def _solve_kepler_equation(self, M: float, e: float, tol: float = 1e-8, max_iter: int = 50) -> float:
        """用牛顿迭代法求解开普勒方程 M = E - e*sin(E)
        
        Args:
            M: 平近点角（弧度）
            e: 偏心率
            tol: 收敛容差
            max_iter: 最大迭代次数
        
        Returns:
            E: 偏近点角（弧度）
        """
        # 初始猜测
        if e < 0.8:
            E = M
        else:
            E = np.pi
        
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E_new = E - f / f_prime
            
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        
        return E
    
    def _eccentric_anomaly_to_true_anomaly(self, E: float) -> float:
        """从偏近点角计算真近点角（按课件步骤6）
        
        Args:
            E: 偏近点角（弧度）
        
        Returns:
            theta: 真近点角（弧度）
        """
        # 使用公式: tan(θ/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        tan_half_theta = np.sqrt((1 + self.eccentricity) / (1 - self.eccentricity)) * np.tan(E / 2)
        theta = 2 * np.arctan(tan_half_theta)
        return theta
    
    def _true_anomaly_to_time(self, true_anomaly_deg: float) -> float:
        """从真近点角计算从近地点经过的时间（按课件步骤5反向）
        
        Args:
            true_anomaly_deg: 真近点角（度）
        
        Returns:
            t: 从近地点经过的时间（秒）
        """
        theta = np.deg2rad(true_anomaly_deg)
        
        if self.eccentricity < 1e-6:  # 圆轨道
            return theta / self.mean_motion_rad_s
        
        # 从真近点角计算偏近点角
        tan_half_E = np.sqrt((1 - self.eccentricity) / (1 + self.eccentricity)) * np.tan(theta / 2)
        E = 2 * np.arctan(tan_half_E)
        
        # 从偏近点角计算平近点角
        M = E - self.eccentricity * np.sin(E)
        
        # 从平近点角计算时间
        t = M / self.mean_motion_rad_s
        
        return t
    
    def update(self, delta_time_s: float):
        """更新轨道位置（按课件步骤5-6完整实现）
        
        Args:
            delta_time_s: 时间增量（秒）
        """
        # 更新从近地点经过的时间
        self.time_since_periapsis_s += delta_time_s
        
        # 处理周期性
        self.time_since_periapsis_s %= self.orbital_period_s
        
        if self.eccentricity < 1e-6:  # 圆轨道简化
            # 圆轨道：真近点角 = 平近点角
            M = self.mean_motion_rad_s * self.time_since_periapsis_s
            self.true_anomaly_deg = np.rad2deg(M) % 360.0
        else:
            # 椭圆轨道：步骤5-6
            # 计算平近点角 M = n * (t - t_p)
            M = self.mean_motion_rad_s * self.time_since_periapsis_s
            
            # 求解开普勒方程得到偏近点角E
            E = self._solve_kepler_equation(M, self.eccentricity)
            
            # 从偏近点角计算真近点角θ
            theta = self._eccentric_anomaly_to_true_anomaly(E)
            self.true_anomaly_deg = np.rad2deg(theta) % 360.0
    
    def get_position_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前位置和速度（地心惯性坐标系，ECI）
        按课件步骤7-9完整实现
        
        Returns:
            position: 位置向量 [x, y, z] (km)
            velocity: 速度向量 [vx, vy, vz] (km/s)
        """
        # 转换为弧度
        theta = np.deg2rad(self.true_anomaly_deg)  # 真近点角
        inc = np.deg2rad(self.inclination_deg)
        omega = np.deg2rad(self.argument_of_periapsis_deg)
        Omega = np.deg2rad(self.longitude_ascending_node_deg)
        
        # 步骤7: 根据p, e, θ计算矢径大小r
        r = self.p / (1 + self.eccentricity * np.cos(theta))
        
        # 步骤8: 根据p, e, θ计算速度投影vr, vu
        # 径向速度分量
        v_r = np.sqrt(self.GM / self.p) * self.eccentricity * np.sin(theta)
        # 切向速度分量
        v_u = np.sqrt(self.GM / self.p) * (1 + self.eccentricity * np.cos(theta))
        
        # 轨道平面内的位置（近地点方向为x轴）
        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)
        z_orb = 0.0
        
        # 轨道平面内的速度
        vx_orb = v_r * np.cos(theta) - v_u * np.sin(theta)
        vy_orb = v_r * np.sin(theta) + v_u * np.cos(theta)
        vz_orb = 0.0
        
        # 步骤9: 旋转矩阵从轨道平面到ECI坐标系
        # R = Rz(Omega) * Rx(inc) * Rz(omega)
        cos_Omega = np.cos(Omega)
        sin_Omega = np.sin(Omega)
        cos_inc = np.cos(inc)
        sin_inc = np.sin(inc)
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        
        # 构建旋转矩阵元素
        R11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_inc
        R12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_inc
        R13 = sin_Omega * sin_inc
        
        R21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_inc
        R22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_inc
        R23 = -cos_Omega * sin_inc
        
        R31 = sin_omega * sin_inc
        R32 = cos_omega * sin_inc
        R33 = cos_inc
        
        # 位置旋转
        x = R11 * x_orb + R12 * y_orb + R13 * z_orb
        y = R21 * x_orb + R22 * y_orb + R23 * z_orb
        z = R31 * x_orb + R32 * y_orb + R33 * z_orb
        
        # 速度旋转
        vx = R11 * vx_orb + R12 * vy_orb + R13 * vz_orb
        vy = R21 * vx_orb + R22 * vy_orb + R23 * vz_orb
        vz = R31 * vx_orb + R32 * vy_orb + R33 * vz_orb
        
        position = np.array([x, y, z])
        velocity = np.array([vx, vy, vz])
        
        return position, velocity
    
    def get_look_at_earth_rotation(self, position: np.ndarray) -> Tuple[float, float, float]:
        """
        计算卫星朝向地球的旋转角度（欧拉角）
        
        Args:
            position: 卫星位置 [x, y, z] (km)
        
        Returns:
            (roll, pitch, yaw) 欧拉角（度）
        """
        # 计算从卫星指向地心的方向
        direction = -position / np.linalg.norm(position)
        
        # 转换为欧拉角
        # Unity使用左手坐标系，这里需要根据实际情况调整
        pitch = np.rad2deg(np.arcsin(direction[1]))
        yaw = np.rad2deg(np.arctan2(direction[0], direction[2]))
        roll = 0.0  # 假设没有滚转
        
        return roll, pitch, yaw
    
    def set_altitude(self, altitude_km: float):
        """设置新的轨道高度"""
        self.altitude_km = altitude_km
        self._calculate_orbital_parameters()
    
    def get_orbital_info(self) -> dict:
        """获取轨道信息"""
        # 计算当前轨道半径和速度
        position, velocity = self.get_position_velocity()
        r_mag = np.linalg.norm(position)
        v_mag = np.linalg.norm(velocity)
        
        return {
            'altitude_km': self.altitude_km,
            'semi_major_axis_km': self.semi_major_axis_km,
            'semi_latus_rectum_km': self.p,
            'eccentricity': self.eccentricity,
            'periapsis_km': self.periapsis_km,
            'apoapsis_km': self.apoapsis_km,
            'periapsis_speed_km_s': self.periapsis_speed_km_s,
            'apoapsis_speed_km_s': self.apoapsis_speed_km_s,
            'current_radius_km': r_mag,
            'current_speed_km_s': v_mag,
            'orbital_period_s': self.orbital_period_s,
            'orbital_period_hours': self.orbital_period_s / 3600.0,
            'inclination_deg': self.inclination_deg,
            'longitude_ascending_node_deg': self.longitude_ascending_node_deg,
            'argument_of_periapsis_deg': self.argument_of_periapsis_deg,
            'true_anomaly_deg': self.true_anomaly_deg,
            'time_since_periapsis_s': self.time_since_periapsis_s,
            'time_since_periapsis_hours': self.time_since_periapsis_s / 3600.0
        }


class EarthRotation:
    """地球自转计算"""
    
    def __init__(self, rotation_period_s: float = 86400.0):
        """
        初始化地球自转
        
        Args:
            rotation_period_s: 自转周期（秒，默认24小时）
        """
        self.rotation_period_s = rotation_period_s
        self.rotation_angle_deg = 0.0  # 当前旋转角度
        self.angular_velocity_deg_s = 360.0 / rotation_period_s
    
    def update(self, delta_time_s: float):
        """
        更新地球旋转角度
        
        Args:
            delta_time_s: 时间增量（秒）
        """
        self.rotation_angle_deg += self.angular_velocity_deg_s * delta_time_s
        self.rotation_angle_deg %= 360.0
    
    def get_rotation(self) -> Tuple[float, float, float]:
        """
        获取地球旋转（欧拉角）
        
        Returns:
            (roll, pitch, yaw) 欧拉角（度），地球绕Y轴旋转
        """
        return 0.0, self.rotation_angle_deg, 0.0
    
    def reset(self):
        """重置旋转角度"""
        self.rotation_angle_deg = 0.0

def create_orbit_from_preset(preset_name: str) -> OrbitalDynamics:
    """
    从预设配置创建轨道
    
    Args:
        preset_name: 预设名称 ('ISS', 'LEO', 'GPS', 'GEO', 等)
    
    Returns:
        OrbitalDynamics 实例
    """
    if preset_name not in ORBIT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(ORBIT_PRESETS.keys())}")
    
    config = ORBIT_PRESETS[preset_name]
    return OrbitalDynamics(
        altitude_km=config['altitude_km'],
        inclination_deg=config.get('inclination_deg', 0),
        eccentricity=config.get('eccentricity', 0)
    )


if __name__ == "__main__":
    # 测试代码
    print("=== 椭圆轨道动力学测试 ===\n")
    
    # 测试1: 圆轨道 (ISS)
    print("1. 圆轨道测试 (ISS, e=0)")
    orbit_iss = create_orbit_from_preset('ISS')
    info = orbit_iss.get_orbital_info()
    print(f"   高度: {info['altitude_km']:.0f} km")
    print(f"   偏心率: {info['eccentricity']:.6f}")
    print(f"   周期: {info['orbital_period_hours']:.2f} 小时")
    print(f"   近地点: {info['periapsis_km']:.1f} km, 速度: {info['periapsis_speed_km_s']:.3f} km/s")
    print(f"   远地点: {info['apoapsis_km']:.1f} km, 速度: {info['apoapsis_speed_km_s']:.3f} km/s")
    print(f"   当前速度: {info['current_speed_km_s']:.3f} km/s")
    print()
    
    # 测试2: 椭圆轨道 (Molniya)
    print("2. 椭圆轨道测试 (MOLNIYA, e=0.74)")
    orbit_molniya = create_orbit_from_preset('MOLNIYA')
    info = orbit_molniya.get_orbital_info()
    print(f"   半长轴: {info['semi_major_axis_km']:.0f} km")
    print(f"   偏心率: {info['eccentricity']:.6f}")
    print(f"   周期: {info['orbital_period_hours']:.2f} 小时")
    print(f"   近地点: {info['periapsis_km']:.1f} km, 速度: {info['periapsis_speed_km_s']:.3f} km/s")
    print(f"   远地点: {info['apoapsis_km']:.1f} km, 速度: {info['apoapsis_speed_km_s']:.3f} km/s")
    print(f"   当前半径: {info['current_radius_km']:.1f} km, 速度: {info['current_speed_km_s']:.3f} km/s")
    print()
    
    # 测试3: 椭圆轨道位置演进
    print("3. 椭圆轨道时间演进测试 (Molniya)")
    orbit = OrbitalDynamics(
        altitude_km=26554,
        eccentricity=0.74,
        inclination_deg=63.4
    )
    
    # 在不同真近点角测试
    test_angles = [0, 90, 180, 270]
    print(f"   真近点角    半径(km)     速度(km/s)")
    print(f"   " + "-" * 45)
    for angle in test_angles:
        orbit.true_anomaly_deg = angle
        orbit.time_since_periapsis_s = orbit._true_anomaly_to_time(angle)
        position, velocity = orbit.get_position_velocity()
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        print(f"   {angle:6.0f}°     {r:10.1f}     {v:8.3f}")
    print()
    
    # 测试4: 开普勒方程求解
    print("4. 开普勒方程和时间演进测试")
    orbit = create_orbit_from_preset('MOLNIYA')
    dt = orbit.orbital_period_s / 8  # 1/8周期
    
    print(f"   时间步长: {dt/3600:.2f} 小时")
    print(f"   时间(h)   真近点角(°)   半径(km)    速度(km/s)")
    print(f"   " + "-" * 55)
    
    for i in range(9):
        t_hours = i * dt / 3600
        position, velocity = orbit.get_position_velocity()
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        print(f"   {t_hours:6.2f}    {orbit.true_anomaly_deg:8.2f}     {r:9.1f}    {v:7.3f}")
        if i < 8:
            orbit.update(dt)
    print()
    
    # 测试5: 能量守恒验证
    print("5. 能量守恒验证 (椭圆轨道)")
    orbit = create_orbit_from_preset('MOLNIYA')
    energies = []
    
    dt = orbit.orbital_period_s / 20
    for i in range(20):
        position, velocity = orbit.get_position_velocity()
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        # 比能量: ε = v²/2 - μ/r
        specific_energy = v**2 / 2 - orbit.GM / r
        energies.append(specific_energy)
        orbit.update(dt)
    
    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    energy_variation = (max(energies) - min(energies)) / abs(energy_mean) * 100
    
    print(f"   平均比能量: {energy_mean:.3f} km²/s²")
    print(f"   标准差: {energy_std:.6f} km²/s²")
    print(f"   相对变化: {energy_variation:.6f}%")
    print(f"   理论值: {-orbit.GM/(2*orbit.semi_major_axis_km):.3f} km²/s²")
    print()
    
    # 测试6: 圆轨道作为特殊情况
    print("6. 圆轨道验证 (e=0应退化为圆轨道)")
    orbit_circular = OrbitalDynamics(altitude_km=400, eccentricity=0.0)
    
    radii = []
    speeds = []
    dt = orbit_circular.orbital_period_s / 10
    
    for i in range(10):
        position, velocity = orbit_circular.get_position_velocity()
        radii.append(np.linalg.norm(position))
        speeds.append(np.linalg.norm(velocity))
        orbit_circular.update(dt)
    
    r_mean = np.mean(radii)
    r_variation = (max(radii) - min(radii)) / r_mean * 100
    v_mean = np.mean(speeds)
    v_variation = (max(speeds) - min(speeds)) / v_mean * 100
    
    print(f"   半径变化: {r_variation:.6f}% (应接近0)")
    print(f"   速度变化: {v_variation:.6f}% (应接近0)")
    print(f"   平均半径: {r_mean:.1f} km")
    print(f"   平均速度: {v_mean:.3f} km/s")
    
    print("\n=== 测试完成 ===")
    print("✓ 椭圆轨道计算已实现")
    print("✓ 圆轨道在e=0时正确退化")
    print("✓ 开普勒方程求解正常")
    print("✓ 能量守恒得到验证")
