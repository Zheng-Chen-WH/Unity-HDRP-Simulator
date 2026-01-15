# 轨道力学参数

# 定义地球参数
Earth={"radius_km":6371.0, # 地球半径（千米）
       "GM_km3_s2":398600.4418, # 地球标准引力参数 (km³/s²)
       "rotation_period_s":86400.0, # 地球自转周期（秒，太阳日）
       }

# 预定义轨道配置
ORBIT_PRESETS = {
    'ISS': {
        'altitude_km': 408,
        'inclination_deg': 51.6,
        'description': '国际空间站轨道'
    },
    'LEO': {
        'altitude_km': 400,
        'inclination_deg': 0,
        'description': '低地球轨道（赤道）'
    },
    'LEO_POLAR': {
        'altitude_km': 600,
        'inclination_deg': 90,
        'description': '低地球轨道（极轨）'
    },
    'GPS': {
        'altitude_km': 20200,
        'inclination_deg': 55,
        'description': 'GPS卫星轨道'
    },
    'GEO': {
        'altitude_km': 35786,
        'inclination_deg': 0,
        'description': '地球同步轨道'
    },
    'MOLNIYA': {
        'altitude_km': 26562,
        'inclination_deg': 63.4,
        'eccentricity': 0.74,
        'description': '闪电轨道（高度为近地点+远地点平均）'
    }
}