using UnityEngine;

/// <summary>
/// 控制卫星沿开普勒轨道运动
/// 使用简化的圆轨道模型，与TimeController同步
/// </summary>
public class SatelliteOrbit : MonoBehaviour
{
    [Header("轨道参数")]
    [Tooltip("轨道高度（千米，从地表算起）")]
    public float orbitAltitudeKm = 400f; // 低轨道卫星
    
    [Tooltip("轨道倾角（度，相对于赤道平面）")]
    public float orbitInclinationDeg = 0f;
    
    [Tooltip("升交点赤经（度，轨道起始角度）")]
    public float longitudeOfAscendingNode = 0f;
    
    [Tooltip("初始真近点角（度，卫星在轨道上的初始位置）")]
    public float initialTrueAnomaly = 0f;
    
    [Header("引用")]
    [Tooltip("地球物体")]
    public Transform earthTransform;
    
    [Tooltip("时间控制器（用于同步时间流速）")]
    public TimeController timeController;
    
    [Tooltip("是否让卫星相机始终指向地球")]
    public bool lookAtEarth = true;
    
    [Header("物理常数")]
    [Tooltip("地球半径（千米）")]
    public float earthRadiusKm = 6371f;
    
    [Tooltip("地球标准引力参数 GM (km³/s²)")]
    public float earthGM = 398600.4418f;
    
    [Header("调试")]
    public bool showOrbitGizmo = true;
    public Color gizmoColor = Color.cyan;
    
    // 内部变量
    private float orbitRadiusKm; // 轨道半径（从地心算）
    private float orbitalSpeed; // 轨道速度 (km/s)
    private float orbitalPeriod; // 轨道周期 (s)
    private float currentTrueAnomaly; // 当前真近点角
    
    // Unity场景缩放因子（Unity单位 vs 千米）
    // 假设场景中地球的scale是这样的：地球直径 = 1 Unity单位 = 12742 km
    private float scaleKmToUnity = 1f / 12742f; // 需要根据实际场景调整
    
    void Start()
    {
        // 如果没有指定地球，尝试自动查找
        if (earthTransform == null)
        {
            GameObject earth = GameObject.Find("Earth");
            if (earth != null) earthTransform = earth.transform;
        }
        
        // 如果没有指定时间控制器，尝试自动查找
        if (timeController == null)
        {
            timeController = FindObjectOfType<TimeController>();
        }
        
        // 根据场景中地球的实际大小调整缩放因子
        if (earthTransform != null)
        {
            // 假设地球的scale.x就是直径（Unity单位）
            float earthDiameterUnity = earthTransform.localScale.x * 2f;
            scaleKmToUnity = earthDiameterUnity / 12742f;
        }
        
        // 计算轨道参数
        CalculateOrbitParameters();
        
        // 设置初始位置
        currentTrueAnomaly = initialTrueAnomaly;
        UpdatePosition();
    }
    
    void Update()
    {
        // 获取时间增量（考虑时间加速）
        float deltaTime = Time.deltaTime;
        if (timeController != null)
        {
            deltaTime *= timeController.timeScale;
        }
        
        // 计算角速度（度/秒）
        float angularVelocity = 360f / orbitalPeriod;
        
        // 更新真近点角
        currentTrueAnomaly += angularVelocity * deltaTime;
        currentTrueAnomaly %= 360f;
        
        // 更新位置
        UpdatePosition();
    }
    
    void CalculateOrbitParameters()
    {
        // 轨道半径 = 地球半径 + 轨道高度
        orbitRadiusKm = earthRadiusKm + orbitAltitudeKm;
        
        // 轨道速度：v = sqrt(GM/r)
        orbitalSpeed = Mathf.Sqrt(earthGM / orbitRadiusKm);
        
        // 轨道周期：T = 2π * sqrt(r³/GM)
        orbitalPeriod = 2f * Mathf.PI * Mathf.Sqrt(Mathf.Pow(orbitRadiusKm, 3f) / earthGM);
        
        Debug.Log($"[SatelliteOrbit] 轨道高度: {orbitAltitudeKm} km");
        Debug.Log($"[SatelliteOrbit] 轨道半径: {orbitRadiusKm} km");
        Debug.Log($"[SatelliteOrbit] 轨道速度: {orbitalSpeed:F2} km/s");
        Debug.Log($"[SatelliteOrbit] 轨道周期: {orbitalPeriod:F0} s ({orbitalPeriod/3600f:F2} hours)");
        
        // 检查是否接近同步轨道
        if (Mathf.Abs(orbitalPeriod - 86400f) < 1000f)
        {
            Debug.Log("[SatelliteOrbit] 这是一个接近地球同步轨道的设置！");
        }
    }
    
    void UpdatePosition()
    {
        if (earthTransform == null) return;
        
        // 将角度转换为弧度
        float anomalyRad = currentTrueAnomaly * Mathf.Deg2Rad;
        float inclinationRad = orbitInclinationDeg * Mathf.Deg2Rad;
        float loanRad = longitudeOfAscendingNode * Mathf.Deg2Rad;
        
        // 在轨道平面内的位置（简化的圆轨道）
        float x = orbitRadiusKm * Mathf.Cos(anomalyRad);
        float z = orbitRadiusKm * Mathf.Sin(anomalyRad);
        float y = 0f;
        
        // 应用轨道倾角（绕X轴旋转）
        float y_inclined = y * Mathf.Cos(inclinationRad) - z * Mathf.Sin(inclinationRad);
        float z_inclined = y * Mathf.Sin(inclinationRad) + z * Mathf.Cos(inclinationRad);
        
        // 应用升交点赤经（绕Y轴旋转）
        float x_final = x * Mathf.Cos(loanRad) - z_inclined * Mathf.Sin(loanRad);
        float z_final = x * Mathf.Sin(loanRad) + z_inclined * Mathf.Cos(loanRad);
        
        // 转换为Unity坐标系统
        Vector3 positionKm = new Vector3(x_final, y_inclined, z_final);
        Vector3 positionUnity = positionKm * scaleKmToUnity;
        
        // 相对于地球中心设置位置
        transform.position = earthTransform.position + positionUnity;
        
        // 让卫星相机朝向地球
        if (lookAtEarth)
        {
            transform.LookAt(earthTransform.position);
        }
    }
    
    /// <summary>
    /// 设置轨道高度并重新计算参数
    /// </summary>
    public void SetOrbitAltitude(float altitudeKm)
    {
        orbitAltitudeKm = altitudeKm;
        CalculateOrbitParameters();
    }
    
    /// <summary>
    /// 获取当前轨道周期（秒）
    /// </summary>
    public float GetOrbitalPeriod()
    {
        return orbitalPeriod;
    }
    
    /// <summary>
    /// 获取当前轨道速度（km/s）
    /// </summary>
    public float GetOrbitalSpeed()
    {
        return orbitalSpeed;
    }
    
    void OnDrawGizmos()
    {
        if (!showOrbitGizmo || earthTransform == null) return;
        
        Gizmos.color = gizmoColor;
        
        // 绘制轨道圆环
        int segments = 64;
        float angleStep = 360f / segments;
        
        Vector3 earthPos = earthTransform.position;
        float orbitRadiusUnity = (earthRadiusKm + orbitAltitudeKm) * scaleKmToUnity;
        
        for (int i = 0; i < segments; i++)
        {
            float angle1 = i * angleStep * Mathf.Deg2Rad;
            float angle2 = (i + 1) * angleStep * Mathf.Deg2Rad;
            
            // 应用轨道倾角和升交点
            Vector3 point1 = GetOrbitPoint(angle1, orbitRadiusUnity);
            Vector3 point2 = GetOrbitPoint(angle2, orbitRadiusUnity);
            
            Gizmos.DrawLine(earthPos + point1, earthPos + point2);
        }
        
        // 绘制卫星到地心的连线
        Gizmos.color = Color.yellow;
        Gizmos.DrawLine(transform.position, earthPos);
    }
    
    Vector3 GetOrbitPoint(float angle, float radius)
    {
        float x = radius * Mathf.Cos(angle);
        float z = radius * Mathf.Sin(angle);
        float y = 0f;
        
        // 应用轨道倾角
        float inclinationRad = orbitInclinationDeg * Mathf.Deg2Rad;
        float y_inclined = y * Mathf.Cos(inclinationRad) - z * Mathf.Sin(inclinationRad);
        float z_inclined = y * Mathf.Sin(inclinationRad) + z * Mathf.Cos(inclinationRad);
        
        // 应用升交点赤经
        float loanRad = longitudeOfAscendingNode * Mathf.Deg2Rad;
        float x_final = x * Mathf.Cos(loanRad) - z_inclined * Mathf.Sin(loanRad);
        float z_final = x * Mathf.Sin(loanRad) + z_inclined * Mathf.Cos(loanRad);
        
        return new Vector3(x_final, y_inclined, z_final);
    }
}
