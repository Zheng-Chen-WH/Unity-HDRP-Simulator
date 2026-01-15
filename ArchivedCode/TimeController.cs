using UnityEngine;

/// <summary>
/// 控制仿真时间流速和地球自转
/// Earth rotates 360° per 24 hours (86400 seconds)
/// </summary>
public class TimeController : MonoBehaviour
{
    [Header("时间设置")]
    [Tooltip("时间加速倍数 (1 = 实时, 60 = 1分钟等于1小时)")]
    public float timeScale = 60f;
    
    [Tooltip("地球自转周期（秒，真实值：86400）")]
    public float earthRotationPeriod = 86400f;
    
    [Header("引用")]
    [Tooltip("地球物体（将会绕Y轴旋转）")]
    public Transform earthTransform;
    
    [Tooltip("是否旋转地球（如果false，则旋转太阳光源）")]
    public bool rotateEarth = true;
    
    [Tooltip("太阳光源（如果不旋转地球，则旋转此光源）")]
    public Transform sunTransform;
    
    // 当前仿真时间（秒）
    private float simulationTime = 0f;
    
    // 地球每秒旋转角度（度/秒）
    private float earthRotationSpeed;
    
    void Start()
    {
        // 计算地球旋转速度：360度 / 自转周期
        earthRotationSpeed = 360f / earthRotationPeriod;
        
        // 设置初始时间（可以从0开始，或从某个特定时间开始）
        simulationTime = 0f;
    }
    
    void Update()
    {
        // 更新仿真时间
        float deltaTime = Time.deltaTime * timeScale;
        simulationTime += deltaTime;
        
        // 计算旋转角度
        float rotationAngle = earthRotationSpeed * deltaTime;
        
        if (rotateEarth && earthTransform != null)
        {
            // 方案1：旋转地球（绕Y轴，逆时针从北极看）
            earthTransform.Rotate(Vector3.up, rotationAngle, Space.World);
        }
        else if (!rotateEarth && sunTransform != null)
        {
            // 方案2：保持地球不动，旋转太阳光源
            // 太阳需要绕地球旋转（相对运动）
            sunTransform.RotateAround(earthTransform != null ? earthTransform.position : Vector3.zero, 
                                      Vector3.up, rotationAngle);
        }
    }
    
    /// <summary>
    /// 设置时间加速倍数
    /// </summary>
    public void SetTimeScale(float scale)
    {
        timeScale = Mathf.Max(0f, scale);
    }
    
    /// <summary>
    /// 获取当前仿真时间
    /// </summary>
    public float GetSimulationTime()
    {
        return simulationTime;
    }
    
    /// <summary>
    /// 重置仿真时间
    /// </summary>
    public void ResetTime()
    {
        simulationTime = 0f;
    }
    
    /// <summary>
    /// 获取地球旋转角度（度，相对于初始位置）
    /// </summary>
    public float GetEarthRotationAngle()
    {
        return (simulationTime * earthRotationSpeed) % 360f;
    }
}
