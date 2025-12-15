using UnityEngine;

[ExecuteAlways] // 让你在编辑器里拖动太阳时也能实时看到效果，不用运行游戏
public class SunSynchronizer : MonoBehaviour
{
    [Tooltip("场景里的真实太阳光")]
    public Light sunLight;

    [Tooltip("地球的材质球")]
    public Material earthMaterial;

    [Tooltip("光照强度缩放（HDRP光源很亮，可能需要调整）")]
    [Range(0.01f, 1f)]
    public float intensityScale = 0.1f;

    // Shader 里的属性名 ID，提前获取以提高性能
    private int sunDirID;
    private int sunColorID;

    void OnEnable()
    {
        // 对应 Shader 里的 _SunDir 和 _SunColor
        sunDirID = Shader.PropertyToID("_SunDir");
        sunColorID = Shader.PropertyToID("_SunColor");
    }

    void Update()
    {
        if (sunLight == null || earthMaterial == null) return;

        // 获取光源方向（指向太阳）
        Vector3 lightDir = -sunLight.transform.forward;

        // 获取光源颜色，根据需要缩放强度
        Color lightColor = sunLight.color * (sunLight.intensity * intensityScale);

        // 传给 Shader
        earthMaterial.SetVector(sunDirID, lightDir);
        earthMaterial.SetColor(sunColorID, lightColor); 
    }
}