Shader "Earth"
{
    Properties 
    {
        _AtmosphereColor ("Atmosphere Color", Color) = (0.1, 0.35, 1.0, 1.0)
        _AtmospherePow ("Atmosphere Power", Range(1.5, 8)) = 2
        _AtmosphereMultiply ("Atmosphere Multiply", Range(1, 3)) = 1.5

        _DiffuseTex("Diffuse", 2D) = "white" {}
        _CloudAndNightTex("Cloud And Night", 2D) = "black" {}
        
        // 添加高光控制
        _SpecularPower("Specular Power", Range(1, 128)) = 32
        _SpecularIntensity("Specular Intensity", Range(0, 2)) = 0.3
        _OceanShininess("Ocean Shininess", Range(0, 1)) = 0.8

        _SunDir("Sun Direction", Vector) = (0, 1, 0, 0)
        _SunColor("Sun Color", Color) = (1, 1, 1, 1)
    }

    SubShader 
    {
        ZWrite On
        ZTest LEqual

        pass
        {
        CGPROGRAM
            #include "UnityCG.cginc"
            #pragma vertex vert 
            #pragma fragment frag
            
            sampler2D _DiffuseTex;
            sampler2D _CloudAndNightTex;

            float4 _AtmosphereColor;
            float _AtmospherePow;
            float _AtmosphereMultiply;
            
            float _SpecularPower;
            float _SpecularIntensity;
            float _OceanShininess;

            float4 _SunDir;
            float4 _SunColor;

            struct vertexInput 
            {
                float4 pos				: POSITION;
                float3 normal			: NORMAL;
                float2 uv				: TEXCOORD0;
            };

            struct vertexOutput 
            {
                float4 pos			: SV_POSITION;
                float2 uv			: TEXCOORD0;
                float3 worldNormal	: TEXCOORD1;
                float3 worldPos		: TEXCOORD2;
                half diffuse		: TEXCOORD3;
                half night			: TEXCOORD4;
                half3 atmosphere	: TEXCOORD5;
            };
            
            vertexOutput vert(vertexInput input) 
            {
                vertexOutput output;
                output.pos = UnityObjectToClipPos(input.pos);
                output.uv = input.uv;

                output.worldNormal = UnityObjectToWorldNormal(input.normal);
                output.worldPos = mul(unity_ObjectToWorld, input.pos).xyz;
                
                float3 lightDir = normalize(_SunDir.xyz);
                output.diffuse = saturate(dot(lightDir, output.worldNormal) * 1.2);
                output.night = 1 - saturate(output.diffuse * 2);

                half3 viewDir = normalize(ObjSpaceViewDir(input.pos));
                half3 normalDir = input.normal;
                output.atmosphere = output.diffuse * _AtmosphereColor.rgb * 
                    pow(1 - saturate(dot(viewDir, normalDir)), _AtmospherePow) * _AtmosphereMultiply;

                return output;
            }

            half4 frag(vertexOutput input) : SV_Target
            {
                half3 colorSample = tex2D(_DiffuseTex, input.uv).rgb;
                half3 cloudAndNightSample = tex2D(_CloudAndNightTex, input.uv).rgb;
                half3 nightSample = cloudAndNightSample.ggb;
                half cloudSample = cloudAndNightSample.r;

                // 计算镜面高光（Blinn-Phong）
                float3 lightDir = normalize(_SunDir.xyz);
                float3 viewDir = normalize(_WorldSpaceCameraPos - input.worldPos);
                float3 halfDir = normalize(lightDir + viewDir);
                
                float specular = pow(saturate(dot(input.worldNormal, halfDir)), _SpecularPower);
                
                // 使用颜色亮度判断海洋（暗色=海洋，亮色=陆地）
                float oceanMask = 1.0 - saturate(dot(colorSample, float3(0.3, 0.59, 0.11)));
                oceanMask = pow(oceanMask, 2); // 让遮罩更锐利
                
                // 只在海洋区域和被光照射的地方显示高光
                specular *= oceanMask * _OceanShininess * input.diffuse;

                half4 result;
                result.rgb = (colorSample + cloudSample) * input.diffuse * _SunColor.rgb 
                            + nightSample * input.night 
                            + input.atmosphere
                            + specular * _SpecularIntensity * _SunColor.rgb; // 添加高光
                result.a = 1;
                return result;
            }
        ENDCG
        }
    }
    
    Fallback "Diffuse"
}