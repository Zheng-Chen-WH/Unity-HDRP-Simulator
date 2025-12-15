using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[Serializable]
public class CommandData
{
    public string command;
    public string name;
    public float[] position;
    public float[] rotation;
    public string camera_name;
    public int width;
    public int height;
    public float dt;
}

public class SimulationServer : MonoBehaviour
{
    public int port = 5000;
    private TcpListener server;
    private TcpClient client;
    private NetworkStream stream;
    
    // 简单的初始状态记录 (你可以扩展为 Dictionary 来记录所有物体的初始位置)
    private Vector3 initialPos = Vector3.zero;
    private Quaternion initialRot = Quaternion.identity;

    void Start()
    {
        server = new TcpListener(IPAddress.Any, port);
        server.Start();
        Debug.Log($"[Server] Simulation Server started on port {port}.");
    }

    void Update()
    {
        if (server.Pending())
        {
            client = server.AcceptTcpClient();
            stream = client.GetStream();
            Debug.Log("[Server] Python client connected!");
        }

        if (client != null && client.Connected && stream.DataAvailable)
        {
            ProcessClientRequest();
        }
    }

    void ProcessClientRequest()
    {
        byte[] lenBytes = new byte[4];
        int bytesRead = stream.Read(lenBytes, 0, 4);
        if (bytesRead < 4) return;

        int jsonLen = BitConverter.ToInt32(lenBytes, 0);
        byte[] jsonBytes = new byte[jsonLen];
        int totalRead = 0;
        while (totalRead < jsonLen)
        {
            totalRead += stream.Read(jsonBytes, totalRead, jsonLen - totalRead);
        }
        
        string jsonStr = Encoding.UTF8.GetString(jsonBytes);
        CommandData data = JsonUtility.FromJson<CommandData>(jsonStr);

        HandleCommand(data);
    }

    void HandleCommand(CommandData data)
    {
        string responseJson = "{\"status\":\"ok\"}";
        byte[] extraData = null;

        try
        {
            switch (data.command)
            {
                case "set_pose":
                    GameObject obj = GameObject.Find(data.name);
                    if (obj != null)
                    {
                        obj.transform.position = new Vector3(data.position[0], data.position[1], data.position[2]);
                        obj.transform.rotation = Quaternion.Euler(data.rotation[0], data.rotation[1], data.rotation[2]);
                    }
                    break;

                case "get_image":
                    extraData = CaptureImage(data.camera_name, data.width, data.height);
                    responseJson = $"{{\"status\":\"ok\", \"image_size\":{extraData.Length}}}";
                    break;

                case "switch_view":
                    SwitchCamera(data.camera_name);
                    break;

                case "reset":
                    // 简单重置：把 Sat1 归零。
                    // 在实际项目中，你可能需要重置整个场景或重新加载 Scene
                    GameObject sat = GameObject.Find("Sat1");
                    if (sat != null)
                    {
                        sat.transform.position = initialPos;
                        sat.transform.rotation = initialRot;
                    }
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error: {e.Message}");
            responseJson = $"{{\"status\":\"error\", \"message\":\"{e.Message}\"}}";
        }

        SendResponse(responseJson, extraData);
    }

    void SwitchCamera(string camName)
    {
        // 找到所有相机，关闭它们
        Camera[] allCams = Camera.allCameras;
        foreach (var c in allCams)
        {
            c.enabled = false;
        }

        // 开启目标相机
        GameObject targetObj = GameObject.Find(camName);
        if (targetObj != null)
        {
            Camera targetCam = targetObj.GetComponent<Camera>();
            if (targetCam != null)
            {
                targetCam.enabled = true;
            }
        }
    }

    byte[] CaptureImage(string camName, int width, int height)
    {
        GameObject camObj = GameObject.Find(camName);
        if (camObj == null) return new byte[0];

        Camera cam = camObj.GetComponent<Camera>();
        // 临时启用相机以进行渲染
        bool wasEnabled = cam.enabled;
        cam.enabled = true;

        RenderTexture rt = new RenderTexture(width, height, 24);
        cam.targetTexture = rt;
        cam.Render();

        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        // 恢复相机状态
        cam.enabled = wasEnabled;

        return tex.EncodeToPNG();
    }

    void SendResponse(string json, byte[] binaryData)
    {
        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
        byte[] lenBytes = BitConverter.GetBytes(jsonBytes.Length);
        stream.Write(lenBytes, 0, 4);
        stream.Write(jsonBytes, 0, jsonBytes.Length);
        if (binaryData != null) stream.Write(binaryData, 0, binaryData.Length);
    }

    void OnApplicationQuit()
    {
        if (server != null) server.Stop();
    }
}