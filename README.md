# Unity Space Simulator (HDRP)

This project is a high-fidelity space simulation environment based on Unity HDRP (High Definition Render Pipeline), designed for end-to-end reinforcement learning training for spacecraft guidance. It adopts a Client-Server architecture, where Unity handles rendering and Python handles dynamics calculation and control.

## 1. Architecture Overview

*   **Server (Unity)**: Responsible for high-quality rendering (HDRP), physical lighting simulation (shadows, reflections), and scene management.
*   **Client (Python)**: Responsible for orbital dynamics calculation, RL agent control, and sending commands to Unity via TCP.

## 2. Quick Start

### Prerequisites
*   Unity 2022.3 LTS or later (with HDRP support)
*   Python 3.8+
*   Required Python packages: `numpy`, `pillow`

### Setup Steps
1.  **Unity Side**:
    *   Open the project in Unity.
    *   Open `Assets/SpaceScene`.
    *   Click the **Play (▶)** button to start the simulation server.
2.  **Python Side**:
    *   Run the client script:
        ```bash
        python unity_client.py
        ```

## 3. Python API Reference (`unity_client.py`)

The `UnityClient` class provides the following methods to interact with the simulator:

*   **`connect()`**: Connect to the Unity server.
*   **`add_object(name, prefab_name, position, rotation)`**: Instantiate a new object in the scene.
*   **`set_object_pose(name, position, rotation)`**: Set the position (x, y, z) and rotation (roll, pitch, yaw) of an object.
*   **`get_image(camera_name, width, height)`**: Capture an image from a specific camera. Returns a numpy array.
*   **`switch_view(camera_name)`**: Switch the main view in the Unity window to the specified camera.
*   **`reset()`**: Reset the environment (e.g., move satellite back to origin).
*   **`disconnect()`**: Close the connection.

## 4. Current Features
*   **HDRP Rendering**: Realistic space lighting with physical units (Lux) and correct shadows.
*   **Dual Camera System**:
    *   `MainCamera`: Mounted on the satellite for RL observation.
    *   `ObserverCamera`: External view for debugging and monitoring.
*   **TCP Communication**: Robust JSON-based protocol for real-time control.
*   **Custom Planet Shaders**: Two high-quality Earth rendering shaders with realistic lighting effects.

## 5. Planet Shader System

### 5.1 Available Shaders

#### **Earth Shader** (`Assets/Shaders/Earth.shader`)
A streamlined shader optimized for performance:
*   **Decoupled Textures**: Separate day texture, clouds, and night lights.
*   **Procedural Ocean Specular**: Automatically detects ocean areas based on color brightness.
*   **Atmospheric Scattering**: Blue rim lighting effect.
*   **Adjustable Parameters**:
    *   `Specular Power` (1-128): Controls highlight sharpness. Higher = tighter spots.
    *   `Specular Intensity` (0-2): Overall brightness of ocean reflections.
    *   `Ocean Shininess` (0-1): Ocean reflectivity strength.
    *   `Atmosphere Color/Power/Multiply`: Fine-tune the atmospheric rim effect.

#### **Reconfigurable Shader** (`Assets/Shaders/Reconfigurable Shader.shader`)
An advanced shader with full artistic control:
*   **Normal Mapping**: Adds surface detail (mountains, terrain bumps).
*   **Specular Texture**: Precise ocean mask via alpha channel.
*   **Detail Texture**: Additional high-frequency surface detail.
*   **Smart Ocean Detection**: Uses specular texture if available, falls back to brightness detection.
*   **All Earth Shader features** plus enhanced visual fidelity.

### 5.2 Lighting Synchronization System

Both shaders use the `SunSynchronizer.cs` script to automatically sync with Unity's Directional Light:

#### **Setup Instructions**
1.  **Create Synchronizer Object**:
    *   In Hierarchy, right-click → `Create Empty`.
    *   Rename to `SunSync`.
    *   Click `Add Component` → Search for `SunSynchronizer`.

2.  **Configure References**:
    *   Drag your scene's `Directional Light` (Sun) to the `Sun Light` field.
    *   Drag the planet's material to the `Earth Material` field.

3.  **Adjust Parameters**:
    *   `Intensity Scale` (0.01-1): Scales HDRP's high light intensity for shader use.
    *   Start with 0.1 and adjust based on visual result.

#### **How It Works**
*   The script reads the Directional Light's **direction** and **color** every frame.
*   It passes these values to shader properties `_SunDir` and `_SunColor`.
*   Marked with `[ExecuteAlways]`: Updates in both Play Mode and Edit Mode.
*   Rotate the Sun in Scene View → Planet lighting updates in real-time.

#### **Troubleshooting**
*   **Planet too dark**: Increase `Intensity Scale` (try 0.3-0.5).
*   **Harsh specular highlights**: Increase `Specular Power`, decrease `Specular Intensity`.
*   **No lighting response**: Check that material uses Earth or Reconfigurable shader.

---


