import socket
import json
import struct
import time
import numpy as np
from PIL import Image
import io

class UnityClient:
    def __init__(self, ip='127.0.0.1', port=5000, buffer_size=4096):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.is_connected = False

    def connect(self):
        """Connect to the Unity server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
            self.is_connected = True
            print(f"Successfully connected to Unity at {self.ip}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to Unity: {e}")
            self.is_connected = False

    def disconnect(self):
        """Disconnect from the Unity server."""
        if self.socket:
            self.socket.close()
            self.is_connected = False

    def _send_request(self, command, data=None):
        """Send a JSON request to Unity and receive a response."""
        if not self.is_connected:
            raise Exception("Not connected to Unity server.")

        request = {'command': command}
        if data:
            request.update(data)

        # Send JSON length followed by JSON string
        json_str = json.dumps(request)
        json_bytes = json_str.encode('utf-8')
        
        # Header: 4 bytes for length of the message
        self.socket.sendall(struct.pack('<I', len(json_bytes)))
        self.socket.sendall(json_bytes)

        # Receive response
        # First read header (length of response)
        len_bytes = self._recv_all(4)
        if not len_bytes:
            return None
        
        response_len = struct.unpack('<I', len_bytes)[0]
        response_bytes = self._recv_all(response_len)
        
        # Check if response is binary (image) or JSON
        # This depends on your protocol design. 
        # Here we assume the response is always JSON unless it's 'get_image' which might need special handling
        # But for simplicity, let's assume Unity sends a JSON wrapper, possibly with base64 image or a separate binary stream.
        # A robust way: Unity sends JSON. If JSON says "type": "image", then read raw bytes next.
        
        try:
            response_str = response_bytes.decode('utf-8')
            return json.loads(response_str)
        except:
            # If it fails to decode as JSON, it might be raw data (handle accordingly if needed)
            return response_bytes

    def _recv_all(self, n):
        """Helper to receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def get_image(self, camera_name="MainCamera", width=640, height=480):
        """
        Request an image from Unity.
        Protocol: Send command -> Receive JSON header (with image size) -> Receive Raw Image Bytes
        """
        req = {
            'command': 'get_image',
            'camera_name': camera_name,
            'width': width,
            'height': height
        }
        
        # Send request
        json_str = json.dumps(req)
        json_bytes = json_str.encode('utf-8')
        self.socket.sendall(struct.pack('<I', len(json_bytes)))
        self.socket.sendall(json_bytes)

        # Receive header (JSON)
        len_bytes = self._recv_all(4)
        header_len = struct.unpack('<I', len_bytes)[0]
        header_bytes = self._recv_all(header_len)
        header = json.loads(header_bytes.decode('utf-8'))

        if header.get('status') != 'ok':
            print(f"Error getting image: {header.get('message')}")
            return None

        image_size = header['image_size']
        image_bytes = self._recv_all(image_size)
        
        # Convert to numpy array or PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)

    def switch_view(self, camera_name):
        """Switch the main view in Unity to the specified camera."""
        return self._send_request('switch_view', {'camera_name': camera_name})

    def add_object(self, name, prefab_name, position=[0,0,0], rotation=[0,0,0]):
        """Instantiate an object in Unity."""
        return self._send_request('add_object', {
            'name': name,
            'prefab_name': prefab_name,
            'position': position,
            'rotation': rotation
        })

    def set_object_pose(self, name, position, rotation):
        """Set the position and rotation of an object."""
        return self._send_request('set_pose', {
            'name': name,
            'position': position,
            'rotation': rotation
        })

    def step(self, dt):
        """Advance the simulation by dt seconds."""
        return self._send_request('step', {'dt': dt})

    def reset(self):
        """Reset the environment."""
        return self._send_request('reset')
    
    def set_time_scale(self, time_scale):
        """Set the simulation time scale (1 = real-time, 60 = 1 minute = 1 hour)."""
        return self._send_request('set_time_scale', {'time_scale': time_scale})
    
    def get_time_scale(self):
        """Get current time scale and simulation time."""
        return self._send_request('get_time_scale')
    
    def set_orbit(self, orbit_altitude_km):
        """
        Set satellite orbit altitude in kilometers.
        Returns orbital period in seconds.
        
        Example altitudes:
        - 400 km: Low Earth Orbit (LEO), ~90 min period
        - 35786 km: Geostationary Orbit (GEO), ~24 hour period
        """
        return self._send_request('set_orbit', {'orbit_altitude': orbit_altitude_km})
    
    def save_image(self, camera_name="MainCamera", width=1920, height=1080, file_path=None):
        """
        Capture and save an image from Unity to disk.
        
        Args:
            camera_name: Name of the camera in Unity
            width: Image width
            height: Image height
            file_path: Full path where to save (optional, auto-generated if None)
        
        Returns:
            dict with 'file_path' and 'size' if successful
        """
        return self._send_request('save_image', {
            'camera_name': camera_name,
            'width': width,
            'height': height,
            'file_path': file_path or ""
        })
    
    def update_satellite_pose(self, name, position, rotation):
        """
        Update satellite position and rotation (optimized for frequent updates).
        This is the same as set_object_pose but with a clearer name for satellites.
        
        Args:
            name: Object name in Unity
            position: [x, y, z] in Unity units
            rotation: [roll, pitch, yaw] in degrees
        """
        return self.set_object_pose(name, position, rotation)
    
    def update_earth_rotation(self, name, rotation_y):
        """
        Update Earth rotation around Y axis.
        
        Args:
            name: Earth object name in Unity
            rotation_y: Rotation angle in degrees
        """
        return self.set_object_pose(name, [0, 0, 0], [0, rotation_y, 0])
    
    def batch_update_poses(self, updates):
        """
        Update multiple object poses in one request (future optimization).
        
        Args:
            updates: List of dicts with 'name', 'position', 'rotation'
        
        Note: This requires Unity server support for batch commands.
        For now, it calls update_satellite_pose multiple times.
        """
        results = []
        for update in updates:
            result = self.update_satellite_pose(
                update['name'],
                update['position'],
                update['rotation']
            )
            results.append(result)
        return results

if __name__ == "__main__":
    client = UnityClient()
    client.connect() 
    
    try:
        # 切换到外部视角 (需要在 Unity 里先创建 ObserverCamera)
        print("Switching to Observer View...")
        client.switch_view("ObserverCamera")
        time.sleep(1)

        # 移动物体
        print("Moving satellite...")
        for i in range(10):
            # 简单的移动动画
            client.set_object_pose("Sat1", [i*0.5, 0, 0], [0, 90, 0])
            time.sleep(0.1)
        
        # 切换回卫星视角并截图
        print("Switching to Satellite View & Capturing...")
        client.switch_view("MainCamera")
        time.sleep(0.5)
        img = client.get_image(camera_name="MainCamera")
        if img is not None:
            print(f"Image received! Shape: {img.shape}")
            Image.fromarray(img).show()

    finally:
        # 重置位置
        print("Resetting simulation...")
        client.reset()
        client.disconnect()