import io
import struct
import websocket
import threading
import numpy as np
from PIL import Image
from typing import Union, Dict, List


class MagiClawClient:
    all_obs_keys = ['rgb', 'depth', 'pose', 'width']

    def __init__(
        self,
        ip_address: str,
        obs_keys: Union[List, None] = None
    ) -> None:
        if obs_keys is None:
            obs_keys = self.all_obs_keys
        self.obs_keys = obs_keys   
        self.obs = dict()
        self.wsapp = websocket.WebSocketApp(ip_address, on_message=self.on_message)
        self.thread = threading.Thread(target=self.wsapp.run_forever)
        self.thread.start()
        print(f"Client \"{ip_address}\" is running.")

    def on_message(self, ws, message) -> None:
        offset = 0
        # Gripper width
        width = struct.unpack('f', message[offset: offset + 4])[0]
        offset += 4
        if 'width' in self.obs_keys:
            self.obs['width'] = width

        # Phone pose (4, 4)
        pose = []
        for _ in range(16):
            value = struct.unpack('f', message[offset: offset + 4])[0]
            pose.append(value)
            offset += 4
        pose = np.array(pose).reshape((4, 4)).transpose()
        if 'pose' in self.obs_keys:
            self.obs['pose'] = pose

        # Depth image (192, 256)
        depth_size = 256 * 192 * 2
        depth = np.frombuffer(message[offset: offset + depth_size], dtype=np.uint16).reshape((192, 256))
        offset += depth_size
        if 'depth' in self.obs_keys:
            self.obs['depth'] = depth

        # RGB image (480, 640)
        rgb = Image.open(io.BytesIO(message[offset:]))
        rgb = np.array(rgb, dtype=np.uint8)
        if 'rgb' in self.obs_keys:
            self.obs['rgb'] = rgb

    def get_obs(self) -> Union[Dict, None]:
        if self.obs:
            return self.obs.copy()
        else:
            return None

    def shut_down(self) -> None:
        self.thread.join()
        self.wsapp.close()


def test() -> None:
    client = MagiClawClient(ip_address="ws://10.16.3.51:8080")
    while True:
        obs = client.get_obs()
        if obs is not None:
            print("obs: ", obs)