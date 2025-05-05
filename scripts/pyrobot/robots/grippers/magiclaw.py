import struct
import logging
import threading
import websocket
import numpy as np
from pyrobot.robots.grippers.base_gripper import Gripper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MagiClawGripper(Gripper):
    """
    Gripper control client (synchronous version)
    Provides simple and easy-to-use API to control the gripper, no need to use await
    """
    def __init__(self, hostname="magiclawpi2.local", port=1234):
        """
        Initialize the client
        
        Parameters:
            host: Server IP address or hostname
            port: Server port
        """
        self.server_url = f"ws://{hostname}:{port}"
        self.ws = None
        self.is_connected = False
        self.is_initialized = False
        self.lock = threading.Lock()  # Add thread lock to ensure thread safety
        # Connect to server
        self.connect()
        # Initialize gripper (only needed once)
        self.initialize()
        # Calibrate gripper (find max and min angles)
        self.calibrate()
        print("MagiClaw gripper is initialized.")
    
    def connect(self):
        """Connect to the gripper server"""
        if not self.is_connected:
            try:
                # Create connection using websocket-client library
                self.ws = websocket.create_connection(self.server_url)
                self.is_connected = True
                return True
            except Exception as e:
                print(f"Failed to connect to server: {str(e)}")
                return False
        return True
    
    def disconnect(self):
        """Disconnect from the server"""
        if self.is_connected and self.ws:
            self.ws.close()
            self.is_connected = False
            print("Disconnected from server")
    
    def _send_command(self, command):
        """Send string command and wait for response"""
        with self.lock:  # Use thread lock to ensure thread safety
            if not self.is_connected:
                if not self.connect():
                    return "Error: Not connected to server"
            
            try:
                self.ws.send(command)
                response = self.ws.recv()
                return response
            except Exception as e:
                print(f"Error sending command: {str(e)}")
                self.is_connected = False
                return f"Error: {str(e)}"
    
    def _send_binary(self, value):
        """Send binary data (float32) and wait for response"""
        with self.lock:  # Use thread lock to ensure thread safety
            if not self.is_connected:
                if not self.connect():
                    return "Error: Not connected to server"
            
            try:
                # Convert float to binary data
                binary_data = struct.pack('f', value)
                self.ws.send_binary(binary_data)
                response = self.ws.recv()
                return response
            except Exception as e:
                print(f"Error sending binary data: {str(e)}")
                self.is_connected = False
                return f"Error: {str(e)}"
    
    def initialize(self):
        """Initialize the gripper"""
        if not self.is_initialized:
            response = self._send_command("initialize")
            if "Error" not in response:
                self.is_initialized = True
            return response
        return "Gripper already initialized"
    
    def calibrate(self):
        """Calibrate the gripper"""
        return self._send_command("calibrate")
    
    def homing(self):
        """Return gripper to home position"""
        return self._send_command("home")
    
    def stop(self):
        """Stop the gripper"""
        return self._send_command("stop")
    
    def shutdown(self):
        """Shutdown the gripper"""
        response = self._send_command("shutdown")
        self.is_initialized = False
        return response
    
    def move(self, open_range):
        """
        Move gripper to specified open range
        
        Parameters:
            open_range: Position value (0.0-1.0), 0 means fully closed, 1 means fully open
        """
        # if not 0 <= open_range <= 1:
        #     return f"Error: Position value {open_range} must be between 0-1"
        
        # Use string command
        return self._send_command(f"move({open_range})")
    
    def move_binary(self, position):
        """
        Move gripper to specified position using binary data
        
        Parameters:
            position: Position value (0.0-1.0), 0 means fully closed, 1 means fully open
        """
        if not 0 <= position <= 1:
            return f"Error: Position value {position} must be between 0-1"
        
        return self._send_binary(position)
    
    def get_open_range(self) -> float:
        """Get gripper multi-turn open_range"""
        return float(self._send_command("get_open_range"))
    
    def get_status(self):
        """Get gripper status"""
        return self._send_command("status")