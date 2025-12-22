"""
MAVLink Interface for Pixhawk 2.4.8
Companion computer side integration (not running on Pixhawk)

This module provides functions to:
- Connect to Pixhawk via MAVLink (USB/UART/Network)
- Read GPS coordinates, altitude, heading
- Attach telemetry metadata to detections
- Prepare data for GPS mapping

NOTE: This code runs on the companion computer (laptop/Jetson/SBC),
      NOT on the Pixhawk flight controller itself.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import time


@dataclass
class TelemetryData:
    """Container for UAV telemetry data."""
    timestamp: float
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters (MSL or relative)
    heading: float  # degrees (0-360)
    pitch: float  # degrees
    roll: float  # degrees
    yaw: float  # degrees (same as heading)
    speed: float  # m/s
    gps_fix_type: int  # 0=no fix, 1=no GPS, 2=2D, 3=3D
    gps_satellites: int
    battery_voltage: Optional[float] = None
    battery_current: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'heading': self.heading,
            'pitch': self.pitch,
            'roll': self.roll,
            'yaw': self.yaw,
            'speed': self.speed,
            'gps_fix_type': self.gps_fix_type,
            'gps_satellites': self.gps_satellites,
            'battery_voltage': self.battery_voltage,
            'battery_current': self.battery_current
        }


class MAVLinkInterface:
    """
    MAVLink interface for Pixhawk 2.4.8.
    
    This class handles communication with the Pixhawk flight controller
    via MAVLink protocol. It can connect via:
    - USB serial (e.g., /dev/ttyUSB0, COM3)
    - UART serial
    - Network (TCP/UDP)
    
    The Pixhawk 2.4.8 supports both PX4 and ArduPilot firmware,
    both of which use MAVLink for communication.
    """
    
    def __init__(self, connection_string: Optional[str] = None, baudrate: int = 57600):
        """
        Initialize MAVLink interface.
        
        Args:
            connection_string: Connection string (e.g., '/dev/ttyUSB0', 'udp:127.0.0.1:14550')
                              If None, creates a mock interface for testing
            baudrate: Serial baudrate (for serial connections)
        """
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.connection = None
        self.mavlink = None
        self.is_connected = False
        self.is_mock = connection_string is None
        
        if not self.is_mock:
            try:
                from pymavlink import mavutil
                self.mavutil = mavutil
            except ImportError:
                print("Warning: pymavlink not installed. Using mock mode.")
                self.is_mock = True
    
    def connect(self) -> bool:
        """
        Connect to Pixhawk.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.is_mock:
            print("Mock mode: No actual connection to Pixhawk")
            self.is_connected = True
            return True
        
        try:
            # Create connection
            if 'udp:' in self.connection_string or 'tcp:' in self.connection_string:
                self.connection = self.mavutil.mavlink_connection(self.connection_string)
            else:
                # Serial connection
                self.connection = self.mavutil.mavlink_connection(
                    self.connection_string,
                    baud=self.baudrate
                )
            
            # Wait for heartbeat
            print("Waiting for heartbeat from Pixhawk...")
            self.connection.wait_heartbeat(timeout=10)
            print(f"Connected to Pixhawk (system {self.connection.target_system})")
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"Failed to connect to Pixhawk: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Pixhawk."""
        if self.connection:
            self.connection.close()
        self.is_connected = False
        print("Disconnected from Pixhawk")
    
    def get_telemetry(self) -> Optional[TelemetryData]:
        """
        Get current telemetry data from Pixhawk.
        
        Returns:
            TelemetryData object, or None if unavailable
        """
        if not self.is_connected:
            return None
        
        if self.is_mock:
            # Return mock data for testing
            return TelemetryData(
                timestamp=time.time(),
                latitude=37.7749,  # Example: San Francisco
                longitude=-122.4194,
                altitude=100.0,
                heading=45.0,
                pitch=0.0,
                roll=0.0,
                yaw=45.0,
                speed=5.0,
                gps_fix_type=3,
                gps_satellites=12,
                battery_voltage=12.6,
                battery_current=15.0
            )
        
        try:
            # Request telemetry messages
            # Note: In real implementation, you'd parse MAVLink messages
            # Here's a simplified version
            
            # Get GPS global position
            msg_gps = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            msg_attitude = self.connection.recv_match(type='ATTITUDE', blocking=True, timeout=1)
            msg_sys_status = self.connection.recv_match(type='SYS_STATUS', blocking=True, timeout=1)
            
            if msg_gps and msg_attitude:
                return TelemetryData(
                    timestamp=time.time(),
                    latitude=msg_gps.lat / 1e7,  # Convert from int32 to degrees
                    longitude=msg_gps.lon / 1e7,
                    altitude=msg_gps.alt / 1000.0,  # Convert from mm to meters
                    heading=msg_gps.hdg / 100.0,  # Convert from centidegrees
                    pitch=msg_attitude.pitch * 180.0 / 3.14159,  # Convert from radians
                    roll=msg_attitude.roll * 180.0 / 3.14159,
                    yaw=msg_attitude.yaw * 180.0 / 3.14159,
                    speed=0.0,  # Would need VELOCITY message
                    gps_fix_type=3,  # Would parse from GPS_RAW_INT
                    gps_satellites=0,  # Would parse from GPS_RAW_INT
                    battery_voltage=msg_sys_status.voltage_battery / 1000.0 if msg_sys_status else None,
                    battery_current=msg_sys_status.current_battery / 100.0 if msg_sys_status else None
                )
        except Exception as e:
            print(f"Error reading telemetry: {e}")
        
        return None
    
    def attach_telemetry_to_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attach telemetry data to a detection.
        
        Args:
            detection: Detection dictionary with bbox, confidence, etc.
        
        Returns:
            Detection dictionary with added telemetry fields
        """
        telemetry = self.get_telemetry()
        
        if telemetry:
            detection['telemetry'] = telemetry.to_dict()
            detection['gps_coordinates'] = {
                'latitude': telemetry.latitude,
                'longitude': telemetry.longitude,
                'altitude': telemetry.altitude
            }
            detection['uav_attitude'] = {
                'heading': telemetry.heading,
                'pitch': telemetry.pitch,
                'roll': telemetry.roll,
                'yaw': telemetry.yaw
            }
        
        return detection
    
    def get_gps_coordinates(self) -> Optional[tuple]:
        """
        Get current GPS coordinates as (latitude, longitude, altitude).
        
        Returns:
            Tuple of (lat, lon, alt) or None
        """
        telemetry = self.get_telemetry()
        if telemetry:
            return (telemetry.latitude, telemetry.longitude, telemetry.altitude)
        return None


# Example usage and testing
if __name__ == '__main__':
    # Test with mock interface
    print("Testing MAVLink interface (mock mode)...")
    interface = MAVLinkInterface()  # No connection string = mock mode
    interface.connect()
    
    telemetry = interface.get_telemetry()
    if telemetry:
        print(f"Telemetry: {telemetry}")
        print(f"GPS: ({telemetry.latitude}, {telemetry.longitude})")
        print(f"Altitude: {telemetry.altitude}m")
        print(f"Heading: {telemetry.heading}°")
    
    # Example with real connection (commented out)
    # interface = MAVLinkInterface('/dev/ttyUSB0', baudrate=57600)
    # interface.connect()
    # telemetry = interface.get_telemetry()

