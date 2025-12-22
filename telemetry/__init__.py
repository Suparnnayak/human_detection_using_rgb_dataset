"""
Telemetry module for Pixhawk 2.4.8 / MAVLink integration
Provides GPS, altitude, and attitude data for SAR detections
"""

from .mavlink_interface import MAVLinkInterface, TelemetryData

__all__ = ['MAVLinkInterface', 'TelemetryData']

