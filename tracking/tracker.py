"""
SORT (Simple Online and Realtime Tracking) implementation
for tracking detected humans across video frames
"""

import numpy as np
from collections import defaultdict


class KalmanFilter:
    """
    Simple Kalman filter for bounding box tracking.
    State: [x, y, s, r, x', y', s']
    where (x,y) is center, s is scale, r is aspect ratio
    """
    
    def __init__(self):
        # State dimension: [cx, cy, s, r, vx, vy, vs]
        self.ndim = 7
        self.dt = 1.0
        
        # State transition matrix
        self.F = np.eye(self.ndim)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt
        self.F[2, 6] = self.dt
        
        # Measurement matrix (we observe x, y, s, r)
        self.H = np.zeros((4, self.ndim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        
        # Process noise covariance
        self.Q = np.eye(self.ndim) * 0.03
        
        # Measurement noise covariance
        self.R = np.eye(4) * 1.0
        
        # Error covariance
        self.P = np.eye(self.ndim) * 1000.0
        
        # State
        self.x = np.zeros((self.ndim, 1))
    
    def predict(self):
        """Predict next state."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
    
    def update(self, z):
        """
        Update state with measurement.
        z: [cx, cy, s, r]
        """
        y = z.reshape(-1, 1) - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.ndim) - np.dot(K, self.H), self.P)
        return self.x
    
    def get_state(self):
        """Get current state as [cx, cy, s, r]."""
        return self.x[:4].reshape(-1)


class Track:
    """Represents a tracked object."""
    
    def __init__(self, detection, track_id, max_age=30):
        """
        Initialize track from detection.
        
        Args:
            detection: Detection dict with 'bbox' key
            track_id: Unique track ID
            max_age: Maximum frames to keep track without update
        """
        self.track_id = track_id
        self.max_age = max_age
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Convert bbox to [cx, cy, s, r] format
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h  # area
        r = w / h if h > 0 else 1.0
        
        # Initialize Kalman filter
        self.kf = KalmanFilter()
        self.kf.x[:4] = [cx, cy, s, r]
        
        # Store detection info
        self.detection = detection.copy()
    
    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection):
        """Update track with new detection."""
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h
        r = w / h if h > 0 else 1.0
        
        z = np.array([cx, cy, s, r])
        self.kf.update(z)
        self.detection = detection.copy()
        self.hits += 1
        self.time_since_update = 0
    
    def get_bbox(self):
        """Get predicted bounding box."""
        state = self.kf.get_state()
        cx, cy, s, r = state
        
        # Convert back to [x1, y1, x2, y2]
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return [x1, y1, x2, y2]
    
    def is_confirmed(self):
        """Check if track is confirmed (has enough hits)."""
        return self.hits >= 3
    
    def is_deleted(self):
        """Check if track should be deleted."""
        return self.time_since_update > self.max_age


class SORTTracker:
    """
    SORT (Simple Online and Realtime Tracking) tracker.
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum frames to keep track without update
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _associate_detections_to_tracks(self, detections, track_predictions):
        """Associate detections to tracks using IoU."""
        if len(track_predictions) == 0:
            return [], list(range(len(detections)))
        
        if len(detections) == 0:
            return list(range(len(track_predictions))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(track_predictions), len(detections)))
        for i, track_pred in enumerate(track_predictions):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track_pred, det['bbox'])
        
        # Simple greedy association
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_predictions)))
        
        # Sort by IoU (highest first)
        iou_pairs = []
        for i in range(len(track_predictions)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.iou_threshold:
                    iou_pairs.append((iou_matrix[i, j], i, j))
        
        iou_pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Greedy matching
        for _, i, j in iou_pairs:
            if i in unmatched_tracks and j in unmatched_detections:
                matched_indices.append((i, j))
                unmatched_tracks.remove(i)
                unmatched_detections.remove(j)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox' key
            frame: Optional frame (for future extensions)
        
        Returns:
            List of detections with added 'track_id' field
        """
        self.frame_count += 1
        
        # Predict tracks
        track_predictions = []
        for track in self.tracks:
            track.predict()
            track_predictions.append(track.get_bbox())
        
        # Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, track_predictions
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(detections[det_idx], self.next_id, self.max_age)
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return confirmed tracks with track IDs
        confirmed_tracks = []
        for track in self.tracks:
            if track.is_confirmed():
                det = track.detection.copy()
                det['track_id'] = track.track_id
                det['bbox'] = track.get_bbox()  # Use predicted bbox
                confirmed_tracks.append(det)
        
        return confirmed_tracks

