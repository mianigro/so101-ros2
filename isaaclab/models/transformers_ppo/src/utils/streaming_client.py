#!/usr/bin/env python3
"""
Simple Frame Client
Sends numpy arrays to the streaming server
"""

import socket
import numpy as np
import time
import pickle
import struct
import cv2
import numpy as np


def format_time(seconds):
    seconds = seconds
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.0f}s"


class SimpleFrameClient:
    def __init__(self, host="192.168.0.110", port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        """Connect to the server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.connected = False
            print("Disconnected from server")

    def send_frame(
        self, frame1, frame2, time_taken=None, training_steps=None, mean_score=None
    ):
        """Send two numpy frames to the server, combined side by side"""
        if not self.connected:
            return False
        try:
            # Ensure frames have compatible dimensions
            if frame1.shape[0] != frame2.shape[0]:  # Different heights
                # Resize frame2 to match frame1's height using OpenCV
                new_width = int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0]))
                frame2 = cv2.resize(frame2, (new_width, frame1.shape[0]))

            if len(frame2.shape) == 2:  # Handle grayscale
                frame2 = np.expand_dims(frame2, axis=-1)

            # Combine frames side by side
            combined_frame = np.concatenate((frame1, frame2), axis=1)

            # Add overlays if provided
            if (
                training_steps is not None
                or mean_score is not None
                or time_taken is not None
            ):
                # Create a copy to avoid modifying the original
                combined_frame = combined_frame.copy()

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                color = (255, 255, 255)  # White text

                # Get frame dimensions
                frame_height, frame_width = combined_frame.shape[:2]

                # Top-left overlays
                y_offset = 20

                # Add training steps text overlay (top-left)
                if training_steps is not None:
                    steps_text = f"Steps Learned: {training_steps.item()}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        steps_text, font, font_scale, thickness
                    )

                    x = 5
                    y = text_height + y_offset

                    # Handle different image formats (grayscale vs color)
                    if len(combined_frame.shape) == 2:  # Grayscale
                        cv2.putText(
                            combined_frame,
                            steps_text,
                            (x, y),
                            font,
                            font_scale,
                            255,
                            thickness,
                        )
                    elif combined_frame.shape[2] == 1:  # Single channel
                        cv2.putText(
                            combined_frame,
                            steps_text,
                            (x, y),
                            font,
                            font_scale,
                            255,
                            thickness,
                        )
                    else:  # Color image (BGR or RGB)
                        cv2.putText(
                            combined_frame,
                            steps_text,
                            (x, y),
                            font,
                            font_scale,
                            color,
                            thickness,
                        )

                    y_offset += text_height + 10  # Move down for next line

                # Add mean score text overlay (top-left)
                if mean_score is not None:
                    score_text = f"Ave Score: {mean_score:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        score_text, font, font_scale, thickness
                    )

                    x = 5
                    y = text_height + y_offset

                    # Handle different image formats (grayscale vs color)
                    if len(combined_frame.shape) == 2:  # Grayscale
                        cv2.putText(
                            combined_frame,
                            score_text,
                            (x, y),
                            font,
                            font_scale,
                            255,
                            thickness,
                        )
                    elif combined_frame.shape[2] == 1:  # Single channel
                        cv2.putText(
                            combined_frame,
                            score_text,
                            (x, y),
                            font,
                            font_scale,
                            255,
                            thickness,
                        )
                    else:  # Color image (BGR or RGB)
                        cv2.putText(
                            combined_frame,
                            score_text,
                            (x, y),
                            font,
                            font_scale,
                            color,
                            thickness,
                        )

                # Add training time text overlay (bottom-right)
                if time_taken is not None:
                    font_scale_time = font_scale - 0.1
                    thickness_time = thickness - 1
                    time_text = f"Played: {format_time(time_taken)}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        time_text, font, font_scale_time, thickness
                    )

                    # Position in bottom-right corner
                    x = (
                        frame_width - text_width - 10
                    )  # 10 pixels padding from right edge
                    y = 15

                    # Handle different image formats (grayscale vs color)
                    if len(combined_frame.shape) == 2:  # Grayscale
                        cv2.putText(
                            combined_frame,
                            time_text,
                            (x, y),
                            font,
                            font_scale_time,
                            255,
                            thickness,
                        )
                    elif combined_frame.shape[2] == 1:  # Single channel
                        cv2.putText(
                            combined_frame,
                            time_text,
                            (x, y),
                            font,
                            font_scale_time,
                            255,
                            thickness,
                        )
                    else:  # Color image (BGR or RGB)
                        cv2.putText(
                            combined_frame,
                            time_text,
                            (x, y),
                            font,
                            font_scale_time,
                            color,
                            thickness,
                        )

            # Serialize frame
            frame_data = pickle.dumps(combined_frame)
            frame_size = len(frame_data)

            # Send size first (4 bytes)
            self.socket.send(struct.pack("!I", frame_size))
            # Send frame data
            self.socket.send(frame_data)
            return True

        except Exception as e:
            print(f"Error sending frame: {e}")
            return False

    def send_frame_training(
        self, frame1, frame2, epoch, epoch_size, batch_n, batch_size
    ):
        """Send two numpy frames to the server, combined side by side"""
        if not self.connected:
            return False
        try:
            # Ensure frames have compatible dimensions
            if frame1.shape[0] != frame2.shape[0]:  # Different heights
                # Resize frame2 to match frame1's height using OpenCV
                new_width = int(frame2.shape[1] * (frame1.shape[0] / frame2.shape[0]))
                frame2 = cv2.resize(frame2, (new_width, frame1.shape[0]))

            if len(frame2.shape) == 2:  # Handle grayscale
                frame2 = np.expand_dims(frame2, axis=-1)

            # Combine frames side by side
            combined_frame = np.concatenate((frame1, frame2), axis=1)

            # Create a copy to avoid modifying the original
            combined_frame = combined_frame.copy()

            # Dim the image by reducing brightness
            combined_frame = (combined_frame * 0.6).astype(combined_frame.dtype)

            # Add "Training Interval" text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            main_text = f"Training Epoch: {epoch + 1} / {epoch_size}"
            sub_text = f"Batch: {batch_n + 1} / {batch_size}"
            main_font_scale = 1.0
            sub_font_scale = 0.6
            main_thickness = 3
            sub_thickness = 2
            color = (255, 255, 255)  # White text

            # Get text sizes
            (main_width, main_height), main_baseline = cv2.getTextSize(
                main_text, font, main_font_scale, main_thickness
            )
            (sub_width, sub_height), sub_baseline = cv2.getTextSize(
                sub_text, font, sub_font_scale, sub_thickness
            )

            # Calculate positions to center both texts
            frame_height, frame_width = combined_frame.shape[:2]

            # Main text position (centered, slightly above center)
            main_x = (frame_width - main_width) // 2
            main_y = (frame_height - main_height) // 2

            # Sub text position (centered, below main text)
            sub_x = (frame_width - sub_width) // 2
            sub_y = main_y + main_height + 20  # 20 pixels below main text

            # Handle different image formats (grayscale vs color)
            if len(combined_frame.shape) == 2:  # Grayscale
                cv2.putText(
                    combined_frame,
                    main_text,
                    (main_x, main_y),
                    font,
                    main_font_scale,
                    255,
                    main_thickness,
                )
                cv2.putText(
                    combined_frame,
                    sub_text,
                    (sub_x, sub_y),
                    font,
                    sub_font_scale,
                    255,
                    sub_thickness,
                )
            elif combined_frame.shape[2] == 1:  # Single channel
                cv2.putText(
                    combined_frame,
                    main_text,
                    (main_x, main_y),
                    font,
                    main_font_scale,
                    255,
                    main_thickness,
                )
                cv2.putText(
                    combined_frame,
                    sub_text,
                    (sub_x, sub_y),
                    font,
                    sub_font_scale,
                    255,
                    sub_thickness,
                )
            else:  # Color image (BGR or RGB)
                cv2.putText(
                    combined_frame,
                    main_text,
                    (main_x, main_y),
                    font,
                    main_font_scale,
                    color,
                    main_thickness,
                )
                cv2.putText(
                    combined_frame,
                    sub_text,
                    (sub_x, sub_y),
                    font,
                    sub_font_scale,
                    color,
                    sub_thickness,
                )

            # Serialize frame
            frame_data = pickle.dumps(combined_frame)
            frame_size = len(frame_data)

            # Send size first (4 bytes)
            self.socket.send(struct.pack("!I", frame_size))
            # Send frame data
            self.socket.send(frame_data)
            return True

        except Exception as e:
            print(f"Error sending frame: {e}")
            return False


def test_random_noise():
    """Test with random noise"""
    client = SimpleFrameClient()

    if not client.connect():
        return

    try:
        print("Sending random noise frames...")
        print("Check OBS/VLC: rtmp://localhost:1935/live/stream")

        for i in range(1000):
            # Generate random RGB frame (480x640x3)
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

            if not client.send_frame(frame):
                print("Failed to send frame")
                break

            if i % 30 == 0:
                print(f"Sent {i} frames...")

            time.sleep(1 / 30)  # 30 FPS

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()


def test_gradient():
    """Test with a moving gradient"""
    client = SimpleFrameClient()

    if not client.connect():
        return

    try:
        print("Sending gradient animation...")
        print("Check OBS/VLC: rtmp://localhost:1935/live/stream")

        for i in range(1000):
            # Create a moving gradient
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Create gradient effect
            x = np.linspace(0, 1, 640)
            y = np.linspace(0, 1, 480)
            X, Y = np.meshgrid(x, y)

            # Animated colors
            t = i * 0.1
            frame[:, :, 0] = ((np.sin(X * 5 + t) + 1) * 127).astype(np.uint8)
            frame[:, :, 1] = ((np.cos(Y * 5 + t) + 1) * 127).astype(np.uint8)
            frame[:, :, 2] = ((np.sin(X * Y * 10 + t) + 1) * 127).astype(np.uint8)

            if not client.send_frame(frame):
                print("Failed to send frame")
                break

            if i % 30 == 0:
                print(f"Sent {i} frames...")

            time.sleep(1 / 30)  # 30 FPS

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()


def main():
    import sys

    print("Simple Frame Client Test")
    print("Make sure the server is running: python3 simple_frame_server.py")
    print()
    print("Choose test:")
    print("1. Random noise")
    print("2. Animated gradient")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        test_random_noise()
    elif choice == "2":
        test_gradient()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
