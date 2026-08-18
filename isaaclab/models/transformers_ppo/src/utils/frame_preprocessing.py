import numpy as np
import cv2


def preprocess_frame(frame: np.array, target_size: int) -> np.array:
    """
    Preprocess a frame by converting to grayscale, resizing while maintaining aspect ratio,
    and padding to square with letterboxing/pillarboxing as needed.

    Args:
        frame (np.array): Input of a single frame
        target_size (int): Goal size of each side of the eventual square image
        final_size (int, optional): For resizing an image to a smaller size. Defaults to None.

    Returns:
        np.array: Resultant image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Calculate aspect ratio
    height, width = gray.shape
    aspect_ratio = width / height

    # Calculate new dimensions to fit within target_size while maintaining aspect ratio
    if aspect_ratio > 1:  # Landscape
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:  # Portrait or square
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # Resize
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a black square canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    # Calculate offsets to center the image
    x_offset = (target_size - new_width) // 2  # Pillarboxing
    y_offset = (target_size - new_height) // 2  # Letterboxing

    # Place the resized image onto the canvas
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    # Normalize
    normalized = canvas / 255.0
    return normalized


def stack_frames(
    stacked_frames: np.array,
    frame: np.array,
    is_new_episode: bool,
    target_height: int,
    num_stack: int,
) -> np.array:
    """
    Adds a frame onto the frame stack.

    Args:
        stacked_frames (np.array): Existing stack of frames
        frame (np.array): New frame to add on top
        is_new_episode (bool): New episode flag
        final_size (int): Goal size of the square
        target_height (int): For resizing an image to a smaller size.
        num_stack (int): Amount of frames to stack

    Returns:
        np.array: Resultant frame stack
    """
    preprocessed = preprocess_frame(frame, target_height)

    if is_new_episode or stacked_frames is None:
        # Clear our stacked_frames
        stacked_frames = np.stack([preprocessed] * num_stack, axis=0)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
        stacked_frames[-1] = preprocessed

    return stacked_frames
