# Volume-Controller

This project uses OpenCV and MediaPipe to control system volume with hand gestures.  By tracking the distance between the thumb and index finger, it allows for intuitive volume adjustments.

## Features

*   Real-time hand tracking using MediaPipe.
*   Volume control based on finger distance.
*   Visual feedback with an on-screen volume bar and percentage.

## Requirements

*   Python 3
*   See `requirements.txt` for a full list of dependencies.

## Usage

1.  Clone the repository.

2.  Create a Python virtual environment (recommended):

    ```bash
    python3 -m venv myenv  # Create the virtual environment
    source myenv/bin/activate  # Activate the environment (Linux/macOS)
    myenv\Scripts\activate  # Activate the environment (Windows)
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Run the main script (`main.py`):

    ```bash
    python main.py
    ```

## How it works

The script captures video from your webcam, detects hand landmarks using MediaPipe, and calculates the distance between your thumb and index finger. This distance is then mapped to the system volume using `np.interp`.  The `pycaw` library is used to interact with the Windows audio API.
