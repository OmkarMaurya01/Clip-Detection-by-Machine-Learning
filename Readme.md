# Scene Comparison Project

This project compares scenes from two video files, detecting similar scenes by calculating similarity scores between frames. It supports splitting, compressing, and merging video clips based on the similarity of scenes.

## Features

- **Scene Detection**: Detects scenes from the provided video files using content detection.
- **Scene Comparison**: Compares scenes from two video files based on frame similarity.
- **Clip Creation**: Splits the original videos into clips based on similar scenes and merges them into a new video.
- **Support for Two Videos**: Allows comparing two videos by providing their file paths.

## Requirements

- Python 3.x
- OpenCV
- ffmpeg
- tqdm
- scenedetect

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/scene-comparison.git
    ```

2. Navigate to the project directory:

    ```bash
    cd scene-comparison
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that **ffmpeg** is installed on your system. You can download it from [here](https://ffmpeg.org/download.html).

## Usage

To use the scene comparison, follow these steps:

1. Replace the `video_path_1` and `video_path_2` with the paths to your video files.

2. Run the script:

    ```bash
    python scene_comparison.py
    ```

The script will:

- Detect scenes from both videos.
- Compare scenes from both videos and calculate similarity scores.
- Split the similar scenes into clips and merge them into a final output video.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for image processing.
- `scenedetect` for scene detection.
- ffmpeg for video manipulation.
