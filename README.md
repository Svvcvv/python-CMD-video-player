# python-CMD-video-player
## Play video (without sound) or display img in cmd
### Requirement:Python
### `pip install opencv-python numpy` to install requirements
### ` python3 ascii.py -h` To obtain command-line parameters


```
usage: ASCII.exe [-h] [--scale SCALE] [--block] [--no-color] [--skip-threshold SKIP_THRESHOLD]
                 [--skip-frames SKIP_FRAMES] [--aspect-ratio ASPECT_RATIO] [--brightness BRIGHTNESS]
                 [--contrast CONTRAST] [--loop]
                 file_path

用途:将图片或视频转换为文本输出 本工具由Svvcvv@github制作 版本v0.1.6

positional arguments:
  file_path             图片或视频文件路径

options:
  -h, --help            show this help message and exit
  --scale SCALE         缩放因子 (0.1-1.0, 默认: 1.0)
  --block               使用块字符模式 (默认: ASCII字符)
  --no-color            禁用彩色输出 (默认: 启用颜色)
  --skip-threshold SKIP_THRESHOLD
                        跳帧阈值FPS (默认: 1.0)
  --skip-frames SKIP_FRAMES
                        跳帧数量 (默认: 10)
  --aspect-ratio ASPECT_RATIO
                        字符宽高比 (默认: 2.0)
  --brightness BRIGHTNESS
                        初始亮度调整 (-100 到 100, 默认: 0)
  --contrast CONTRAST   初始对比度调整 (0.1 到 3.0, 默认: 1.0)
  --loop                启用循环播放

示例: python ascii.py video.mp4 --scale 1.0 --block //关于播放时的操作帮助在播放时按i显示
```
