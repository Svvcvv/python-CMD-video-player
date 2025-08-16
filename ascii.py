import cv2
import numpy as np
import os
import time
import platform
import shutil
import random
import sys
import textwrap
import argparse
from functools import lru_cache
from datetime import datetime, timedelta
from collections import deque

if platform.system() == "Windows":
    import msvcrt
else:
    import select
    import tty
    import termios

CHAR_SET = [
    ' ', '.', ',', ':', ';', '-', '+', '*', 
    '=', '%', '$', '#', 'a', 'b', 'c', 'd', 
    'e', 'f', 'g', 'h', 'k', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 
    'x', 'z', '0', '1', '2', '3', '4', '5', 
    '6', '7', '8', '9', 'A', 'B', 'C', 'D', 
    'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 
    'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', '@', '#', '&', '^',
    '~', '?', '!', '/', '\\', '|', '(', ')',
    '[', ']', '{', '}', '<', '>'
]

@lru_cache(maxsize=None)
def create_brightness_map(num_buckets=64):
    """创建亮度映射表"""
    brightness_map = [''] * 256
    bucket_size = 256 // num_buckets
    
    for i in range(num_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < num_buckets - 1 else 256   
        char_index = int(i * len(CHAR_SET) / num_buckets)
        char_index = max(0, min(len(CHAR_SET)-1, char_index))
        char = CHAR_SET[char_index]
        for brightness in range(start, end):
            brightness_map[brightness] = char
    
    return brightness_map

def get_terminal_size():
    try:
        columns, rows = shutil.get_terminal_size()
        return max(40, columns), max(10, rows)
    except:
        return 120, 30

def calculate_scaled_size(original_width, original_height, term_width, term_height, char_aspect_ratio=2.0):
    if term_width <= 0 or term_height <= 0 or original_width <= 0 or original_height <= 0:
        return 80, 24

    target_aspect_ratio = char_aspect_ratio * term_width / term_height
    source_aspect_ratio = original_width / original_height
    
    if source_aspect_ratio > target_aspect_ratio:
        new_width = term_width
        new_height = int(new_width / source_aspect_ratio / char_aspect_ratio)
    else:
        new_height = term_height
        new_width = int(new_height * source_aspect_ratio * char_aspect_ratio)
    new_width = max(1, min(term_width, new_width))
    new_height = max(1, min(term_height, new_height))
    
    return new_width, new_height

def adjust_frame(frame, brightness=0, contrast=1.0):
    frame = np.clip(frame.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
    frame = np.clip(frame.astype(np.int32) + brightness, 0, 255).astype(np.uint8)
    
    return frame

def frame_to_ascii(frame, brightness_map, term_w, term_h, scale_factor=1.0, 
                  char_aspect_ratio=2.0, use_block_char=False, use_color=True, is_image=0):
    
    if term_w <= 0 or term_h <= 0:
        return ""
    
    if is_image == 1:
        video_display_height = max(1, term_h)
    else:
        video_display_height = max(1, term_h - 5)
    
    h, w = frame.shape[:2]
    
    # 计算最大宽度和高度限制
    max_width = int(term_w * scale_factor)
    if is_image == 1:
        max_height = float('inf')
    else:
        max_height = int(video_display_height * scale_factor)
    
    target_width, target_height = calculate_scaled_size(
        w, h, 
        max_width, 
        max_height,
        char_aspect_ratio
    )
    
    if target_width < 2 or target_height < 2:
        return ""
    
    # 插值调整大小
    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    
    if use_color:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_lines = []
    
    for i in range(h):
        line = []
        if use_color:
            line.append("\033[0m")
        
        prev_r, prev_g, prev_b = -1, -1, -1
        for j in range(w):
            brightness_val = gray_frame[i, j]      
            if use_color:
                r, g, b = rgb_frame[i, j]
                if r != prev_r or g != prev_g or b != prev_b:
                    line.append(f"\033[38;2;{r};{g};{b}m")
                    prev_r, prev_g, prev_b = r, g, b
            
            if use_block_char:
                line.append('█')
            else:
                line.append(brightness_map[brightness_val])
        
        output_lines.append(''.join(line))
    
    return '\n'.join(output_lines)

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))

def interactive_image_viewer(image_path, scale_factor=1.0, use_block_char=False, use_color=True,
                            brightness=0, contrast=1.0):
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        return
        
    if not os.path.isfile(image_path):
        print(f"错误: 不是一个文件 - {image_path}")
        return
    
    try:
        if platform.system() == "Windows" and '\\' in image_path:
            image_path = image_path.replace('\\', '/')
        
        img_bytes = np.fromfile(image_path, dtype=np.uint8)
        original_image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if original_image is None:
            print(f"无法打开图片文件: {image_path}")
            return

        # 初始化参数
        current_scale = scale_factor
        current_brightness = brightness
        current_contrast = contrast
        current_use_block_char = use_block_char
        current_use_color = use_color
        image_filename = os.path.basename(image_path)
        
        # ANSI 控制序列
        clear_screen = "\033[2J"
        move_cursor_top = "\033[H"
        clear_line = "\033[K"
        hide_cursor = "\033[?25l"
        show_cursor = "\033[?25h"
        
        print(hide_cursor)
        
        # 状态变量
        paused = False
        last_terminal_check = time.time()
        terminal_check_interval = 0.5
        term_w, term_h = get_terminal_size()
        last_term_size = (term_w, term_h)
        
        brightness_map = create_brightness_map(num_buckets=64)
        
        def refresh_display():
            """刷新图像显示"""
            nonlocal term_w, term_h
            print(clear_screen)
            
            # 应用当前调整参数
            adjusted_image = adjust_frame(original_image.copy(), current_brightness, current_contrast)
            
            # 显示状态信息
            status_parts = [
                f"文件: {image_filename}",
                f"缩放: {current_scale:.1f}x",
                f"亮度: {current_brightness}",
                f"对比度: {current_contrast:.1f}",
                f"模式: {'块字符' if current_use_block_char else 'ASCII'}",
                f"颜色: {'开' if current_use_color else '关'}"
            ]
            
            status_lines = format_progress_line(status_parts, term_w)

            # 生成ASCII艺术
            ascii_art = frame_to_ascii(adjusted_image, brightness_map, term_w, term_h, 
                                      current_scale, use_block_char=current_use_block_char, 
                                      use_color=current_use_color, is_image=1)
            
            print(f"{move_cursor_top}{ascii_art}")
            
            for i, line in enumerate(status_lines):
                print(f"{clear_line}{line}")
            
            # 显示操作提示
            print(f"{clear_line}按 'i' 查看帮助 | 按 'q' 退出 | 按 's' 保存截图")
        
        refresh_display()
        
        # 主循环
        running = True
        while running:
            # 检查终端尺寸变化
            current_time = time.time()
            if current_time - last_terminal_check >= terminal_check_interval:
                new_term_w, new_term_h = get_terminal_size()
                if (new_term_w, new_term_h) != last_term_size:
                    term_w, term_h = new_term_w, new_term_h
                    last_term_size = (term_w, term_h)
                    refresh_display()
                last_terminal_check = current_time
            
            # 处理按键
            key = get_key_press()
            if key == 'quit':
                running = False
            elif key == 'info':
                # 显示帮助信息
                print(clear_screen)
                show_help_info()
                wait_for_any_key()
                refresh_display()
            elif key == 'toggle_display':
                current_use_block_char = not current_use_block_char
                refresh_display()
            elif key == 'toggle_color':
                current_use_color = not current_use_color
                refresh_display()
            elif key == 'screenshot':
                # 保存截图
                adjusted_image = adjust_frame(original_image.copy(), current_brightness, current_contrast)
                img_path, txt_path = save_screenshot(adjusted_image, os.path.splitext(image_filename)[0], 
                                              0, current_use_color, current_use_block_char)
                print(f"\033[s\033[2;1H截图已保存: {img_path} 和 {txt_path}\033[u")
                time.sleep(2)
                refresh_display()
            elif key == 'brightness_up':
                current_brightness = min(100, current_brightness + 10)
                refresh_display()
            elif key == 'brightness_down':
                current_brightness = max(-100, current_brightness - 10)
                refresh_display()
            elif key == 'contrast_up':
                current_contrast = min(3.0, current_contrast + 0.1)
                refresh_display()
            elif key == 'contrast_down':
                current_contrast = max(0.1, current_contrast - 0.1)
                refresh_display()
            elif key == 'fast':  # 缩放放大
                current_scale = min(1.0, current_scale + 0.1)
                refresh_display()
            elif key == 'slow':  # 缩放缩小
                current_scale = max(0.1, current_scale - 0.1)
                refresh_display()
            elif key and key.startswith('speed_'):
                # 快速缩放设置
                speed_value = float(key.split('_')[1])
                current_scale = max(0.1, min(1.0, speed_value))
                refresh_display()
            
            time.sleep(0.01)
        
        print(show_cursor)
        print("\033[0m")
        
    except Exception as e:
        print(f"处理图片时发生错误: {str(e)}")
        print(show_cursor)

def init_log_file(video_path, fps, total_frames, width, height, skip_threshold, skip_frames):
    """初始化日志文件"""
    log_dir = "video_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    log_file = os.path.join(log_dir, f"{video_name}_{timestamp}.log")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"视频日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"视频文件: {video_path}\n")
        f.write(f"视频分辨率: {width}x{height}\n")
        f.write(f"视频帧率: {fps:.2f} FPS\n")
        f.write(f"总帧数: {total_frames}\n")
        f.write(f"视频时长: {format_duration(total_frames/fps)}\n")
        f.write(f"跳帧阈值: {skip_threshold} FPS\n")
        f.write(f"跳帧数量: {skip_frames} 帧\n")
        f.write("时间戳,帧处理速度(帧/秒),输出速度(帧/秒),当前帧率(帧/秒),已处理帧数,已跳过帧数,进度(%)\n")
    
    return log_file

def write_log(log_file, timestamp, frame_processing_speed, output_speed, current_fps, 
             frame_count, skipped_frames, total_frames):
    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp},{frame_processing_speed:.2f},{output_speed:.2f},{current_fps:.2f},")
        f.write(f"{frame_count},{skipped_frames},{progress:.2f}\n")

def get_key_press():
    if platform.system() == "Windows":
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\xe0':
                key = msvcrt.getch()
                if key == b'H': return 'up'
                elif key == b'P': return 'down'
                elif key == b'M': return 'right'
                elif key == b'K': return 'left'
            elif key == b' ': return 'space'
            elif key == b'i' or key == b'I': return 'info'
            elif key == b',': return 'slow'
            elif key == b'.': return 'fast'
            elif key == b'f' or key == b'F': return 'toggle_display'
            elif key == b'c' or key == b'C': return 'toggle_color'
            elif key == b'q' or key == b'Q': return 'quit'
            elif key == b'r' or key == b'R': return 'reset_speed'
            elif key == b's' or key == b'S': return 'screenshot'
            elif key == b'o' or key == b'O': return 'toggle_verbose'
            elif key == b'p' or key == b'P': return 'toggle_progress'
            elif key == b'0': return 'speed_0.25'
            elif key == b'1': return 'speed_0.5'
            elif key == b'2': return 'speed_1.0'
            elif key == b'3': return 'speed_1.5'
            elif key == b'4': return 'speed_2.0'
            elif key == b'5': return 'speed_4.0'
            elif key == b'd': return 'brightness_up'
            elif key == b'a': return 'brightness_down'
            elif key == b'[': return 'contrast_down'
            elif key == b']': return 'contrast_up'
            elif key == b'l' or key == b'L': return 'toggle_loop'
            return None
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == '\x1b':
                    key += sys.stdin.read(2)
                    if key == '\x1b[A': return 'up'
                    elif key == '\x1b[B': return 'down'
                    elif key == '\x1b[C': return 'right'
                    elif key == '\x1b[D': return 'left'
                elif key == ' ': return 'space'
                elif key.lower() == 'i': return 'info'
                elif key == ',': return 'slow'
                elif key == '.': return 'fast'
                elif key.lower() == 'f': return 'toggle_display'
                elif key.lower() == 'c': return 'toggle_color'
                elif key.lower() == 'q': return 'quit'
                elif key.lower() == 'r': return 'reset_speed'
                elif key.lower() == 's': return 'screenshot'
                elif key.lower() == 'o': return 'toggle_verbose'
                elif key.lower() == 'p': return 'toggle_progress'
                elif key == '0': return 'speed_0.25'
                elif key == '1': return 'speed_0.5'
                elif key == '2': return 'speed_1.0'
                elif key == '3': return 'speed_1.5'
                elif key == '4': return 'speed_2.0'
                elif key == '5': return 'speed_4.0'
                elif key == 'd': return 'brightness_up'
                elif key == 'a': return 'brightness_down'
                elif key == '[': return 'contrast_down'
                elif key == ']': return 'contrast_up'
                elif key.lower() == 'l': return 'toggle_loop'
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return None

def wait_for_any_key():
    if platform.system() == "Windows":
        msvcrt.getch()
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def show_help_info():
    print("\033[2J\033[H")
    print("=" * 60)
    print("                操作帮助")
    print("=" * 60)
    print("  空格键: 暂停/继续播放")
    print("  左方向键: 快退5秒")
    print("  右方向键: 快进5秒")
    print("  ,键: 降低播放速度")
    print("  .键: 提高播放速度")
    print("  0-5键: 快速设置速度 (0=0.25x, 1=0.5x, 2=1.0x, 3=1.5x, 4=2.0x, 5=4.0x)")
    print("  r键: 重置播放速度为1.0x")
    print("  i键: 显示/隐藏本帮助信息")
    print("  F键: 切换显示模式 (ASCII字符 / 块字符)")
    print("  C键: 切换颜色显示")
    print("  S键: 截图保存当前帧")
    print("  O键: 切换详细状态显示")
    print("  P键: 切换进度条显示")
    print("  D键: 增加亮度")
    print("  A键: 降低亮度")
    print("  [键: 降低对比度")
    print("  ]键: 增加对比度")
    print("  L键: 切换循环播放")
    print("  Q键: 退出播放")
    print("  冷知识：本程序支持命令行参数，针对照片输出可能需要通过命令行参数进行控制")
    print("\n按任意键继续播放...")
    print("=" * 60)

def format_progress_line(parts, term_w):
    lines = []
    current_line = []
    current_length = 0
    total_parts = len(parts)
    
    for i in range(total_parts):
        part = parts[i]
        part_len = len(part) + 5  # 加上分隔符的估计长度
        
        next_part_len = 0
        if i + 1 < total_parts:
            next_part = parts[i + 1]
            next_part_len = len(next_part) + 5
        
        # 判断条件：当前行有内容，且当前长度+当前部分+下一个部分会超过终端宽度时换行
        if current_line and (current_length + part_len + next_part_len > term_w):
            lines.append(" | ".join(current_line))
            current_line = []
            current_length = 0
        
        current_line.append(part)
        current_length += part_len
    
    if current_line:
        lines.append(" | ".join(current_line))
    
    return [f"\033[0m{line}" for line in lines]

def create_progress_bar(progress, width=20):
    filled = int(progress * width / 100)
    empty = width - filled
    return f"[{'=' * filled}{' ' * empty}] {progress:.1f}%"

def save_screenshot(frame, video_name, frame_count, use_color, use_block_char):
    """保存当前帧为图片"""
    screenshot_dir = "screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{video_name}_{timestamp}_f{frame_count}.png"
    filepath = os.path.join(screenshot_dir, filename) 
    # 原始帧
    cv2.imwrite(filepath, frame)
    # ASCII
    term_w, term_h = get_terminal_size()
    video_display_height = max(1, term_h - 3)
    brightness_map = create_brightness_map(num_buckets=64)
    ascii_art = frame_to_ascii(frame, brightness_map, term_w, video_display_height, 
                              1.0, use_block_char=use_block_char, use_color=use_color, is_image=0)
    
    txt_filename = f"{video_name}_{timestamp}_f{frame_count}.txt"
    txt_filepath = os.path.join(screenshot_dir, txt_filename)
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(ascii_art)
    
    return filepath, txt_filepath

def play_video_as_ascii(video_path, scale_factor=1.0, skip_threshold=1.0, skip_frames=2, use_color=True):
    if platform.system() == "Windows":
        video_path = video_path.replace('\\', '/')
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_path.encode('gbk'))
                if not cap.isOpened():
                    print(f"无法打开视频文件: {video_path}")
                    return
        except Exception as e:
            print(f"打开视频时发生错误: {str(e)}")
            return
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    
    base_frame_delay = 1.0 / video_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0
    
    log_file = init_log_file(video_path, video_fps, total_frames, width, height, skip_threshold, skip_frames)
    print(f"日志文件已创建: {log_file}")
    
    brightness_map = create_brightness_map(num_buckets=64)
    term_w, term_h = get_terminal_size()
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    
    # 播放控制变量
    playback_speed = 1.0
    original_speed = playback_speed
    speed_step = 0.5
    min_speed = 0.25
    max_speed = 8.0
    seek_seconds = 5
    use_block_char = False
    verbose_status = False
    show_progress_bar = True
    brightness = 0  # 亮度调整值 (-100 到 100)
    contrast = 1.0  # 对比度调整值 (0.1 到 3.0)
    loop_play = False  # 循环播放
    
    # ANSI
    clear_screen = "\033[2J"
    move_cursor_top = "\033[H"
    clear_line = "\033[K"
    hide_cursor = "\033[?25l"
    show_cursor = "\033[?25h"
    
    print(hide_cursor)
    print(clear_screen)
    
    # 状态变量
    paused = False
    last_terminal_check = time.time()
    terminal_check_interval = 0.5
    last_term_size = (term_w, term_h)
    paused_frame = None
    start_time = time.time()
    prev_time = start_time
    last_log_time = start_time
    frame_count = 0
    total_skipped_frames = 0
    frames_since_last_log = 0
    skipped_since_last_log = 0
    current_fps = 0
    last_frame_processed = False
    fps_history = deque(maxlen=10)  # 存储最近10帧的FPS
    last_term_progress_h=1
    
    # 帧缓存
    frame_cache = {}
    last_cached_frame = None
    
    try:
        while frame_count < total_frames or loop_play:
            # 循环播放处理
            if frame_count >= total_frames and loop_play:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                total_skipped_frames = 0
                start_time = time.time()
                prev_time = start_time
                last_log_time = start_time
                print(clear_screen)
            
            term_w, term_h = get_terminal_size()
            video_display_height = max(1, term_h-last_term_progress_h+2)
            
            key = get_key_press()
            if key == 'space':
                paused = not paused
                if paused and 'frame' in locals():
                    paused_frame = frame.copy()  # 保存当前帧用于暂停时显示
            elif key == 'quit':
                print("\n用户退出播放")
                break
            elif key == 'info':
                current_paused_state = paused
                paused = True
                show_help_info()
                wait_for_any_key()
                paused = current_paused_state
                print(clear_screen)
            elif key == 'toggle_display':
                use_block_char = not use_block_char
                frame_cache = {}
            elif key == 'toggle_color':
                use_color = not use_color
                frame_cache = {}
            elif key == 'toggle_verbose':
                verbose_status = not verbose_status
            elif key == 'toggle_progress':
                show_progress_bar = not show_progress_bar
            elif key == 'toggle_loop':
                loop_play = not loop_play
            elif key == 'right' and not paused:
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_pos = min(total_frames - 1, current_pos + seek_seconds * video_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                frame_count = int(new_pos)
            elif key == 'left' and not paused:
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_pos = max(0, current_pos - seek_seconds * video_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                frame_count = int(new_pos)
            elif key == 'fast' and not paused:
                playback_speed = min(max_speed, playback_speed + speed_step)
            elif key == 'slow' and not paused:
                playback_speed = max(min_speed, playback_speed - speed_step)
            elif key == 'reset_speed' and not paused:
                playback_speed = original_speed
            elif key == 'screenshot' and not paused and 'frame' in locals():
                img_path, txt_path = save_screenshot(frame, video_name, frame_count, use_color, use_block_char)
                print(f"\033[0m\033[s\033[2;1H截图已保存: {img_path} 和 {txt_path}\033[u")
                time.sleep(2)
            elif key and key.startswith('speed_'):
                speed_value = float(key.split('_')[1])
                playback_speed = max(min_speed, min(max_speed, speed_value))
            elif key == 'brightness_up':
                brightness = min(100, brightness + 10)
                frame_cache = {}
            elif key == 'brightness_down':
                brightness = max(-100, brightness - 10)
                frame_cache = {}
            elif key == 'contrast_up':
                contrast = min(3.0, contrast + 0.1)
                frame_cache = {}
            elif key == 'contrast_down':
                contrast = max(0.1, contrast - 0.1)
                frame_cache = {}
            
            if paused:
                if paused_frame is not None:
                    # 显示暂停帧
                    ascii_frame = frame_to_ascii(paused_frame, brightness_map, term_w, video_display_height, 
                                               scale_factor, use_block_char=use_block_char, use_color=use_color, is_image=0)
                    print(f"{move_cursor_top}{ascii_frame}")
                
                elapsed_time = time.time() - start_time
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                current_pos_seconds = frame_count / video_fps if video_fps > 0 else 0
                
                status_parts = [
                    f"文件: {video_filename}",
                    f"速度: {playback_speed:.1f}x",
                    f"帧率: {avg_fps:.1f}/{video_fps:.1f}FPS",
                    f"位置: {format_duration(current_pos_seconds)}/{format_duration(video_duration)}",
                    f"循环: {'开' if loop_play else '关'}",
                    "已暂停"
                ]
                
                if verbose_status:
                    status_parts.extend([
                        f"尺寸: {width}x{height}",
                        f"进度: {progress:.1f}% ({frame_count}/{total_frames})",
                        f"已用: {format_duration(elapsed_time)}",
                        f"剩余: {format_duration(remaining_time)}",
                        f"跳帧: {total_skipped_frames}"
                        f"模式: {'块字符' if use_block_char else 'ASCII'}",
                        f"颜色: {'开' if use_color else '关'}",
                        f"亮度: {brightness}",
                        f"对比度: {contrast:.1f}",
                        f"终端大小: {term_w}x{term_h}",
                    ])
                
                if show_progress_bar and term_w > 50:
                    progress_bar = create_progress_bar(progress, min(50, term_w - 20))
                    status_parts.append(progress_bar)
                
                status_lines = format_progress_line(status_parts, term_w)
                last_term_progress_h = len(status_lines)
                for i, line in enumerate(status_lines):
                    print(f"\033[{term_h - len(status_lines) + i};0H{clear_line}{line}")
                
                current_time = time.time()
                if current_time - last_terminal_check >= terminal_check_interval:
                    new_term_w, new_term_h = get_terminal_size()
                    if (new_term_w, new_term_h) != last_term_size:
                        last_term_size = (new_term_w, new_term_h)
                        print(clear_screen)
                        if paused_frame is not None:
                            ascii_frame = frame_to_ascii(paused_frame, brightness_map, new_term_w, new_term_h-3, 
                                                       scale_factor, use_block_char=use_block_char, use_color=use_color, is_image=0)
                            print(f"{move_cursor_top}{ascii_frame}")
                    last_terminal_check = current_time
                
                time.sleep(0.05)
                continue
            
            current_time = time.time()
            if current_time - last_terminal_check >= terminal_check_interval:
                new_term_w, new_term_h = get_terminal_size()
                if (new_term_w, new_term_h) != last_term_size:
                    last_term_size = (new_term_w, new_term_h)
                    print(clear_screen)
                    frame_cache = {}
                last_terminal_check = current_time
            
            frame_delay = base_frame_delay / playback_speed
            
            ret, frame = cap.read()
            if not ret:
                if loop_play:
                    continue
                else:
                    last_frame_processed = True
                    break
            
            frame = adjust_frame(frame, brightness, contrast)
            frame_process_start = time.time()
            
            cache_key = f"{frame_count}_{term_w}_{term_h}_{int(use_block_char)}_{int(use_color)}"
            if cache_key in frame_cache:
                ascii_frame = frame_cache[cache_key]
            else:
                ascii_frame = frame_to_ascii(frame, brightness_map, term_w, video_display_height, 
                                           scale_factor, use_block_char=use_block_char, use_color=use_color, is_image=0)
                if len(frame_cache) > 30:
                    oldest_key = next(iter(frame_cache.keys()))
                    del frame_cache[oldest_key]
                frame_cache[cache_key] = ascii_frame
                last_cached_frame = cache_key
            
            frame_process_time = time.time() - frame_process_start
            
            print(f"{move_cursor_top}{ascii_frame}")
            current_time = time.time()
            elapsed = current_time - prev_time
            sleep_time = max(0, frame_delay - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            output_time = time.time() - prev_time
            output_speed = 1.0 / output_time if output_time > 0 else 0
            current_fps = output_speed * playback_speed
            fps_history.append(current_fps)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else current_fps
            prev_time = time.time()
            
            frame_count += 1
            frames_since_last_log += 1
            
            # 倍速播放时跳帧
            if playback_speed > 1.0 and current_fps < video_fps * playback_speed:
                frames_to_skip = min(int(playback_speed), skip_frames, total_frames - frame_count - 1)
                if frames_to_skip > 0:
                    for _ in range(frames_to_skip):
                        cap.read()
                    frame_count += frames_to_skip
                    total_skipped_frames += frames_to_skip
                    skipped_since_last_log += frames_to_skip
            
            # 每5秒记录日志
            if current_time - last_log_time >= 5.0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                avg_frame_processing_speed = frames_since_last_log / (current_time - last_log_time)
                avg_output_speed = frames_since_last_log / (current_time - last_log_time)
                avg_current_fps = avg_output_speed * playback_speed
                
                write_log(
                    log_file, timestamp, 
                    avg_frame_processing_speed, 
                    avg_output_speed, 
                    avg_current_fps, 
                    frame_count, 
                    total_skipped_frames,
                    total_frames
                )
                
                last_log_time = current_time
                frames_since_last_log = 0
                skipped_since_last_log = 0
            
            elapsed_time = time.time() - start_time
            remaining_time = (total_frames - frame_count) / video_fps / playback_speed if video_fps > 0 else 0
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            current_pos_seconds = frame_count / video_fps if video_fps > 0 else 0
            
            status_parts = [
                f"文件: {video_filename}",
                f"速度: {playback_speed:.1f}x",
                f"帧率: {avg_fps:.1f}/{video_fps:.1f}FPS",
                f"位置: {format_duration(current_pos_seconds)}/{format_duration(video_duration)}",
                f"循环: {'开' if loop_play else '关'}"
            ]
            
            if verbose_status:
                status_parts.extend([
                    f"尺寸: {width}x{height}",
                    f"进度: {progress:.1f}% ({frame_count}/{total_frames})",
                    f"已用: {format_duration(elapsed_time)}",
                    f"剩余: {format_duration(remaining_time)}",
                    f"跳帧: {total_skipped_frames}"
                    f"模式: {'块字符' if use_block_char else 'ASCII'}",
                    f"颜色: {'开' if use_color else '关'}",
                    f"亮度: {brightness}",
                    f"对比度: {contrast:.1f}",
                    f"终端大小: {term_w}x{term_h}",
                ])
            
            if show_progress_bar and term_w > 50:
                progress_bar = create_progress_bar(progress, min(50, term_w - 20))
                status_parts.append(progress_bar)
            
            status_lines = format_progress_line(status_parts, term_w)
            last_term_progress_h = len(status_lines)
            for i, line in enumerate(status_lines):
                print(f"\033[{term_h - len(status_lines) + i};0H{clear_line}{line}")
        
        if not last_frame_processed and not loop_play:
            # 显示最后一帧
            ascii_frame = frame_to_ascii(frame, brightness_map, term_w, video_display_height, 
                                       scale_factor, use_block_char=use_block_char, use_color=use_color, is_image=0)
            print(f"{move_cursor_top}{ascii_frame}")
            
            # 显示完成信息
            elapsed_time = time.time() - start_time
            status_parts = [
                f"视频播放完成!",
                f"总时间: {format_duration(elapsed_time)}",
                f"总帧数: {frame_count}",
                f"跳帧数: {total_skipped_frames}",
                f"平均帧率: {frame_count/elapsed_time:.1f}FPS",
                f"按任意键退出..."
            ]
            
            status_lines = format_progress_line(status_parts, term_w)
            for i, line in enumerate(status_lines):
                print(f"\033[{term_h - len(status_lines) + i};0H{clear_line}{line}")
            
            wait_for_any_key()
    
    except KeyboardInterrupt:
        print("\n播放被中断")
    except Exception as e:
        print(f"\n播放时发生错误: {str(e)}")
    finally:
        cap.release()
        print(show_cursor)
        print("\033[0m")
        print(f"\n日志已保存至: {log_file}")
        print(f"总处理帧数: {frame_count}, 总跳过帧数: {total_skipped_frames}")
        print(f"总耗时: {format_duration(time.time() - start_time)}")

def is_image_file(file_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.jfif')
    return file_path.lower().endswith(image_extensions)

def is_video_file(file_path):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg', '.m4v', '.3gp')
    return file_path.lower().endswith(video_extensions)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(prog="ascii.py", description='用途:将图片或视频转换为文本输出 本工具由Svvcvv@github制作 版本v0.1.6', 
                                    epilog="示例: python ascii.py video.mp4 --scale 1.0 --block //关于播放时的操作帮助在播放时按i显示")
    parser.add_argument('file_path', help='图片或视频文件路径')
    parser.add_argument('--scale', type=float, default=1.0, 
                        help='缩放因子 (0.1-1.0, 默认: 1.0)')
    parser.add_argument('--block', action='store_true', 
                        help='使用块字符模式 (默认: ASCII字符)')
    parser.add_argument('--no-color', action='store_true', 
                        help='禁用彩色输出 (默认: 启用颜色)')
    parser.add_argument('--skip-threshold', type=float, default=1.0, 
                        help='跳帧阈值FPS (默认: 1.0)')
    parser.add_argument('--skip-frames', type=int, default=10, 
                        help='跳帧数量 (默认: 10)')
    parser.add_argument('--aspect-ratio', type=float, default=2.0, 
                        help='字符宽高比 (默认: 2.0)')
    parser.add_argument('--brightness', type=int, default=0, 
                        help='初始亮度调整 (-100 到 100, 默认: 0)')
    parser.add_argument('--contrast', type=float, default=1.0, 
                        help='初始对比度调整 (0.1 到 3.0, 默认: 1.0)')
    parser.add_argument('--loop', action='store_true', 
                        help='启用循环播放')
    
    args = parser.parse_args()
    
    file_path = args.file_path
    scale_factor = max(0.1, min(1.0, args.scale))
    brightness = max(-100, min(100, args.brightness))
    contrast = max(0.1, min(3.0, args.contrast))
    
    if platform.system() == "Windows":
        file_path = file_path.strip('"\'')
        file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        sys.exit(1)
    
    if is_image_file(file_path):
        interactive_image_viewer(file_path, scale_factor, 
                      use_block_char=args.block, 
                      use_color=not args.no_color,
                      brightness=brightness,
                      contrast=contrast)
    elif is_video_file(file_path):
        play_video_as_ascii(
            file_path, 
            scale_factor, 
            skip_threshold=args.skip_threshold, 
            skip_frames=args.skip_frames,
            use_color=not args.no_color
        )
    else:
        print(f"不支持的文件格式: {file_path}")

        sys.exit(1)
