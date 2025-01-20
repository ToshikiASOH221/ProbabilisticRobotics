import imageio
import os

video_path = "../figs/"
mp4_files = [f for f in os.listdir(video_path) if f.endswith(".mp4")]
for mp4_file in mp4_files:
    input_path = os.path.join(video_path, mp4_file)
    output_path = os.path.splitext(input_path)[0] + ".gif"
    
    # 動画を GIF に変換
    reader = imageio.get_reader(input_path, format="ffmpeg")
    frames = [frame for frame in reader]
    imageio.mimsave(output_path, frames, fps=20)