"""Streaming video and portable Gaussian exports; no training dependencies."""
from dataclasses import fields
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch


def cpu_fields(obj):
    return {f.name: (getattr(obj, f.name).detach().cpu() if isinstance(getattr(obj, f.name), torch.Tensor)
                     else getattr(obj, f.name)) for f in fields(obj)}


class VideoWriter:
    """Pipe RGB frames to ffmpeg. Pad odd dimensions for H.264 yuv420p."""
    def __init__(self, path, width, height, fps):
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('MP4 export requires ffmpeg with libx264 on PATH.')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.path.with_suffix('.ffmpeg.log'), 'w+')
        self.proc = subprocess.Popen([
            executable, '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
            '-i', '-', '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.path),
        ], stdin=subprocess.PIPE, stderr=self.log)

    def write(self, rgb):
        data = (rgb.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        self.proc.stdin.write(data.tobytes())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self.proc.wait()
        self.log.close()
        if code and exc_type is None:
            raise RuntimeError(f'ffmpeg failed ({code}); see {self.path.with_suffix(".ffmpeg.log")}')

