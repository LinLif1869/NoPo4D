#!/usr/bin/env python
"""True GPU-interactive 4D Gaussian viewer for the current eight-camera data.

This is deliberately not an image-grid web viewer.  Every mouse update changes
the target c2w matrix and calls the gsplat decoder on the resident 4D Gaussian
scene.  The two low-overlap capture zones remain separate; switching anchors is
explicit, never hidden inside a supposedly continuous camera orbit.
"""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference import _axis_angle_rotation, estimate_camera_focus
from src.mutiwin.realtime_4d_demo import (
    Anchor,
    compose,
    configure_model,
    load_calibrations,
    reconstruct_window,
    render_local_anchor,
)


@dataclass
class ViewerState:
    anchor_index: int
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    distance: float = 1.0
    timestamp: float = 1.0 / 3.0
    playing: bool = False
    dragging: bool = False
    time_dragging: bool = False
    last_x: int = 0
    last_y: int = 0
    dirty: bool = True


def target_pose(scene, source_index: int, yaw_deg: float, pitch_deg: float, distance: float):
    camera_c2w = scene.ext[:, ::scene.num_frames]
    centre = estimate_camera_focus(camera_c2w[0])
    reference = camera_c2w[0, source_index]
    up = -camera_c2w[0, :, :3, 1].mean(0)
    yaw = _axis_angle_rotation(up, reference.new_tensor(np.deg2rad(yaw_deg)))

    pose = torch.eye(4, device=reference.device, dtype=reference.dtype)
    pose[:3, 3] = yaw @ (reference[:3, 3] - centre) + centre
    pose[:3, :3] = yaw @ reference[:3, :3]
    right = pose[:3, 0]
    pitch = _axis_angle_rotation(right, reference.new_tensor(np.deg2rad(pitch_deg)))
    pose[:3, 3] = pitch @ (pose[:3, 3] - centre) + centre
    pose[:3, :3] = pitch @ pose[:3, :3]
    pose[:3, 3] = centre + (pose[:3, 3] - centre) * float(distance)
    return pose


def render_state(windows, anchors, state: ViewerState, crop: float, background: float):
    anchor = anchors[state.anchor_index]
    window = windows[anchor.window]
    scene = window.scene
    pose = target_pose(
        scene, anchor.source_index, state.yaw_deg, state.pitch_deg, state.distance,
    )
    K = scene.intr[0, anchor.source_index * scene.num_frames]
    result = render_local_anchor(
        window, anchor.source_index, pose, K, state.timestamp,
    )
    frame, holes = compose([result], np.ones(1), crop, background)
    return frame, holes


def add_hud(frame, anchors, state: ViewerState, holes: float, fps: float):
    anchor = anchors[state.anchor_index]
    zone = "A" if anchor.window == 0 else "B"
    lines = [
        f"LIVE 4D GS | 8 input cams | zone {zone} | source cam{anchor.camera_id}",
        f"yaw {state.yaw_deg:+.2f}  pitch {state.pitch_deg:+.2f}  "
        f"zoom {state.distance:.2f}  time {state.timestamp:.3f}",
        f"GPU render {fps:5.1f} FPS | alpha holes {holes*100:.1f}%",
        "L-drag orbit | wheel zoom | R-drag time | Space play | 1-8 anchor | R reset | Q quit",
    ]
    for row, line in enumerate(lines):
        y = 25 + row * 23
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (245, 245, 245), 1, cv2.LINE_AA)
    return frame


def build_scenes(args, device):
    calibrations = load_calibrations(args.calibration_json)
    model = configure_model(device)
    windows = [
        reconstruct_window(model, "A", args.window_a, calibrations, [40, 41, 64, 65], device),
        reconstruct_window(model, "B", args.window_b, calibrations, [42, 66, 67, 43], device),
    ]
    # Key order is chosen for usability rather than azimuth sorting.  cam64 is
    # the validated best default source and occupies key 1.
    anchors = [
        Anchor(64, 0, 2), Anchor(65, 0, 3), Anchor(40, 0, 0), Anchor(41, 0, 1),
        Anchor(66, 1, 1), Anchor(67, 1, 2), Anchor(42, 1, 0), Anchor(43, 1, 3),
    ]
    return windows, anchors


def run_recorded_demo(args, windows, anchors):
    args.record.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.record), fps=args.fps, codec="libx264", quality=8, macro_block_size=2,
    )
    state = ViewerState(anchor_index=0)
    samples = []
    try:
        for index in range(args.frames):
            phase = index / max(args.frames - 1, 1)
            state.yaw_deg = args.yaw_limit * np.sin(2 * np.pi * phase)
            state.pitch_deg = 0.65 * args.pitch_limit * np.sin(4 * np.pi * phase)
            state.distance = 1.0 + 0.04 * np.sin(2 * np.pi * phase + np.pi / 2)
            state.timestamp = 0.5 - 0.5 * np.cos(2 * np.pi * phase)
            torch.cuda.synchronize()
            started = time.perf_counter()
            frame, holes = render_state(windows, anchors, state, args.crop, args.background)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            samples.append(elapsed)
            add_hud(frame, anchors, state, holes, 1.0 / elapsed)
            writer.append_data(frame)
    finally:
        writer.close()
    values = np.asarray(samples)
    print(f"[VIEWER] median={np.median(values)*1000:.2f} ms "
          f"({1/np.median(values):.1f} FPS), p95={np.quantile(values,.95)*1000:.2f} ms")
    print(f"[DONE] {args.record}")


def run_gui(args, windows, anchors):
    title = "NoPo4D true interactive 4D Gaussian viewer"
    state = ViewerState(anchor_index=0)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1008, 756)
    cv2.createTrackbar("4D time", title, round(state.timestamp * 1000), 1000, lambda _: None)

    def mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging, state.last_x, state.last_y = True, x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.time_dragging, state.last_x = True, x
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state.dragging = state.time_dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
            state.yaw_deg = float(np.clip(
                state.yaw_deg + (x - state.last_x) * 0.025,
                -args.yaw_limit, args.yaw_limit,
            ))
            state.pitch_deg = float(np.clip(
                state.pitch_deg - (y - state.last_y) * 0.020,
                -args.pitch_limit, args.pitch_limit,
            ))
            state.last_x, state.last_y, state.dirty = x, y, True
        elif event == cv2.EVENT_MOUSEMOVE and state.time_dragging:
            state.timestamp = float(np.clip(state.timestamp + (x - state.last_x) / 700, 0, 1))
            cv2.setTrackbarPos("4D time", title, round(state.timestamp * 1000))
            state.last_x, state.dirty = x, True
        elif event == cv2.EVENT_MOUSEWHEEL:
            direction = 1 if flags > 0 else -1
            state.distance = float(np.clip(state.distance * (0.97 ** direction), 0.88, 1.12))
            state.dirty = True

    cv2.setMouseCallback(title, mouse)
    last_time = time.perf_counter()
    fps = 0.0
    frame = None
    while True:
        now = time.perf_counter()
        slider_time = cv2.getTrackbarPos("4D time", title) / 1000
        if abs(slider_time - state.timestamp) > 1e-6:
            state.timestamp, state.dirty = slider_time, True
        if state.playing:
            state.timestamp = (state.timestamp + (now - last_time) * 0.25) % 1.0
            cv2.setTrackbarPos("4D time", title, round(state.timestamp * 1000))
            state.dirty = True
        last_time = now
        if state.dirty or frame is None:
            torch.cuda.synchronize()
            started = time.perf_counter()
            frame, holes = render_state(windows, anchors, state, args.crop, args.background)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            fps = 1.0 / elapsed
            add_hud(frame, anchors, state, holes, fps)
            state.dirty = False
        cv2.imshow(title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if ord("1") <= key <= ord("8"):
            state.anchor_index = key - ord("1")
            state.yaw_deg = state.pitch_deg = 0.0
            state.distance, state.dirty = 1.0, True
        elif key == ord(" "):
            state.playing = not state.playing
        elif key == ord("r"):
            state.yaw_deg = state.pitch_deg = 0.0
            state.distance, state.timestamp, state.dirty = 1.0, 1.0 / 3.0, True
            cv2.setTrackbarPos("4D time", title, round(state.timestamp * 1000))
        elif key == ord("s") and frame is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.screenshot), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"[SAVED] {args.screenshot}")
    cv2.destroyAllWindows()


def remote_viewer_html(args) -> str:
    """Thin remote control client; all pixels still come from live gsplat."""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NoPo4D Live 4D Gaussian Viewer</title>
<style>
  *{{box-sizing:border-box}} body{{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif}}
  main{{max-width:1120px;margin:auto;padding:14px}} h2{{margin:0 0 8px}}
  #view{{width:100%;aspect-ratio:4/3;object-fit:contain;background:#292929;cursor:grab;user-select:none}}
  #view.drag{{cursor:grabbing}} .bar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:9px 0}}
  button{{background:#333;color:#eee;border:1px solid #666;padding:6px 11px;border-radius:5px}}
  button.active{{background:#1769aa}} input[type=range]{{width:330px}}
  .note{{color:#bbb;line-height:1.55}} #status{{font-family:monospace;color:#8fdaff}}
</style></head><body><main>
<h2>NoPo4D 实时 4D Gaussian（服务器 GPU 渲染）</h2>
<div id="status">连接中…</div>
<img id="view" draggable="false" alt="live gsplat frame">
<div class="bar"><span>物理锚点：</span><span id="anchors"></span>
  <button id="reset">复位</button><button id="play">播放时间</button></div>
<div class="bar"><label>4D time <input id="time" type="range" min="0" max="1" step="0.001" value="0.333"></label>
  <span id="timeValue">0.333</span></div>
<div class="note">左键拖动：连续旋转真实目标相机；滚轮：缩放；右键横拖：时间；数字 1–8：显式切换锚点。<br>
当前网页只传控制量和显示服务器即时 JPEG；没有预渲染视角图片。连续拖动被限制在 ±{args.yaw_limit:g}° 可靠区域，因而不会自动跨低重叠缺口跳源。</div>
<script>
const img=document.getElementById('view'), status=document.getElementById('status');
const slider=document.getElementById('time'), timeValue=document.getElementById('timeValue');
const cams=[64,65,40,41,66,67,42,43];
let s={{anchor:0,yaw:0,pitch:0,distance:1,time:.333}}, busy=false, dirty=true, playing=false;
let drag=-1,lastX=0,lastY=0,lastTick=performance.now(),lastUrl=null;
const anchorBox=document.getElementById('anchors');
cams.forEach((cam,i)=>{{let b=document.createElement('button');b.textContent=(i+1)+': cam'+cam;
 b.onclick=()=>{{s.anchor=i;s.yaw=s.pitch=0;s.distance=1;dirty=true;paintButtons()}};b.id='a'+i;anchorBox.appendChild(b)}});
function paintButtons(){{cams.forEach((_,i)=>document.getElementById('a'+i).classList.toggle('active',i===s.anchor))}} paintButtons();
function clamp(v,a,b){{return Math.max(a,Math.min(b,v))}}
img.oncontextmenu=e=>e.preventDefault();
img.onmousedown=e=>{{drag=e.button;lastX=e.clientX;lastY=e.clientY;img.classList.add('drag')}};
window.onmouseup=()=>{{drag=-1;img.classList.remove('drag')}};
window.onmousemove=e=>{{if(drag===0){{s.yaw=clamp(s.yaw+(e.clientX-lastX)*.025,-{args.yaw_limit},{args.yaw_limit});
 s.pitch=clamp(s.pitch-(e.clientY-lastY)*.020,-{args.pitch_limit},{args.pitch_limit});dirty=true}}
 else if(drag===2){{s.time=clamp(s.time+(e.clientX-lastX)/700,0,1);slider.value=s.time;timeValue.textContent=s.time.toFixed(3);dirty=true}}
 lastX=e.clientX;lastY=e.clientY}};
img.onwheel=e=>{{e.preventDefault();s.distance=clamp(s.distance*Math.pow(.97,e.deltaY<0?1:-1),.88,1.12);dirty=true}};
slider.oninput=()=>{{s.time=+slider.value;timeValue.textContent=s.time.toFixed(3);dirty=true}};
document.getElementById('reset').onclick=()=>{{s.yaw=s.pitch=0;s.distance=1;s.time=.333;slider.value=s.time;dirty=true}};
document.getElementById('play').onclick=e=>{{playing=!playing;e.target.textContent=playing?'暂停时间':'播放时间'}};
window.onkeydown=e=>{{if(e.key>='1'&&e.key<='8')document.getElementById('a'+(+e.key-1)).click(); if(e.key===' '){{e.preventDefault();document.getElementById('play').click()}}}};
async function tick(now){{let dt=(now-lastTick)/1000;lastTick=now;if(playing){{s.time=(s.time+dt*.25)%1;slider.value=s.time;timeValue.textContent=s.time.toFixed(3);dirty=true}}
 if(dirty&&!busy){{dirty=false;busy=true;let q=new URLSearchParams(s);let t=performance.now();
  try{{let r=await fetch('/frame?'+q,{{cache:'no-store'}});if(!r.ok)throw Error(await r.text());let blob=await r.blob();let u=URL.createObjectURL(blob);
   img.onload=()=>{{if(lastUrl)URL.revokeObjectURL(lastUrl);lastUrl=u}};img.src=u;
   status.textContent=`live ${{r.headers.get('X-Render-Ms')}} ms | ${{r.headers.get('X-Render-Fps')}} FPS | alpha holes ${{r.headers.get('X-Alpha-Holes')}}% | round trip ${{(performance.now()-t).toFixed(1)}} ms`;
  }}catch(err){{status.textContent='错误：'+err}}finally{{busy=false;if(playing)dirty=true}}}}
 requestAnimationFrame(tick)}} requestAnimationFrame(tick);
</script></main></body></html>"""


def run_server(args, windows, anchors):
    render_lock = threading.Lock()
    html = remote_viewer_html(args).encode("utf-8")

    # Compile/lazily load gsplat before accepting the first browser request.
    # Keeping this on the server thread also avoids reporting the one-time JIT
    # cost as if it were interactive steady-state latency.
    warm_state = ViewerState(anchor_index=0)
    for _ in range(2):
        render_state(windows, anchors, warm_state, args.crop, args.background)
    torch.cuda.synchronize()
    print("[SERVER] gsplat warm-up complete.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            if args.http_log:
                super().log_message(fmt, *values)

        def send_bytes(self, payload: bytes, content_type: str, status=HTTPStatus.OK, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self.send_bytes(html, "text/html; charset=utf-8")
            if parsed.path == "/health":
                return self.send_bytes(b"ok", "text/plain")
            if parsed.path != "/frame":
                return self.send_bytes(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            try:
                query = parse_qs(parsed.query)
                value = lambda key, default: float(query.get(key, [default])[0])
                anchor_index = int(np.clip(value("anchor", 0), 0, len(anchors) - 1))
                state = ViewerState(
                    anchor_index=anchor_index,
                    yaw_deg=float(np.clip(value("yaw", 0), -args.yaw_limit, args.yaw_limit)),
                    pitch_deg=float(np.clip(value("pitch", 0), -args.pitch_limit, args.pitch_limit)),
                    distance=float(np.clip(value("distance", 1), 0.88, 1.12)),
                    timestamp=float(np.clip(value("time", 1 / 3), 0, 1)),
                )
                with render_lock:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    frame, holes = render_state(
                        windows, anchors, state, args.crop, args.background,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                add_hud(frame, anchors, state, holes, 1 / elapsed)
                ok, encoded = cv2.imencode(
                    ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                return self.send_bytes(
                    encoded.tobytes(), "image/jpeg", headers={
                        "X-Render-Ms": f"{elapsed * 1000:.2f}",
                        "X-Render-Fps": f"{1 / elapsed:.1f}",
                        "X-Alpha-Holes": f"{holes * 100:.1f}",
                    },
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}".encode("utf-8")
                return self.send_bytes(
                    message, "text/plain; charset=utf-8", HTTPStatus.BAD_REQUEST,
                )

    # One outstanding request is enough for the browser client.  A persistent
    # single server thread is materially faster for CUDA than creating a fresh
    # worker thread (and CUDA thread-local state) for every rendered frame.
    server = HTTPServer((args.host, args.port), Handler)
    print(f"[SERVER] Live gsplat viewer: http://{args.host}:{args.port}")
    if args.host in ("127.0.0.1", "localhost"):
        print(
            f"[SERVER] Remote machine: ssh -L {args.port}:127.0.0.1:{args.port} "
            "yang@lyg0168, then open the same URL locally."
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] stopped")
    finally:
        server.server_close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_a", type=Path, default=ROOT / "assets/nas_cam40_64_65_41_t118_127_s3")
    parser.add_argument("--window_b", type=Path, default=ROOT / "assets/nas_window_b_42_66_67_43_t118_127_s3")
    parser.add_argument("--calibration_json", type=Path, default=ROOT / "src/calibration_results-19-11-12_ByteDT.json")
    parser.add_argument("--yaw_limit", type=float, default=4.0,
                        help="Safe continuous yaw on either side of the locked source camera.")
    parser.add_argument("--pitch_limit", type=float, default=1.5)
    parser.add_argument("--crop", type=float, default=0.08)
    parser.add_argument("--background", type=float, default=0.18)
    parser.add_argument("--record", type=Path, default=None,
                        help="Headless validation: record a real gsplat mouse-like orbit instead of opening GUI.")
    parser.add_argument("--server", action="store_true",
                        help="Serve the true GPU renderer to a remote browser; no X11 required.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--http_log", action="store_true")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--screenshot", type=Path, default=ROOT / "output_nas_true_interactive_4d/screenshot.png")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/gsplat is required")
    windows, anchors = build_scenes(args, torch.device("cuda"))
    if args.record is not None:
        run_recorded_demo(args, windows, anchors)
    elif args.server or not os.environ.get("DISPLAY"):
        if not args.server:
            print("[INFO] DISPLAY is unavailable; automatically using remote server mode.")
        run_server(args, windows, anchors)
    else:
        run_gui(args, windows, anchors)


if __name__ == "__main__":
    main()
