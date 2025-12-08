import os
import sys
import time
import threading
import queue
import traceback
import json
import base64
import io
import math
import uuid
import ctypes
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageChops
import pyautogui
from pynput import keyboard
import copy
from datetime import datetime
from collections import namedtuple

# --- 1. 依赖库检查与导入 ---
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("⚠️ 警告: 未安装 opencv-python。")

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SCIKIT_IMAGE = True
except ImportError:
    HAS_SCIKIT_IMAGE = False

# --- 音频库支持 ---
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
    import comtypes 
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# --- 2. 全局系统设置 ---
pyautogui.FAILSAFE = False

def get_scale_factor():
    try:
        if sys.platform.startswith('win'):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except:
                ctypes.windll.user32.SetProcessDPIAware()
        
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hdc = user32.GetDC(None)
        if hdc:
            dpi_x = gdi32.GetDeviceCaps(hdc, 88)
            dpi_y = gdi32.GetDeviceCaps(hdc, 90)
            user32.ReleaseDC(None, hdc)
            scale_x = dpi_x / 96.0
            scale_y = dpi_y / 96.0
            return max(0.5, min(3.0, scale_x)), max(0.5, min(3.0, scale_y))
    except Exception as e:
        print(f"获取缩放比例失败: {e}")
    
    try:
        log_w, log_h = pyautogui.size()
        if log_w == 0 or log_h == 0: return 1.0, 1.0
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        phy_w, phy_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        return max(0.5, min(3.0, phy_w / log_w)), max(0.5, min(3.0, phy_h / log_h))
    except:
        return 1.0, 1.0

SCALE_X, SCALE_Y = get_scale_factor()
SCALE_FACTOR = (SCALE_X + SCALE_Y) / 2.0
Box = namedtuple('Box', 'left top width height')

# --- 3. UI 配色方案 ---
COLORS = {
    'bg_app':     '#1e1e2e', 'bg_sidebar': '#181825', 'bg_canvas':  '#11111b',
    'bg_panel':   '#181825', 'bg_node':    '#313244', 'bg_header':  '#45475a',
    'bg_card':    '#313244', 'fg_title':   '#cdd6f4', 'fg_text':    '#a6adc8',
    'fg_sub':     '#6c7086', 'accent':     '#89b4fa', 'success':    '#a6e3a1',
    'danger':     '#f38ba8', 'warning':    '#f9e2af', 'control':    '#cba6f7',
    'sensor':     '#fab387', 'wire':       '#7f849c', 'wire_active':'#f9e2af',
    'wire_hl':    '#00ffff', 
    'socket':     '#f9e2af', 'socket_yes': '#a6e3a1', 'socket_no':  '#f38ba8',
    'grid':       '#262636', 'shadow':     '#000000', 'hover':      '#45475a',
    'active_border': '#a6e3a1', 'marker':     '#f38ba8', 'hl_running': '#f9e2af',
    'hl_ok':      '#a6e3a1', 'hl_fail':    '#f38ba8', 'drag_ghost': '#45475a',
    'log_bg':     '#11111b', 'log_fg':     '#cdd6f4',
    'var_node':   '#c6a0f6',
    'select_box': '#89b4fa' # 框选颜色
}
FONTS = {
    'node_title': ('Segoe UI', int(10 * SCALE_FACTOR), 'bold'), 
    'node_text': ('Segoe UI', int(8 * SCALE_FACTOR)),
    'code': ('Consolas', int(10 * SCALE_FACTOR)), 
    'h2': ('Segoe UI', int(11 * SCALE_FACTOR), 'bold'), 
    'small': ('Segoe UI', int(9 * SCALE_FACTOR)),
    'log': ('Consolas', int(9 * SCALE_FACTOR))
}

LOG_LEVELS = {
    'info': {'color': '#cdd6f4', 'icon': 'ℹ️'},
    'success': {'color': '#a6e3a1', 'icon': '✅'},
    'warning': {'color': '#f9e2af', 'icon': '⚠️'},
    'error': {'color': '#f38ba8', 'icon': '❌'},
    'exec': {'color': '#89b4fa', 'icon': '▶️'}
}

NODE_WIDTH = int(160 * SCALE_FACTOR)
HEADER_HEIGHT = int(28 * SCALE_FACTOR)
PORT_START_Y = int(45 * SCALE_FACTOR)
PORT_STEP_Y = int(22 * SCALE_FACTOR)
GRID_SIZE = int(20 * SCALE_FACTOR)

# --- 4. 节点配置 ---
NODE_CONFIG = {
    'start':    {'title': '▶ 开始', 'outputs': ['out'], 'color': COLORS['success']},
    'end':      {'title': '🛑 结束', 'outputs': [], 'color': COLORS['danger']},
    'mouse':    {'title': '🖱️ 鼠标操作', 'outputs': ['out'], 'color': COLORS['bg_header']},
    'keyboard': {'title': '⌨️ 键盘输入', 'outputs': ['out'], 'color': COLORS['bg_header']},
    'image':    {'title': '🎯 找图点击', 'outputs': ['found', 'timeout'], 'color': COLORS['accent']},
    'web':      {'title': '🌐 网页操作', 'outputs': ['out'], 'color': COLORS['bg_header']},
    'wait':     {'title': '⏳ 延时', 'outputs': ['out'], 'color': COLORS['control']},
    'if_img':   {'title': '🔍 图像检测', 'outputs': ['yes', 'no'], 'color': COLORS['control']},
    'if_static':{'title': '🔍 静止画面检测', 'outputs': ['yes', 'no'], 'color': COLORS['control']},
    'if_sound': {'title': '🔍 声音检测', 'outputs': ['yes', 'no'], 'color': COLORS['control']},
    'if_file':  {'title': '🔍 文件检测', 'outputs': ['yes', 'no'], 'color': COLORS['control']},
    'loop':     {'title': '🔄 循环', 'outputs': ['loop', 'exit'], 'color': COLORS['control']},
    'sequence': {'title': '🔀 逻辑判断链', 'outputs': ['else'], 'color': COLORS['control']},
    'set_var':  {'title': '💾 设置变量', 'outputs': ['out'], 'color': COLORS['var_node']},
    'var_switch':{'title': '🔍 变量检测与分流', 'outputs': ['else'], 'color': COLORS['var_node']},
}

ACTION_MAP = {'click': '单击左键', 'double_click': '双击左键', 'right_click': '单击右键', 'none': '不执行任何操作'}
ACTION_MAP_REVERSE = {v: k for k, v in ACTION_MAP.items()}
MOUSE_MODE_MAP = {'click': '点击操作', 'move': '移动鼠标', 'scroll': '滚动滑轮'}
MOUSE_MODE_REVERSE = {v: k for k, v in MOUSE_MODE_MAP.items()}
MATCH_STRATEGY_MAP = {'hybrid': '智能混合 (推荐)', 'template': '模板匹配 (快速)', 'feature': '特征匹配 (兼容性强)'}
MATCH_STRATEGY_REVERSE = {v: k for k, v in MATCH_STRATEGY_MAP.items()}
VAR_OP_MAP = {'=': '等于', '!=': '不等于', 'exists': '已定义(非空)', 'not_exists': '未定义/空'}
VAR_OP_REVERSE = {v: k for k, v in VAR_OP_MAP.items()}

PORT_TRANSLATION = {
    'out': '流程继续',
    'yes': '是', 'no': '否', 'found': '找到', 'timeout': '超时/未找到',
    'loop': '循环中', 'exit': '循环结束', 'else': '不匹配/其他'
}

NODE_OUTPUT_DOCS = {
    'image': {'found': '画面中发现了目标图片/锚点','timeout': '在规定时间内未找到目标，或滚动到底部仍未找到'},
    'if_img': {'yes': '屏幕上同时存在所有指定的图片条件','no': '任一图片条件不满足'},
    'if_static': {'yes': '指定区域画面在设定时间内保持静止','no': '画面发生了变化，或超过最大保护时间'},
    'if_sound': {'yes': '检测期间持续静音（未超过阈值）','no': '检测期间出现了声音（超过阈值）'},
    'if_file': {'yes': '文件存在且满足所有设定条件','no': '文件不存在或不满足条件'},
    'var_switch': {'yes': '变量值符合设定的条件（单变量模式）','no': '变量值不符合设定的条件（单变量模式）','else': '所有变量值未能同时满足任一设定的分流条件（多变量模式）'},
    'set_var': {'out': '所有变量已更新，继续下一步'},
    'loop': {'loop': '当前计数未达到设定值','exit': '循环次数已满'},
    'sequence': {'else': '所有前置分支步骤均未命中'},
    'wait': {'out': '延时结束'}, 'start': {'out': '流程开始'}, 'web': {'out': '网页已打开'},
    'mouse': {'out': '动作执行完毕'}, 'keyboard': {'out': '输入执行完毕'}
}

# --- 5. 工具类 ---
class ImageUtils:
    @staticmethod
    def img_to_b64(image):
        if not image: return None
        try: buffered = io.BytesIO(); image.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except: return None
    @staticmethod
    def b64_to_img(b64_str):
        if not b64_str: return None
        try: return Image.open(io.BytesIO(base64.b64decode(b64_str)))
        except: return None
    @staticmethod
    def make_thumb(image, size=(240, 135)):
        if not image: return None
        thumb = image.copy(); thumb.thumbnail(size); return ImageTk.PhotoImage(thumb)

class AudioEngine:
    @staticmethod
    def get_max_audio_peak():
        if not HAS_AUDIO: return 0.0
        try:
            try: comtypes.CoInitialize()
            except: pass
            sessions = AudioUtilities.GetAllSessions()
            max_peak = 0.0
            for session in sessions:
                if session.State == 1: 
                    meter = session._ctl.QueryInterface(IAudioMeterInformation)
                    peak = meter.GetPeakValue()
                    if peak > max_peak: max_peak = peak
            return max_peak
        except Exception: return 0.0

class VisionEngine:
    @staticmethod
    def capture_screen(bbox=None):
        try: return ImageGrab.grab(bbox=bbox)
        except: return None

    @staticmethod
    def locate(needle, confidence=0.8, timeout=0, stop_event=None, grayscale=True, multiscale=True, scaling_ratio=1.0, strategy='hybrid', region=None):
        start_time = time.time()
        while True:
            if stop_event and stop_event.is_set(): return None
            capture_bbox = None
            if region: capture_bbox = (region[0], region[1], region[0] + region[2], region[1] + region[3])
            haystack = VisionEngine.capture_screen(bbox=capture_bbox)
            if haystack is None:
                time.sleep(0.5)
                if timeout == 0 or (timeout > 0 and time.time() - start_time >= timeout): break
                continue
            result, _ = VisionEngine._advanced_match(needle, haystack, confidence, stop_event, grayscale, multiscale, scaling_ratio, strategy)
            if result:
                if region: return Box(result.left + region[0], result.top + region[1], result.width, result.height)
                return result
            if timeout == 0 or (timeout > 0 and time.time() - start_time >= timeout): break
            time.sleep(0.1)
        return None

    @staticmethod
    def _advanced_match(needle, haystack, confidence, stop_event, grayscale, multiscale, scaling_ratio, strategy):
        if not needle or not haystack: return None, 0.0
        if needle.width > haystack.width or needle.height > haystack.height: return None, 0.0
        if HAS_OPENCV:
            try:
                if grayscale: nA, hA = cv2.cvtColor(np.array(needle), cv2.COLOR_RGB2GRAY), cv2.cvtColor(np.array(haystack), cv2.COLOR_RGB2GRAY)
                else: nA, hA = cv2.cvtColor(np.array(needle), cv2.COLOR_RGB2BGR), cv2.cvtColor(np.array(haystack), cv2.COLOR_RGB2BGR)
                if strategy == 'feature': return VisionEngine._feature_match_akaze(nA, hA)
                nH, nW = nA.shape[:2]; hH, hW = hA.shape[:2]
                scales = [1.0]
                if multiscale: scales = np.unique(np.append(np.linspace(scaling_ratio * 0.8, scaling_ratio * 1.2, 10), [1.0, scaling_ratio]))
                best_max, best_rect = -1, None
                for s in scales:
                    if stop_event and stop_event.is_set(): return None, 0.0
                    tW, tH = int(nW * s), int(nH * s)
                    if tW < 5 or tH < 5 or tW > hW or tH > hH: continue
                    res = cv2.matchTemplate(hA, cv2.resize(nA, (tW, tH), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_max: best_max, best_rect = max_val, Box(max_loc[0], max_loc[1], tW, tH)
                    if best_max > 0.99: break
                if best_rect and best_max >= confidence: return best_rect, best_max
            except: pass
        try:
            res = pyautogui.locate(needle, haystack, confidence=confidence, grayscale=grayscale)
            if res: return Box(res.left, res.top, res.width, res.height), 1.0
        except: pass
        return None, 0.0

    @staticmethod
    def _feature_match_akaze(template, target, min_match_count=4):
        try:
            akaze = cv2.AKAZE_create()
            kp1, des1 = akaze.detectAndCompute(template, None); kp2, des2 = akaze.detectAndCompute(target, None)
            if des1 is None or des2 is None: return None, 0.0
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= min_match_count:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = template.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    x_min, y_min = np.min(dst[:, :, 0]), np.min(dst[:, :, 1])
                    x_max, y_max = np.max(dst[:, :, 0]), np.max(dst[:, :, 1])
                    return Box(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)), min(1.0, len(good)/len(kp1)*2.5)
            return None, 0.0
        except: return None, 0.0

    @staticmethod
    def compare_images(img1, img2, threshold=0.99):
        if not img1 or not img2: return False
        try:
            if img1.size != img2.size: img2 = img2.resize(img1.size, Image.LANCZOS)
            diff = ImageChops.difference(img1.convert('L'), img2.convert('L'))
            return (1.0 - (sum(diff.histogram()[10:]) / (img1.size[0] * img1.size[1]))) >= threshold
        except: return False

# --- 6. 日志面板 ---
class LogPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS['bg_panel'], **kwargs)
        self.expanded = False
        self.height_expanded, self.height_collapsed = 250, 30
        self.toolbar = tk.Frame(self, bg=COLORS['bg_header'], height=28); self.toolbar.pack_propagate(False)
        tk.Label(self.toolbar, text="📋 执行日志", bg=COLORS['bg_header'], fg='white', font=('Segoe UI', 9, 'bold')).pack(side='left', padx=10)
        tk.Button(self.toolbar, text="🔽 收起", command=self.toggle, bg=COLORS['bg_header'], fg=COLORS['fg_text'], bd=0, relief='flat').pack(side='right', padx=5)
        tk.Button(self.toolbar, text="🗑️ 清空", command=self.clear, bg=COLORS['bg_header'], fg=COLORS['danger'], bd=0, relief='flat').pack(side='right', padx=5)
        self.text_frame = tk.Frame(self, bg=COLORS['log_bg'])
        self.scrollbar = ttk.Scrollbar(self.text_frame)
        self.text_area = tk.Text(self.text_frame, bg=COLORS['log_bg'], fg=COLORS['log_fg'], font=FONTS['log'], state='disabled', yscrollcommand=self.scrollbar.set, bd=0, padx=5, pady=5)
        self.scrollbar.config(command=self.text_area.yview); self.scrollbar.pack(side='right', fill='y'); self.text_area.pack(side='left', fill='both', expand=True)
        for level, style in LOG_LEVELS.items(): self.text_area.tag_config(level, foreground=style['color'])
        self.status_bar = tk.Frame(self, bg=COLORS['bg_panel'], height=30, cursor="hand2"); self.status_bar.pack_propagate(False); self.status_bar.pack(side='bottom', fill='x')
        self.status_lbl = tk.Label(self.status_bar, text="就绪", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['code'], anchor='w', padx=10); self.status_lbl.pack(fill='both', expand=True)
        self.status_lbl.bind("<Button-1>", lambda e: self.toggle()); self.status_bar.bind("<Button-1>", lambda e: self.toggle())
    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded: self.config(height=self.height_expanded); self.toolbar.pack(side='top', fill='x'); self.text_frame.pack(side='top', fill='both', expand=True); self.status_lbl.config(fg=COLORS['accent'])
        else: self.toolbar.pack_forget(); self.text_frame.pack_forget(); self.config(height=self.height_collapsed); self.status_lbl.config(fg=COLORS['fg_sub'])
    def add_log(self, msg, level='info'):
        icon = LOG_LEVELS.get(level, {}).get('icon', 'ℹ️')
        self.status_lbl.config(text=f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")
        self.text_area.config(state='normal')
        self.text_area.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] [{level.upper()}] {msg}\n", level)
        self.text_area.see('end'); self.text_area.config(state='disabled')
    def clear(self): self.text_area.config(state='normal'); self.text_area.delete(1.0, 'end'); self.text_area.config(state='disabled')

# --- 7. 自动化核心引擎 ---
class AutomationCore:
    def __init__(self, log_callback, app_instance):
        self.running = False; self.stop_event = threading.Event(); 
        self.log = log_callback; self.app = app_instance
        self.project = None; self.runtime_memory = {}; self.sensor_states = {}; self.io_lock = threading.Lock()
        self.active_threads = 0; self.thread_lock = threading.Lock(); self.scaling_ratio = 1.0

    def load_project(self, project_data):
        self.project = project_data; self.scaling_ratio = 1.0
        dev_scale = self.project.get('metadata', {}).get('dev_scale_x', 1.0)
        runtime_scale_x, _ = get_scale_factor()
        if dev_scale > 0.1 and runtime_scale_x > 0.1: self.scaling_ratio = runtime_scale_x / dev_scale
        if self.project and 'nodes' in self.project:
            for nid, node in self.project['nodes'].items():
                data = node.get('data', {})
                if 'b64' in data and 'image' not in data and (img := ImageUtils.b64_to_img(data['b64'])): self.project['nodes'][nid]['data']['image'] = img
                if 'anchors' in data:
                    for anchor in data['anchors']:
                        if 'b64' in anchor and 'image' not in anchor and (img := ImageUtils.b64_to_img(anchor['b64'])): anchor['image'] = img
                if 'images' in data:
                    for img_item in data['images']:
                        if 'b64' in img_item and 'image' not in img_item and (img := ImageUtils.b64_to_img(img_item['b64'])): img_item['image'] = img
                if 'b64_preview' in data and 'roi_preview' not in data and (img := ImageUtils.b64_to_img(data['b64_preview'])):
                    self.project['nodes'][nid]['data']['roi_preview'] = ImageUtils.make_thumb(img)

    def start(self, start_node_id=None):
        if self.running or not self.project: return
        self.running = True; self.stop_event.clear(); self.runtime_memory, self.sensor_states = {}, {}; self.active_threads = 0
        self.log("🚀 引擎启动", "exec"); self.app.iconify(); self._start_sensors()
        self.app.btn_run.config(text="⏹ 停止运行", bg=COLORS['danger'])
        self.app.btn_run_menu.config(state='disabled')
        threading.Thread(target=self._run_flow_engine, args=(start_node_id,), daemon=True).start()

    def stop(self):
        if not self.running: return
        self.stop_event.set(); self.log("🛑 正在停止...", "warning")
        self.app.btn_run.config(text="▶ 启动 (Default)", bg=COLORS['success'])
        self.app.btn_run_menu.config(state='normal')

    def _start_sensors(self): pass

    def _smart_wait(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self.stop_event.is_set(): return False
            time.sleep(0.05)
        return True

    def _get_next_links(self, node_id, port_name='out'):
        return [l['target'] for l in self.project['links'] if l['source'] == node_id and l.get('source_port') == port_name]

    def _run_flow_engine(self, start_node_id=None):
        try:
            if start_node_id: start_nodes = [start_node_id]; self.log(f"🔗 从选中节点启动 (ID: {start_node_id})", "info")
            else: start_nodes = [nid for nid, n in self.project['nodes'].items() if n['type'] == 'start']
            if not start_nodes: self.log("未找到 '开始' 节点或目标节点", "error"); return
            for start_id in start_nodes: self._fork_node(start_id)
            while not self.stop_event.is_set():
                with self.thread_lock:
                    if self.active_threads <= 0: break
                time.sleep(0.5)
        except Exception as e: traceback.print_exc(); self.log(f"引擎异常: {str(e)}", "error")
        finally:
            self.running = False; self.log("🏁 流程结束", "info"); self.app.highlight_node_safe(None)
            self.app.after(0, self.app.deiconify); self.app.after(100, self.app.reset_ui_state)

    def _fork_node(self, node_id):
        with self.thread_lock: self.active_threads += 1
        threading.Thread(target=self._process_node_thread, args=(node_id,), daemon=True).start()

    def _process_node_thread(self, node_id):
        try:
            if self.stop_event.is_set(): return
            if not (node := self.project['nodes'].get(node_id)): return
            self.app.highlight_node_safe(node_id, 'running'); self.app.select_node_safe(node_id) 
            out_port = self._execute_node(node)
            if out_port == '__STOP__' or self.stop_event.is_set(): return
            port_cn = PORT_TRANSLATION.get(out_port, out_port)
            log_icon = "✅" if out_port in ['yes', 'found', 'loop', 'out'] else "🔻"
            if node['type'] == 'var_switch' and out_port.startswith('case_'):
                 self.log(f"  ↳ {log_icon} 满足条件分支: [{port_cn}]", "exec")
            elif node['type'] == 'sequence' and out_port != 'else': 
                self.log(f"  ↳ {log_icon} 命中分支: [第 {out_port} 步]", "exec")
            else: 
                self.log(f"  ↳ {log_icon} 触发出口: [{port_cn}]", "exec")
            
            self.app.highlight_node_safe(node_id, 'fail' if out_port in ['timeout', 'no', 'exit', 'else'] else 'ok')
            for next_id in self._get_next_links(node_id, out_port):
                if self.stop_event.is_set(): break
                self._fork_node(next_id)
        except Exception as e: self.log(f"节点 {node_id} 出错: {e}", "error"); traceback.print_exc()
        finally:
            with self.thread_lock: self.active_threads -= 1

    def _replace_variables(self, text):
        if not text: return text
        for var_name, var_value in self.runtime_memory.items():
            text = text.replace(f'${{{var_name}}}', str(var_value))
        return text

    def _execute_node(self, node):
        if self.stop_event.is_set(): return '__STOP__'
        ntype, data = node['type'], node.get('data', {})
        text = data.get('_user_title', NODE_CONFIG.get(ntype, {}).get('title', ntype))
        self.log(f"执行: {text}", "exec")
        
        if ntype == 'start': return 'out'
        if ntype == 'end': self.stop_event.set(); return '__STOP__'
        if ntype == 'wait': return 'out' if self._smart_wait(float(data.get('seconds', 1.0))) else '__STOP__'
        
        if ntype == 'set_var':
            if 'batch_vars' in data and isinstance(data['batch_vars'], list):
                for item in data['batch_vars']:
                    k, v = item.get('name'), item.get('value')
                    if k:
                        self.runtime_memory[k] = v
                        self.log(f"  💾 设置 [{k}] = {v}", "success")
            
            if data.get('var_name'):
                self.runtime_memory[data['var_name']] = data.get('var_value', '')
            return 'out'

        if ntype == 'var_switch':
            is_single_var = bool(data.get('var_name'))
            var_names = []
            current_values = []
            log_str = []
            
            if is_single_var:
                var_name = data.get('var_name', '').strip()
                if not var_name:
                    self.log("  ⚠️ 未指定变量名称", "warning")
                    return 'else'
                var_names = [var_name]
                operator = data.get('operator', '=')
                target_val = data.get('var_value', '')
                
                val = str(self.runtime_memory.get(var_name, ''))
                current_values.append(val)
                log_str.append(f"{var_name}={val}")
                
                self.log(f"  🔍 检查变量: {', '.join(log_str)}", "info")
                
                is_match = False
                if operator == '=': is_match = (val == target_val)
                elif operator == '!=': is_match = (val != target_val)
                elif operator == 'exists': is_match = (bool(val))
                elif operator == 'not_exists': is_match = (not bool(val))
                
                return 'yes' if is_match else 'no'
            else:
                raw_vars = data.get('var_list', '')
                var_names = [v.strip() for v in raw_vars.split(',') if v.strip()]
                if not var_names:
                    self.log("  ⚠️ 未指定变量列表", "warning")
                    return 'else'

                for vn in var_names:
                    val = str(self.runtime_memory.get(vn, ''))
                    current_values.append(val)
                    log_str.append(f"{vn}={val}")
                
                self.log(f"  🔍 检查变量: {', '.join(log_str)}", "info")

                cases = data.get('cases', [])
                for case in cases:
                    target_val = case.get('value', '')
                    port_id = case.get('id', 'else')
                    if all(cv == target_val for cv in current_values):
                        self.log(f"  ✅ 全部匹配值: '{target_val}'", "success")
                        return port_id
                
                self.log("  🔻 无匹配条件", "warning")
                return 'else'

        if ntype == 'sequence':
            steps = int(data.get('num_steps', 3))
            for i in range(1, steps + 1):
                if self.stop_event.is_set(): return '__STOP__'
                target_id = (self._get_next_links(node['id'], str(i)) or [None])[0]
                if not target_id: continue
                self.log(f"  👉 [分支 {i}] 检测...", "info")
                self.app.highlight_node_safe(target_id, 'running'); self.app.select_node_safe(target_id)
                if self._execute_node(self.project['nodes'][target_id]) in ['yes', 'found', 'out', 'loop']:
                    self.log(f"  ✅ [分支 {i}] 命中", "success"); self.app.highlight_node_safe(target_id, 'ok')
                    for next_nid in self._get_next_links(target_id, self._execute_node(self.project['nodes'][target_id])): self._fork_node(next_nid)
                    return '__STOP__' 
                else: self.app.highlight_node_safe(target_id, 'fail')
            return 'else'

        if ntype == 'if_sound':
            if not HAS_AUDIO: return 'yes'
            dur, thr = float(data.get('duration', 3.0)), float(data.get('threshold', 0.05))
            start = time.time(); is_loud = False
            while time.time() - start < dur:
                if self.stop_event.is_set(): return '__STOP__'
                if AudioEngine.get_max_audio_peak() > thr: is_loud = True; break
                time.sleep(0.1)
            return 'yes' if is_loud else 'no'

        if ntype == 'image':
            conf, timeout = float(data.get('confidence', 0.9)), max(1.0, float(data.get('timeout', 10.0)))
            search_region = None
            if data.get('search_mode') == 'region':
                try: search_region = (int(data.get('target_rect_x')), int(data.get('target_rect_y')), int(data.get('target_rect_w')), int(data.get('target_rect_h')))
                except: pass
            
            anchors = data.get('anchors', [])
            if anchors:
                self.log(f"  ⚓ 多锚点定位 ({len(anchors)}个)...", "info"); primary_res = None
                for i, anchor in enumerate(anchors):
                    if self.stop_event.is_set(): return '__STOP__'
                    res = VisionEngine.locate(anchor['image'], confidence=conf, timeout=(timeout if i==0 else 2.0), stop_event=self.stop_event, grayscale=data.get('use_grayscale',True), scaling_ratio=self.scaling_ratio, strategy=data.get('match_strategy','hybrid'))
                    if not res: return 'timeout'
                    if i == 0: primary_res = res
                if primary_res:
                    off_x, off_y = data.get('target_rect_x',0)-anchors[0].get('rect_x',0), data.get('target_rect_y',0)-anchors[0].get('rect_y',0)
                    search_region = (max(0, int(primary_res.left+off_x)-15), max(0, int(primary_res.top+off_y)-15), data.get('target_rect_w',100)+30, data.get('target_rect_h',100)+30)

            start_time = time.time()
            while True:
                if self.stop_event.is_set(): return '__STOP__'
                res = VisionEngine.locate(data.get('image'), confidence=conf, timeout=0, stop_event=self.stop_event, grayscale=data.get('use_grayscale', True), multiscale=data.get('use_multiscale', True), scaling_ratio=self.scaling_ratio, strategy=data.get('match_strategy', 'hybrid'), region=search_region)
                if res:
                    with self.io_lock:
                        if (act := data.get('click_type', 'click')) != 'none':
                            rx, ry = data.get('relative_click_pos', (0.5, 0.5))
                            bx, by = (res.left + res.width * rx) / SCALE_X, (res.top + res.height * ry) / SCALE_Y
                            pyautogui.moveTo(bx + int(data.get('offset_x', 0)), by + int(data.get('offset_y', 0)))
                            getattr(pyautogui, {'click':'click','double_click':'doubleClick','right_click':'rightClick'}.get(act, 'click'))()
                    return 'found'
                
                if data.get('auto_scroll', False):
                    capture_bbox = (search_region[0],search_region[1],search_region[0]+search_region[2],search_region[1]+search_region[3]) if search_region else None
                    img_before = VisionEngine.capture_screen(bbox=capture_bbox)
                    with self.io_lock:
                        if search_region: pyautogui.moveTo((search_region[0]+search_region[2]/2)/SCALE_X, (search_region[1]+search_region[3]/2)/SCALE_Y)
                        else: pyautogui.moveTo(pyautogui.size()[0]/2, pyautogui.size()[1]/2)
                        pyautogui.scroll(int(data.get('scroll_amount', -500)))
                    if not self._smart_wait(0.8): return '__STOP__'
                    if VisionEngine.compare_images(img_before, VisionEngine.capture_screen(bbox=capture_bbox), 0.995):
                        self.log("  🛑 到底部", "warning"); break
                    if time.time() - start_time > max(60, timeout * 2): break
                else:
                    if time.time() - start_time > timeout: break
                    time.sleep(0.2)
            return 'timeout'
        
        if ntype == 'mouse':
            with self.io_lock:
                m = data.get('mouse_mode', 'click')
                if m == 'click': getattr(pyautogui, {'click':'click','double_click':'doubleClick','right_click':'rightClick'}.get(data.get('click_type', 'click'), 'click'))()
                elif m == 'move': pyautogui.moveTo(int(data.get('x',0)), int(data.get('y',0)))
                elif m == 'scroll': pyautogui.scroll(int(data.get('amount', -500)))
            return 'out'
        
        if ntype == 'keyboard':
            self._smart_wait(0.2)
            with self.io_lock:
                if data.get('kb_mode', 'text') == 'text':
                    if not data.get('slow_type', False) and HAS_PYPERCLIP:
                        try: pyperclip.copy(data.get('text','')); pyautogui.hotkey('ctrl','v')
                        except: pyautogui.write(data.get('text',''))
                    else: pyautogui.write(data.get('text',''))
                    if data.get('press_enter', False): pyautogui.press('enter')
                else: pyautogui.hotkey(*[x.strip() for x in data.get('key_name', 'enter').lower().split('+')])
            return 'out'
        
        if ntype == 'web': webbrowser.open(data.get('url')); self._smart_wait(2); return 'out'
        
        if ntype == 'loop':
            with self.io_lock:
                k = f"loop_{node['id']}"; c = self.runtime_memory.get(k, 0)
                if c < int(data.get('count', 3)): self.runtime_memory[k] = c + 1; return 'loop'
                else: 
                    if k in self.runtime_memory: del self.runtime_memory[k]
                    return 'exit'
        
        if ntype == 'if_static':
            roi = data.get('roi')
            if not roi: return 'no'
            dur, thr, max_t = float(data.get('duration', 5.0)), float(data.get('threshold', 0.98)), float(data.get('timeout', 20.0))
            s_t, o_t, l_f = time.time(), time.time(), VisionEngine.capture_screen(bbox=roi)
            while True:
                if self.stop_event.is_set(): return '__STOP__'
                if time.time() - s_t >= dur: return 'yes'
                if time.time() - o_t > max_t: return 'no'
                if not self._smart_wait(float(data.get('interval', 1.0))): return '__STOP__'
                c_f = VisionEngine.capture_screen(bbox=roi)
                if c_f and l_f and not VisionEngine.compare_images(l_f, c_f, thr): s_t, l_f = time.time(), c_f

        if ntype == 'if_img':
            if not (imgs := data.get('images', [])): return 'no'
            hay = VisionEngine.capture_screen()
            if hay is None: return 'no'
            for img in imgs:
                res, _ = VisionEngine._advanced_match(img.get('image'), hay, float(data.get('confidence',0.9)), self.stop_event, data.get('use_grayscale',True), data.get('use_multiscale',True), self.scaling_ratio, data.get('match_strategy','hybrid'))
                if not res: return 'no'
            return 'yes'
        
        if ntype == 'if_file':
            file_path = data.get('file_path', '')
            if not file_path: return 'no'
            file_path = self._replace_variables(file_path)
            exists = os.path.exists(file_path)
            self.log(f"  🔍 文件检查: [{file_path}] {'存在' if exists else '不存在'}", "info")
            return 'yes' if exists else 'no'
        
        return 'out'

# --- 8. UI 组件 ---
class GraphNode:
    def __init__(self, canvas, node_id, ntype, x, y, data=None):
        self.canvas, self.id, self.type, self.x, self.y = canvas, node_id, ntype, x, y
        self.data = data if data is not None else {}
        cfg = NODE_CONFIG.get(ntype, {})
        self.title_text, self.header_color = cfg.get('title', ntype), cfg.get('color', COLORS['bg_header'])
        
        if ntype == 'sequence': 
            self.outputs = [str(i) for i in range(1, int(self.data.get('num_steps', 3)) + 1)] + ['else']
        elif ntype == 'var_switch':
            is_single_var = bool(self.data.get('var_name'))
            if is_single_var:
                self.outputs = ['yes', 'no']
            else:
                cases = self.data.get('cases', [])
                self.outputs = [c['id'] for c in cases] + ['else']
        else: 
            self.outputs = cfg.get('outputs', [])

        self.w, self.h = NODE_WIDTH, PORT_START_Y + max(1, len(self.outputs)) * PORT_STEP_Y if self.outputs else 50
        self.tags = (f"node_{self.id}", "node"); self.body_item, self.sel_rect = None, None
        self.draw()
    def _truncate_text(self, text, max_width, font):
        if len(text) <= 15:
            return text
        for i in range(15, 0, -1):
            truncated = text[:i] + "..."
            if len(truncated) <= 18:
                return truncated
        return text[:10] + "..."
    
    def draw(self):
        z = self.canvas.zoom; vx, vy, vw, vh = self.x*z, self.y*z, self.w*z, self.h*z
        self.canvas.delete(f"node_{self.id}")
        
        # Glow Effect / Selection border (Drawn behind the node body)
        self.sel_rect = self.canvas.create_rectangle(
            vx - 3*z, vy - 3*z, vx + vw + 3*z, vy + vh + 3*z,
            outline=COLORS['accent'], width=4*z,
            tags=self.tags+('selection',), state='hidden'
        )

        self.canvas.create_rectangle(vx+4*z,vy+4*z,vx+vw+4*z,vy+vh+4*z,fill=COLORS['shadow'],outline="",tags=self.tags)
        self.body_item=self.canvas.create_rectangle(vx,vy,vx+vw,vy+vh,fill=COLORS['bg_node'],outline=COLORS['bg_node'],width=2*z,tags=self.tags+('body',))
        self.canvas.create_rectangle(vx,vy,vx+vw,vy+HEADER_HEIGHT*z,fill=self.header_color,outline="",tags=self.tags+('header',))
        
        title_text = self.data.get('_user_title', self.title_text)
        truncated_title = self._truncate_text(title_text, vw - 20*z, ('Segoe UI', max(6, int(10*z)), 'bold'))
        self.canvas.create_text(vx+10*z, vy+14*z, 
                               text=truncated_title, 
                               fill='#111' if self.type in ['start','end'] else COLORS['fg_title'],
                               font=('Segoe UI', max(6, int(10*z)), 'bold'),
                               anchor="w", 
                               width=(vw - 20*z),
                               tags=self.tags)
        if self.type != 'start':
            iy=self.get_input_port_y(visual=True);self.canvas.create_oval(vx-5*z,iy-5*z,vx+5*z,iy+5*z,fill=COLORS['socket'],outline=COLORS['bg_canvas'],width=2*z,tags=self.tags+('port_in',))
        port_colors = {'yes':'socket_yes','loop':'socket_yes','found':'socket_yes','no':'socket_no','exit':'socket_no','timeout':'socket_no','else':'socket_no'}
        canvas_labels = {'out':'', 'yes':'是', 'no':'否', 'loop':'循环', 'exit':'结束', 'found':'找到', 'timeout':'超时', 'else':'Else'}
        
        if self.type == 'var_switch':
            for i, name in enumerate(self.outputs):
                if name == 'else': 
                    lbl = "其他 (Else)"
                else:
                    case_val = next((c['value'] for c in self.data.get('cases', []) if c['id'] == name), "?")
                    lbl = f"值={case_val}"
                canvas_labels[name] = lbl

        for i,name in enumerate(self.outputs):
            py,color=self.get_output_port_y(i,visual=True),COLORS[port_colors.get(name,'socket')]
            self.canvas.create_oval(vx+vw-5*z,py-5*z,vx+vw+5*z,py+5*z,fill=color,outline=COLORS['bg_canvas'],width=2*z,tags=self.tags+(f'port_out_{name}','port_out',name))
            if (lbl:=canvas_labels.get(name,name)): self.canvas.create_text(vx+vw-12*z,py,text=lbl,fill=COLORS['fg_sub'],font=('Segoe UI', max(5, int(8*z))),anchor="e",tags=self.tags)
        
        is_selected = (self.id in self.canvas.selected_node_ids)
        if is_selected:
            self.canvas.itemconfig(self.sel_rect, state='normal')
            self.canvas.tag_lower(self.sel_rect, self.body_item) # Ensure glow is behind body
        
        self.hover_rect=self.canvas.create_rectangle(vx-1*z,vy-1*z,vx+vw+1*z,vy+vh+1*z,outline=COLORS['hover'],width=1*z,state='hidden',tags=self.tags+('hover',))
    
    def set_sensor_active(self,is_active): self.canvas.itemconfig(self.body_item,outline=COLORS['active_border'] if is_active else COLORS['bg_node'])
    def get_input_port_y(self,visual=False): offset=HEADER_HEIGHT+14; return (self.y+offset)*self.canvas.zoom if visual else self.y+offset
    def get_output_port_y(self,index=0,visual=False): offset=PORT_START_Y+(index*PORT_STEP_Y); return (self.y+offset)*self.canvas.zoom if visual else self.y+offset
    def get_port_y_by_name(self,port_name,visual=False):
        try: idx=self.outputs.index(port_name)
        except ValueError: idx=0
        return self.get_output_port_y(idx,visual)
    def move(self,dx,dy): self.x+=dx; self.y+=dy; self.draw()
    def set_pos(self,x,y): self.x,self.y=x,y; self.draw()
    def set_selected(self,selected): 
        self.canvas.itemconfig(self.sel_rect,state='normal' if selected else 'hidden')
        if selected: self.canvas.tag_lower(self.sel_rect, self.body_item)
    def contains(self,log_x,log_y): return self.x<=log_x<=self.x+self.w and self.y<=log_y<=self.y+self.h
    def update_data(self,key,value): self.data[key]=value; (key=='_user_title' and self.draw())

class FlowEditor(tk.Canvas):
    def __init__(self,parent,app,**kwargs):
        super().__init__(parent,bg=COLORS['bg_canvas'],highlightthickness=0,**kwargs)
        self.app,self.nodes,self.links,self.links_map=app,{},[],{}
        self.selected_node_ids = set() # Changed to set for multi-select
        self.drag_data = {"type": None}
        self.wire_start = None
        self.temp_wire = None
        self.selection_box = None
        self.space_pressed,self.zoom=False,1.0;self.bind_events();self.full_redraw()
        
    @property
    def selected_node_id(self):
        # Backward compatibility property, returns one of the selected nodes or None
        if self.selected_node_ids:
            return next(iter(self.selected_node_ids))
        return None

    def bind_events(self):
        self.bind("<ButtonPress-1>",self.on_lmb_press);self.bind("<B1-Motion>",self.on_lmb_drag);self.bind("<ButtonRelease-1>",self.on_lmb_release)
        self.bind("<ButtonPress-2>",self.on_pan_start);self.bind("<B2-Motion>",self.on_pan_drag);self.bind("<ButtonRelease-2>",self.on_pan_end)
        self.bind("<Button-3>",self.on_right_click);self.bind("<MouseWheel>",self.on_scroll)
        self.bind_all("<KeyPress-space>",self._on_space_down,add="+");self.bind_all("<KeyRelease-space>",self._on_space_up,add="+")
        self.bind_all("<Delete>",self._on_delete_press,add="+");self.bind_all("<BackSpace>",self._on_delete_press,add="+")
        self.bind("<Motion>",self.on_mouse_move);
        self.bind("<Configure>",self.full_redraw)
    def _on_space_down(self,e): self.space_pressed=True;self.config(cursor="fleur")
    def _on_space_up(self,e): self.space_pressed=False;self.config(cursor="arrow")
    def _on_delete_press(self,e):
        if self.selected_node_ids:
            # Create list copy to avoid runtime error during deletion
            to_delete = list(self.selected_node_ids)
            for nid in to_delete:
                if nid in self.nodes: self.delete_node(nid)
    
    def on_mouse_move(self,event):
        lx,ly=self.get_logical_pos(event.x,event.y)
        for node in self.nodes.values():
            self.itemconfig(node.hover_rect, state='hidden')
        for node in self.nodes.values():
            if node.contains(lx, ly) and node.id not in self.selected_node_ids:
                self.itemconfig(node.hover_rect, state='normal')
                break
    def get_logical_pos(self,event_x,event_y): return self.canvasx(event_x)/self.zoom,self.canvasy(event_y)/self.zoom
    def full_redraw(self,event=None): self.delete("all");self._draw_grid(); [n.draw() for n in self.nodes.values()]; self.redraw_links()
    def _draw_grid(self):
        w,h=self.winfo_width(),self.winfo_height(); x1,y1,x2,y2=self.canvasx(0),self.canvasy(0),self.canvasx(w),self.canvasy(h)
        if (step:=int(GRID_SIZE*self.zoom))<5: return
        start_x,start_y=int(x1//step)*step,int(y1//step)*step
        for i in range(start_x,int(x2)+step,step): self.create_line(i,y1,i,y2,fill=COLORS['grid'],tags="grid")
        for i in range(start_y,int(y2)+step,step): self.create_line(x1,i,x2,i,fill=COLORS['grid'],tags="grid")
        self.tag_lower("grid")
    def add_node(self,ntype,x,y,data=None,node_id=None): 
        node=GraphNode(self,node_id or str(uuid.uuid4()),ntype,x,y,data)
        self.nodes[node.id]=node
        self.select_node(node.id) 
        return node
    def duplicate_node(self,node_id):
        if not (src:=self.nodes.get(node_id)): return
        new_data=copy.deepcopy({k:v for k,v in src.data.items() if k not in ['image','tk_image','images', 'roi_preview', 'anchors']})
        if 'b64' in src.data and (img := ImageUtils.b64_to_img(src.data['b64'])): new_data.update({'image':img,'tk_image':ImageUtils.make_thumb(img)})
        if 'anchors' in src.data:
            new_data['anchors'] = []
            for anc in src.data['anchors']:
                new_anc = copy.deepcopy({k:v for k,v in anc.items() if k not in ['image', 'tk_image']})
                if 'b64' in new_anc and (img := ImageUtils.b64_to_img(new_anc['b64'])): new_anc['image'] = img
                new_data['anchors'].append(new_anc)
        if 'b64_preview' in src.data and (img := ImageUtils.b64_to_img(src.data['b64_preview'])): new_data.update({'roi_preview':ImageUtils.make_thumb(img)})
        if 'images' in src.data:
            new_data['images'] = []
            for item in src.data['images']:
                new_item_data = copy.deepcopy({k:v for k,v in item.items() if k not in ['image', 'tk_image']})
                if 'b64' in new_item_data and (img := ImageUtils.b64_to_img(new_item_data['b64'])): new_item_data.update({'image':img,'tk_image':ImageUtils.make_thumb(img, size=(120,67))})
                new_data['images'].append(new_item_data)
        self.select_node(self.add_node(src.type,src.x+20,src.y+20,data=new_data).id)
    def disconnect_node(self,node_id): 
        original_count = len(self.links)
        self.links = [l for l in self.links if l['source'] != node_id and l['target'] != node_id]
        if len(self.links) < original_count: self.redraw_links()
    def disconnect_port(self,node_id,port_name,is_input):
        original_len=len(self.links)
        if is_input: self.links=[l for l in self.links if not l['target']==node_id]
        else: self.links=[l for l in self.links if not (l['source']==node_id and l.get('source_port')==port_name)]
        if len(self.links)<original_len: self.redraw_links()
    def delete_node(self,node_id):
        if node_id in self.nodes:
            self.disconnect_node(node_id)
            self.delete(f"node_{node_id}")
            del self.nodes[node_id]
            if node_id in self.selected_node_ids:
                self.selected_node_ids.remove(node_id)
                self.app.property_panel.clear()
            self.redraw_links()
    
    def on_lmb_press(self,event):
        if self.space_pressed: self.on_pan_start(event); return
        lx,ly=self.get_logical_pos(event.x,event.y)[0],self.get_logical_pos(event.x,event.y)[1]
        vx,vy=self.canvasx(event.x),self.canvasy(event.y)
        
        # 1. Check Wire
        for item in self.find_overlapping(vx-2,vy-2,vx+2,vy+2):
            tags=self.gettags(item)
            if "port_out" in tags:
                if not (nid:=next((t[5:] for t in tags if t.startswith("node_")) ,None)) or nid not in self.nodes: continue
                pname=next((t for t in tags if t in self.nodes[nid].outputs),'out')
                self.wire_start={'node':self.nodes[nid],'port':pname};self.drag_data={"type":"wire"}; return
        
        # 2. Check Node
        clicked_node=next((node for node in reversed(list(self.nodes.values())) if node.contains(lx,ly)),None)
        
        is_ctrl_pressed = (event.state & 0x0004) # Check Ctrl key state
        
        if clicked_node:
            # Logic for Node Selection
            if is_ctrl_pressed:
                # Toggle selection
                if clicked_node.id in self.selected_node_ids:
                    self.deselect_node(clicked_node.id)
                else:
                    self.select_node(clicked_node.id, add=True)
            else:
                # If clicking a node that is not in current selection, select ONLY it
                if clicked_node.id not in self.selected_node_ids:
                    self.select_node(clicked_node.id, add=False)
                # If clicking a node ALREADY in selection, do nothing (wait for drag)
            
            # Prepare drag data for ALL selected nodes
            start_positions = {}
            for nid in self.selected_node_ids:
                if nid in self.nodes:
                    n = self.nodes[nid]
                    start_positions[nid] = (n.x, n.y)
            
            self.drag_data = {
                "type": "node",
                "start_lx": lx,
                "start_ly": ly,
                "start_positions": start_positions,
                "dragged": False
            }
            # Raise all selected nodes visual
            for nid in self.selected_node_ids:
                 self.tag_raise(f"node_{nid}")

        else:
            # 3. Clicked Empty Space
            if not is_ctrl_pressed:
                self.select_node(None) # Clear selection
            
            # Start Box Selection (Marquee)
            self.drag_data = {
                "type": "box_select",
                "start_vx": vx,
                "start_vy": vy
            }
            self.selection_box = self.create_rectangle(vx, vy, vx, vy, outline=COLORS['select_box'], width=2, dash=(4,4), tags="selection_box")

    def on_lmb_drag(self,event):
        if self.space_pressed: self.on_pan_drag(event); return
        lx,ly=self.get_logical_pos(event.x,event.y)[0],self.get_logical_pos(event.x,event.y)[1]
        vx,vy=self.canvasx(event.x),self.canvasy(event.y)
        
        # Auto Scroll logic
        scroll_margin = 30
        scroll_speed = 10
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        scroll_x = 0; scroll_y = 0
        if event.x < scroll_margin: scroll_x = scroll_speed
        elif event.x > canvas_width - scroll_margin: scroll_x = -scroll_speed
        if event.y < scroll_margin: scroll_y = scroll_speed
        elif event.y > canvas_height - scroll_margin: scroll_y = -scroll_speed
        
        if scroll_x != 0 or scroll_y != 0:
            self.scan_dragto(event.x + scroll_x, event.y + scroll_y, gain=1)
            self._draw_grid()
            vx = self.canvasx(event.x); vy = self.canvasy(event.y)
            lx, ly = self.get_logical_pos(event.x, event.y)
        
        if self.drag_data["type"]=="node":
            self.drag_data["dragged"] = True
            dx = lx - self.drag_data["start_lx"]
            dy = ly - self.drag_data["start_ly"]
            
            # Batch move all selected nodes
            start_positions = self.drag_data.get("start_positions", {})
            for nid, (sx, sy) in start_positions.items():
                if nid in self.nodes:
                    node = self.nodes[nid]
                    node.x = sx + dx
                    node.y = sy + dy
                    node.draw()
            
            self.redraw_links()
            
        elif self.drag_data["type"]=="box_select":
            if self.selection_box:
                start_x = self.drag_data["start_vx"]
                start_y = self.drag_data["start_vy"]
                self.coords(self.selection_box, start_x, start_y, vx, vy)

        elif self.drag_data["type"]=="wire":
            if self.temp_wire: self.delete(self.temp_wire)
            n,p=self.wire_start['node'],self.wire_start['port']
            self.temp_wire=self.draw_bezier((n.x+n.w)*self.zoom,n.get_port_y_by_name(p,visual=True),vx,vy,state="active")

    def on_lmb_release(self,event):
        if self.space_pressed: self.on_pan_end(event); return
        
        if self.drag_data.get("type")=="node": 
            if self.drag_data.get("dragged", False):
                # Snap all selected nodes to grid
                for nid in self.selected_node_ids:
                    if nid in self.nodes:
                        node = self.nodes[nid]
                        node.set_pos(round(node.x/GRID_SIZE)*GRID_SIZE, round(node.y/GRID_SIZE)*GRID_SIZE)
                self.redraw_links()
        
        elif self.drag_data.get("type")=="box_select":
            if self.selection_box:
                # Find items within the selection box
                x1, y1, x2, y2 = self.coords(self.selection_box)
                # Ensure correct order for find_overlapping
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                
                # Check overlapping items
                overlapping = self.find_overlapping(x1, y1, x2, y2)
                found_nodes = set()
                
                for item in overlapping:
                    tags = self.gettags(item)
                    for t in tags:
                        if t.startswith("node_"):
                            nid = t[5:]
                            if nid in self.nodes:
                                found_nodes.add(nid)
                
                # Add found nodes to selection (Multi-select behavior)
                is_ctrl_pressed = (event.state & 0x0004)
                if not is_ctrl_pressed:
                     self.select_node(None) # Clear if not adding
                
                for nid in found_nodes:
                    self.select_node(nid, add=True)

                self.delete(self.selection_box)
                self.selection_box = None

        elif self.drag_data.get("type")=="wire":
            if self.temp_wire: self.delete(self.temp_wire)
            lx,ly=self.get_logical_pos(event.x,event.y)
            for node in self.nodes.values():
                if node.type == 'start' or node.id==self.wire_start['node'].id: continue
                if math.hypot(lx-node.x,ly-node.get_input_port_y(visual=False))<(25/self.zoom):
                    src_id,src_port=self.wire_start['node'].id,self.wire_start['port']
                    if not any(l['source']==src_id and l['source_port']==src_port and l['target']==node.id for l in self.links):
                        self.links.append({'id':str(uuid.uuid4()),'source':src_id,'source_port':src_port,'target':node.id}); self.redraw_links()
                    break
        self.drag_data,self.wire_start,self.temp_wire={"type":None},None,None

    def on_pan_start(self,event): self.config(cursor="fleur");self.scan_mark(event.x,event.y)
    def on_pan_drag(self,event): self.scan_dragto(event.x,event.y,gain=1);self._draw_grid()
    def on_pan_end(self,event): self.config(cursor="arrow")
    
    def on_scroll(self,e):
        old_canvas_x = self.canvasx(e.x); old_canvas_y = self.canvasy(e.y)
        old_zoom = self.zoom; new_zoom = max(0.4, min(2.5, self.zoom * (1.1 if e.delta > 0 else 0.9)))
        self.zoom = new_zoom
        scale = new_zoom / old_zoom
        self.full_redraw()
        new_canvas_x = old_canvas_x * scale; new_canvas_y = old_canvas_y * scale
        self.scan_mark(e.x, e.y)
        self.scan_dragto(int(e.x + (old_canvas_x - new_canvas_x)), int(e.y + (old_canvas_y - new_canvas_y)), gain=1)
        self._draw_grid()

    def select_node(self, node_id, add=False):
        if not add:
            # Deselect all
            for nid in list(self.selected_node_ids):
                if nid in self.nodes: self.nodes[nid].set_selected(False)
            self.selected_node_ids.clear()
        
        if node_id and node_id in self.nodes:
            self.selected_node_ids.add(node_id)
            self.nodes[node_id].set_selected(True)
            self.app.property_panel.load_node(self.nodes[node_id]) # Show properties for the last selected
        elif not add:
            self.app.property_panel.clear()
        
        self.redraw_links()

    def deselect_node(self, node_id):
        if node_id in self.selected_node_ids:
            if node_id in self.nodes: self.nodes[node_id].set_selected(False)
            self.selected_node_ids.remove(node_id)
            # Update property panel to show another selected node or clear
            if self.selected_node_ids:
                nid = next(iter(self.selected_node_ids))
                self.app.property_panel.load_node(self.nodes[nid])
            else:
                self.app.property_panel.clear()
            self.redraw_links()

    def draw_bezier(self,x1,y1,x2,y2,state="normal",link_id=None, highlighted=False):
        offset=max(50*self.zoom,abs(x1-x2)*0.5)
        tags=("link",)+((f"link_{link_id}",) if link_id else ())
        width = 4*self.zoom if highlighted else (3*self.zoom if state=="active" else 2*self.zoom)
        color = COLORS['wire_hl'] if highlighted else COLORS['wire_active' if state=="active" else 'wire']
        return self.create_line(x1,y1,x1+offset,y1,x2-offset,y2,x2,y2,smooth=True,splinesteps=50,fill=color,width=width,arrow=tk.LAST,arrowshape=(8*self.zoom,10*self.zoom,3*self.zoom),tags=tags)
    def redraw_links(self):
        self.delete("link"); self.links_map={l.get('id',str(uuid.uuid4())):l for l in self.links}
        for lid,l in self.links_map.items():
            if l['source'] in self.nodes and l['target'] in self.nodes:
                n1,n2=self.nodes[l['source']],self.nodes[l['target']]
                is_hl = (l['source'] in self.selected_node_ids or l['target'] in self.selected_node_ids)
                self.draw_bezier((n1.x+n1.w)*self.zoom,n1.get_port_y_by_name(l.get('source_port','out'),visual=True),n2.x*self.zoom,n2.get_input_port_y(visual=True),link_id=lid, highlighted=is_hl)
        self.tag_lower("link"); self.tag_lower("grid")
    def on_right_click(self,event):
        vx,vy,lx,ly=self.canvasx(event.x),self.canvasy(event.y),self.get_logical_pos(event.x,event.y)[0],self.get_logical_pos(event.x,event.y)[1]
        for item in self.find_overlapping(vx-3,vy-3,vx+3,vy+3):
            tags=self.gettags(item)
            if not (nid:=next((t[5:] for t in tags if t.startswith("node_")),None)): continue
            if "port_out" in tags: pname=next((t for t in tags if t in self.nodes[nid].outputs),'out'); self.disconnect_port(nid,pname,is_input=False); return
            if "port_in" in tags: self.disconnect_port(nid,None,is_input=True); return
        for node in reversed(list(self.nodes.values())):
            if node.contains(lx,ly):
                m=tk.Menu(self,tearoff=0,bg=COLORS['bg_card'],fg=COLORS['fg_text'],font=FONTS['small'])
                m.add_command(label="📥 复制节点",command=lambda n=node.id:self.duplicate_node(n))
                m.add_command(label="🔗 断开所有连接",command=lambda n=node.id:self.disconnect_node(n))
                m.add_command(label="▶ 从此运行",command=lambda n=node.id:self.app.core.start(n))
                m.add_separator()
                m.add_command(label="❌ 删除节点",command=lambda n=node.id:self.delete_node(n),foreground=COLORS['danger'])
                m.post(event.x_root,event.y_root)
                return
    def get_data(self):
        nodes_d, project_data = {}, {'links':self.links}
        for nid,n in self.nodes.items():
            clean_data={k:v for k,v in n.data.items() if k not in ['image','tk_image','images', 'roi_preview', 'anchors']}
            if 'image' in n.data: clean_data['b64']=ImageUtils.img_to_b64(n.data['image'])
            if 'anchors' in n.data:
                clean_data['anchors'] = []
                for anc in n.data['anchors']:
                    new_anc = {k: v for k, v in anc.items() if k not in ['image']}
                    if 'image' in anc: new_anc['b64'] = ImageUtils.img_to_b64(anc['image'])
                    clean_data['anchors'].append(new_anc)
            if 'images' in n.data:
                clean_data['images'] = []
                for img_item in n.data['images']:
                    new_item = {k: v for k, v in img_item.items() if k not in ['image', 'tk_image']}
                    if 'image' in img_item: new_item['b64'] = ImageUtils.img_to_b64(img_item['image'])
                    clean_data['images'].append(new_item)
            if 'b64_preview' in n.data: clean_data['b64_preview'] = n.data['b64_preview']
            nodes_d[nid]={'id':nid,'type':n.type,'x':int(n.x),'y':int(n.y),'data':clean_data}
        for link in self.links: link.setdefault('id',str(uuid.uuid4()))
        project_data['nodes'] = nodes_d; project_data['metadata']={'dev_scale_x':SCALE_X,'dev_scale_y':SCALE_Y}; return project_data
    def load_data(self,data):
        self.delete("all");self.nodes.clear();self.links.clear()
        for nid,n_data in data.get('nodes',{}).items():
            d=n_data.get('data',{})
            if 'b64' in d and (img:=ImageUtils.b64_to_img(d['b64'])): d.update({'image':img,'tk_image':ImageUtils.make_thumb(img)})
            if 'anchors' in d:
                for anc in d['anchors']:
                    if 'b64' in anc and (img:=ImageUtils.b64_to_img(anc['b64'])): anc['image'] = img
            if 'images' in d:
                for img_item in d['images']:
                    if 'b64' in img_item and (img := ImageUtils.b64_to_img(img_item['b64'])): img_item.update({'image': img, 'tk_image': ImageUtils.make_thumb(img, size=(120, 67))})
            if 'b64_preview' in d and (img := ImageUtils.b64_to_img(d['b64_preview'])): d['roi_preview'] = ImageUtils.make_thumb(img)
            self.add_node(n_data['type'],n_data['x'],n_data['y'],data=d,node_id=nid)
        self.links=data.get('links',[]);[link.setdefault('id',str(uuid.uuid4())) for link in self.links]
        self.app.core.load_project(data); self.full_redraw()

class PropertyPanel(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS['bg_panel']); self.app, self.current_node = app, None
        self.snip_button, self.test_match_button, self.test_result_label = None, None, None
        self.is_monitoring_audio = False 
        
        tk.Label(self, text="属性设置", bg=COLORS['bg_sidebar'], fg=COLORS['accent'], font=FONTS['h2'], pady=10).pack(fill='x')
        
        scroll_frame = tk.Frame(self, bg=COLORS['bg_panel'])
        scroll_frame.pack(fill='both', expand=True)
        
        self.scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical")
        self.scrollbar.pack(side='right', fill='y')
        
        self.canvas = tk.Canvas(scroll_frame, bg=COLORS['bg_panel'], yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        
        self.scrollbar.config(command=self.canvas.yview)
        
        self.content = tk.Frame(self.canvas, bg=COLORS['bg_panel'], padx=15, pady=15)
        self.content_id = self.canvas.create_window((0, 0), window=self.content, anchor='nw')
        
        # 修复布局问题：绑定 Canvas 的配置事件，强制调整内容宽度
        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        self.clear()
        
    def _on_canvas_configure(self, event):
        """当画布大小改变时，调整内部窗口的宽度以填满画布"""
        self.canvas.itemconfig(self.content_id, width=event.width)

    def clear(self): 
        [w.destroy() for w in self.content.winfo_children()]
        tk.Label(self.content, text="未选择节点", bg=COLORS['bg_panel'], fg=COLORS['fg_sub']).pack(pady=20)
        self.current_node = None
        
    def _on_content_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _get_all_defined_vars(self):
        vars_set = set()
        if self.app.editor and self.app.editor.nodes:
            for node in self.app.editor.nodes.values():
                if node.type == 'set_var':
                    if node.data.get('var_name'): vars_set.add(node.data.get('var_name'))
                    for v in node.data.get('batch_vars', []):
                        if v.get('name'): vars_set.add(v.get('name'))
        return sorted(list(vars_set))

    def open_visual_offset_picker(self):
        if not self.current_node: return
        data = self.current_node.data
        if 'image' not in data or not data['image']:
            messagebox.showwarning("提示", "请先【截取图像】作为基准锚点。")
            return
        
        self.app.iconify()
        self.app.update()
        time.sleep(0.3) 

        try:
            full_screen = ImageGrab.grab()
            conf = float(data.get('confidence', 0.8))
            strategy = data.get('match_strategy', 'hybrid')
            
            res = VisionEngine.locate(data['image'], confidence=conf, timeout=1.0, strategy=strategy)
            
            if not res:
                self.app.deiconify()
                messagebox.showerror("错误", "❌ 在当前屏幕上未找到基准图！\n\n请确保目标图像在屏幕上可见，\n以便计算相对坐标偏移。")
                return

            anchor_cx = res.left + res.width / 2
            anchor_cy = res.top + res.height / 2
            
            top = tk.Toplevel(self.app)
            top.title("视觉偏移设定 (左键确定 / 右键取消)")
            top.attributes("-fullscreen", True, "-topmost", True, "-alpha", 1.0)
            top.config(cursor="crosshair")
            
            tk_img_full = ImageTk.PhotoImage(full_screen)
            
            cv = tk.Canvas(top, width=full_screen.width, height=full_screen.height, highlightthickness=0)
            cv.pack(fill='both', expand=True)
            cv.create_image(0, 0, image=tk_img_full, anchor='nw')
            
            cv.create_rectangle(res.left, res.top, res.left+res.width, res.top+res.height, outline='#00ff00', width=2, dash=(4,4))
            cv.create_line(anchor_cx-10, anchor_cy, anchor_cx+10, anchor_cy, fill='#00ff00', width=2)
            cv.create_line(anchor_cx, anchor_cy-10, anchor_cx, anchor_cy+10, fill='#00ff00', width=2)
            cv.create_text(res.left, res.top-20, text="基准锚点", fill='#00ff00', anchor='sw', font=('Segoe UI', 12, 'bold'))

            try: curr_off_x, curr_off_y = int(float(data.get('offset_x', 0))), int(float(data.get('offset_y', 0)))
            except: curr_off_x, curr_off_y = 0, 0
            
            init_target_x = anchor_cx + (curr_off_x * SCALE_X)
            init_target_y = anchor_cy + (curr_off_y * SCALE_Y)
            
            markers = []
            def draw_marker(mx, my, is_preview=True):
                for item in markers: cv.delete(item)
                markers.clear()
                markers.append(cv.create_line(anchor_cx, anchor_cy, mx, my, fill='yellow', width=2, dash=(2,2)))
                markers.append(cv.create_oval(mx-5, my-5, mx+5, my+5, outline='red', width=2))
                dx_log = int((mx - anchor_cx) / SCALE_X)
                dy_log = int((my - anchor_cy) / SCALE_Y)
                text = f"X偏移: {dx_log}\nY偏移: {dy_log}\n(左键确定)"
                txt_x, txt_y = mx + 15, my
                if txt_x > full_screen.width - 100: txt_x -= 120
                if txt_y > full_screen.height - 50: txt_y -= 60
                markers.append(cv.create_text(txt_x, txt_y, text=text, fill='red', anchor='nw', font=('Consolas', 12, 'bold'), justify='left'))

            draw_marker(init_target_x, init_target_y)

            def on_move(e): draw_marker(e.x, e.y)
            def on_left_click(e):
                final_off_x = int((e.x - anchor_cx) / SCALE_X)
                final_off_y = int((e.y - anchor_cy) / SCALE_Y)
                self._save('offset_x', final_off_x)
                self._save('offset_y', final_off_y)
                self.load_node(self.current_node) 
                self.app.log(f"📍 视觉偏移已设定: X={final_off_x}, Y={final_off_y}", "success")
                top.destroy()
                self.app.deiconify()
            def on_right_click(e):
                top.destroy()
                self.app.deiconify()
            
            cv.bind("<Motion>", on_move)
            cv.bind("<Button-1>", on_left_click)
            cv.bind("<Button-3>", on_right_click)
            top.bind("<Escape>", on_right_click)
            
            top.image = tk_img_full
            self.wait_window(top)

        except Exception as e:
            self.app.deiconify()
            messagebox.showerror("错误", f"启动视觉设定失败: {str(e)}")
            traceback.print_exc()

    def _delete_anchor(self, idx):
        if not self.current_node: return
        anchors = self.current_node.data.get('anchors', [])
        if 0 <= idx < len(anchors):
            del anchors[idx]
            self.current_node.data['anchors'] = anchors
            if not anchors:
                keys_to_clear = ['image', 'tk_image', 'b64', 'target_rect_x', 'target_rect_y', 'target_rect_w', 'target_rect_h']
                for k in keys_to_clear:
                    if k in self.current_node.data:
                        del self.current_node.data[k]
            self.load_node(self.current_node)

    def load_node(self, node):
        [w.destroy() for w in self.content.winfo_children()]; self.current_node = node; self._input("节点名称:", '_user_title', node.data.get('_user_title', node.title_text))
        ntype, data = node.type, node.data
        if ntype == 'wait': self._input("等待秒数:", 'seconds', data.get('seconds', 1.0))
        elif ntype == 'loop': self._input("循环次数:", 'count', data.get('count', 5))
        elif ntype == 'sequence':
            tk.Label(self.content,text="逻辑链设置:",bg=COLORS['bg_panel'],fg=COLORS['accent'],font=('Segoe UI',9,'bold')).pack(anchor='w',pady=(5,5))
            tk.Label(self.content,text="分支尝试数量:",bg=COLORS['bg_panel'],fg=COLORS['fg_text'],font=FONTS['small']).pack(anchor='w')
            current_steps=data.get('num_steps',3)
            def on_step_change(val):
                try:
                    count=max(1,min(20,int(val))); self._save('num_steps',count); node.outputs=[str(i) for i in range(1,count+1)]+['else']
                    node.h=PORT_START_Y+max(1,len(node.outputs))*PORT_STEP_Y; node.draw(); self.app.editor.redraw_links()
                except ValueError: pass
            e=tk.Entry(self.content,bg=COLORS['bg_app'],fg=COLORS['fg_text'],bd=0,insertbackground='white',font=FONTS['code']);e.insert(0,str(current_steps));e.pack(fill='x',pady=2,ipady=4)
            e.bind("<Return>",lambda ev:on_step_change(e.get()));e.bind("<FocusOut>",lambda ev:on_step_change(e.get()))
            tk.Label(self.content,text="按顺序执行分支。\n如果子节点成功(Yes/Found)，执行该分支后续流程。\n如果失败，尝试下一分支。",bg=COLORS['bg_panel'],fg=COLORS['fg_sub'],font=('Segoe UI',8),justify='left').pack(anchor='w',pady=10)
        elif ntype == 'image':
            if 'tk_image' in data and data['tk_image']: 
                self._draw_image_preview(data)
            elif ntype == 'image' and 'anchors' in data and data['anchors'] and not data.get('image'):
                tk.Label(self.content, text="⚠️ 尚未设定点击目标", bg=COLORS['bg_panel'], fg=COLORS['warning'], font=('Segoe UI', 10, 'bold'), pady=10).pack()

            if ntype == 'image':
                anchors = data.get('anchors', [])
                if anchors:
                    tk.Label(self.content, text="⚠️ 锚点模式下搜索范围由锚点决定", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small']).pack(pady=5)
                    anchors_frame = tk.Frame(self.content, bg=COLORS['bg_panel']); anchors_frame.pack(fill='x')
                    for i, anc in enumerate(anchors):
                        row = tk.Frame(anchors_frame, bg=COLORS['bg_card'], pady=2)
                        row.pack(fill='x', pady=1)
                        lbl_txt = f"锚点 {i+1} (主)" if i == 0 else f"锚点 {i+1} (验证)"
                        color = COLORS['success'] if i == 0 else COLORS['warning']
                        tk.Label(row, text=lbl_txt, bg=COLORS['bg_card'], fg=color, width=12, anchor='w').pack(side='left', padx=5)
                        del_btn = tk.Button(row, text="🗑️", command=lambda idx=i: self._delete_anchor(idx), bg=COLORS['bg_card'], fg=COLORS['danger'], bd=0, relief='flat', cursor='hand2')
                        del_btn.pack(side='right', padx=5)
                    btn_row = tk.Frame(self.content, bg=COLORS['bg_panel']); btn_row.pack(fill='x', pady=5)
                    tk.Button(btn_row, text="➕ 添加锚点", command=self.app.do_add_anchor, bg=COLORS['bg_header'], fg='white', bd=0, relief='flat').pack(side='left', fill='x', expand=True)

                else:
                    tk.Label(self.content, text="🔍 搜索范围:", bg=COLORS['bg_panel'], fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w', pady=(10, 0))
                    search_modes = {'fullscreen': '🖥️ 全屏幕', 'region': '🔲 自定义固定区域'}
                    search_mode_rev = {v: k for k, v in search_modes.items()}
                    current_mode = data.get('search_mode', 'fullscreen')
                    def on_search_mode_change(e):
                        new_mode = search_mode_rev.get(cb_mode.get())
                        self._save('search_mode', new_mode)
                        self.load_node(node) 
                    cb_mode = ttk.Combobox(self.content, values=list(search_modes.values()), state="readonly")
                    cb_mode.set(search_modes.get(current_mode, '🖥️ 全屏幕'))
                    cb_mode.pack(fill='x', pady=2)
                    cb_mode.bind("<<ComboboxSelected>>", on_search_mode_change)
                    if current_mode == 'region':
                        reg_frame = tk.Frame(self.content, bg=COLORS['bg_card'], padx=5, pady=5)
                        reg_frame.pack(fill='x', pady=2)
                        rx, ry = data.get('target_rect_x', 0), data.get('target_rect_y', 0)
                        rw, rh = data.get('target_rect_w', 0), data.get('target_rect_h', 0)
                        info_txt = f"当前区域: X={rx}, Y={ry}\nW={rw}, H={rh}"
                        if rw == 0: info_txt = "⚠️ 尚未设定区域"
                        tk.Label(reg_frame, text=info_txt, bg=COLORS['bg_card'], fg=COLORS['fg_sub'], justify='left', font=('Consolas', 8)).pack(side='left')
                        tk.Button(reg_frame, text="📐 框选区域", command=self.app.do_set_target, bg=COLORS['accent'], fg='#1a1a2e', bd=0, relief='flat', font=('Segoe UI', 8)).pack(side='right')

                    btn_frame = tk.Frame(self.content, bg=COLORS['bg_panel']); btn_frame.pack(fill='x', pady=5)
                    self.snip_button = tk.Button(btn_frame, text="📸 截取目标图片", command=self.app.do_snip, bg=COLORS['accent'], fg='#1a1a2e', bd=0, activebackground=COLORS['hover'], relief='flat', pady=5)
                    self.snip_button.pack(fill='x')

                    scroll_frame = tk.LabelFrame(self.content, text="📜 智能滚动 (找不到目标时)", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small'])
                    scroll_frame.pack(fill='x', pady=10, padx=1)
                    self._chk_in_frame(scroll_frame, "启用自动滚动", 'auto_scroll', data.get('auto_scroll', False))
                    if data.get('auto_scroll', False):
                        tk.Label(scroll_frame, text="滚动幅度 (负数向下):", bg=COLORS['bg_panel'], fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w')
                        e_scroll = tk.Entry(scroll_frame, bg=COLORS['bg_app'], fg=COLORS['fg_text'], bd=0)
                        e_scroll.insert(0, str(data.get('scroll_amount', -500)))
                        e_scroll.pack(fill='x', pady=2)
                        e_scroll.bind("<KeyRelease>", lambda ev: self._save('scroll_amount', e_scroll.get()))
                        if current_mode == 'region': tk.Label(scroll_frame, text="提示: 在区域模式下，将移动至区域中心滚动，且只对比区域内的画面变化。", bg=COLORS['bg_panel'], fg=COLORS['warning'], font=('Segoe UI', 8), justify='left', wraplength=220).pack(pady=5)
                        else: tk.Label(scroll_frame, text="提示: 程序将持续滚动，直到画面不再变化(到底部)或找到目标。", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=('Segoe UI', 8), justify='left', wraplength=220).pack(pady=5)

            else:
                 btn_frame=tk.Frame(self.content,bg=COLORS['bg_panel']);btn_frame.pack(fill='x',pady=5)
                 self.snip_button=tk.Button(btn_frame,text="📸 截取",command=self.app.do_snip,bg=COLORS['accent'],fg='#1a1a2e',bd=0,activebackground=COLORS['hover'],relief='flat',pady=5);self.snip_button.pack(side='left',fill='x',expand=True,padx=(0,2))
            
            if ntype == 'image':
                res_frame=tk.Frame(self.content,bg=COLORS['bg_panel']);res_frame.pack(fill='x',pady=(0,5));self.test_result_label=tk.Label(res_frame,bg=COLORS['bg_panel'],font=FONTS['small']);self.test_result_label.pack(fill='x',expand=True)
                self.test_match_button=tk.Button(res_frame,text="🧪 测试匹配",command=self.start_test_match,bg=COLORS['control'],fg='#1a1a2e',bd=0,relief='flat')
                self.test_match_button.pack(fill='x')

                self._combo("匹配策略:",'match_strategy',list(MATCH_STRATEGY_MAP.values()),MATCH_STRATEGY_MAP.get(data.get('match_strategy','hybrid')),lambda e:self._save('match_strategy',MATCH_STRATEGY_REVERSE.get(e.widget.get())))
                self._input("相似度 (0.1-1.0):",'confidence',data.get('confidence',0.9));self._chk("灰度模式匹配",'use_grayscale',data.get('use_grayscale',True))
                if HAS_OPENCV: self._chk("多尺度模板匹配",'use_multiscale',data.get('use_multiscale',True))
                
                self._input("最大保护超时 (秒):",'timeout',data.get('timeout',30.0))
                self._combo("执行动作:",'click_type',list(ACTION_MAP.values()),ACTION_MAP.get(data.get('click_type','click')),self.on_action_combo_select)
                
                tk.Label(self.content,text="点击坐标偏移 (像素):",bg=COLORS['bg_panel'],fg=COLORS['fg_text'],font=FONTS['small']).pack(anchor='w',pady=(5,0))
                off_frame = tk.Frame(self.content, bg=COLORS['bg_panel']); off_frame.pack(fill='x')
                tk.Label(off_frame, text="X:", bg=COLORS['bg_panel'], fg=COLORS['fg_sub']).pack(side='left')
                e_x=tk.Entry(off_frame,bg=COLORS['bg_app'],fg=COLORS['fg_text'],bd=0,width=6);e_x.insert(0,str(data.get('offset_x',0)));e_x.pack(side='left',padx=2)
                e_x.bind("<KeyRelease>",lambda ev:self._save('offset_x',e_x.get()))
                tk.Label(off_frame, text="Y:", bg=COLORS['bg_panel'], fg=COLORS['fg_sub']).pack(side='left')
                e_y=tk.Entry(off_frame,bg=COLORS['bg_app'],fg=COLORS['fg_text'],bd=0,width=6);e_y.insert(0,str(data.get('offset_y',0)));e_y.pack(side='left',padx=2)
                e_y.bind("<KeyRelease>",lambda ev:self._save('offset_y',e_y.get()))
                tk.Button(off_frame, text="🎯 视觉设定", command=self.open_visual_offset_picker, bg=COLORS['control'], fg='#1a1a2e', bd=0, relief='flat', font=('Segoe UI', 8)).pack(side='right', padx=5)

            else:
                self._input("扫描间隔 (秒):",'interval',data.get('interval',0.5));self._chk("✅ 启用此监视器",'enabled',data.get('enabled',True))

        elif ntype == 'if_img':
            tk.Label(self.content,text="需要同时满足的图像条件:",bg=COLORS['bg_panel'],fg=COLORS['fg_text'],font=FONTS['small']).pack(anchor='w',pady=(5,2))
            img_list_frame = tk.Frame(self.content, bg=COLORS['bg_panel']); img_list_frame.pack(fill='x', pady=5)
            images = data.get('images', [])
            if not images: tk.Label(img_list_frame, text="- 无条件 -", bg=COLORS['bg_panel'], fg=COLORS['fg_sub']).pack()
            else:
                for img_data in images:
                    item_frame = tk.Frame(img_list_frame, bg=COLORS['bg_card']); item_frame.pack(fill='x', pady=2)
                    if 'tk_image' in img_data and img_data['tk_image']: c = tk.Canvas(item_frame, width=120, height=67, bg='black', highlightthickness=0); c.pack(side='left', padx=5, pady=5); c.create_image(60, 33, image=img_data['tk_image'], anchor='center')
                    del_btn = tk.Button(item_frame, text="❌", command=lambda i=img_data.get('id'): self._delete_image_condition(i), bg=COLORS['danger'], fg='#1a1a2e', bd=0, relief='flat', font=('Segoe UI', 8, 'bold')); del_btn.pack(side='right', padx=5)
            btn_frame=tk.Frame(self.content,bg=COLORS['bg_panel']);btn_frame.pack(fill='x',pady=5)
            self.snip_button=tk.Button(btn_frame,text="➕ 添加截图条件",command=self.app.do_snip,bg=COLORS['accent'],fg='#1a1a2e',bd=0,activebackground=COLORS['hover'],relief='flat',pady=5);self.snip_button.pack(side='left',fill='x',expand=True,padx=(0,2))
            self.test_match_button=tk.Button(btn_frame,text="🧪 测试所有条件",command=self.start_test_match,bg=COLORS['accent'],fg='#1a1a2e',bd=0,activebackground=COLORS['hover'],relief='flat',pady=5);self.test_match_button.pack(side='left',fill='x',expand=True,padx=(2,0))
            res_frame=tk.Frame(self.content,bg=COLORS['bg_panel']);res_frame.pack(fill='x',pady=(0,5));self.test_result_label=tk.Label(res_frame,bg=COLORS['bg_panel'],font=FONTS['small']);self.test_result_label.pack(fill='x',expand=True)
            self._combo("匹配策略:",'match_strategy',list(MATCH_STRATEGY_MAP.values()),MATCH_STRATEGY_MAP.get(data.get('match_strategy','hybrid')),lambda e:self._save('match_strategy',MATCH_STRATEGY_REVERSE.get(e.widget.get())))
            self._input("相似度 (0.1-1.0):",'confidence',data.get('confidence',0.9));self._chk("灰度模式匹配",'use_grayscale',data.get('use_grayscale',True))
            if HAS_OPENCV: self._chk("多尺度模板匹配",'use_multiscale',data.get('use_multiscale',True))
        elif ntype == 'if_static':
            tk.Label(self.content, text="监控区域 (ROI):", bg=COLORS['bg_panel'], fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w', pady=(5,0))
            if 'roi_preview' in data and data['roi_preview']: c = tk.Canvas(self.content, width=240, height=135, bg='black', highlightthickness=0); c.pack(pady=5); c.create_image(120, 67, image=data['roi_preview'], anchor='center')
            self._btn("📸 截取监控区域", self.app.do_snip); self._input("静止持续时间 (秒):", 'duration', data.get('duration', 5.0))
            self._input("最大检测超时 (秒):", 'timeout', data.get('timeout', 20.0))
            self._input("检测间隔 (秒):", 'interval', data.get('interval', 1.0)); self._input("灵敏度 (0.9-1.0):", 'threshold', data.get('threshold', 0.98))
            tk.Label(self.content, text="灵敏度越高, 对变化的容忍度越低。\n推荐0.98, 表示98%相似即为静止。", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small'], justify='left').pack(anchor='w')
        elif ntype == 'if_sound':
            tk.Label(self.content, text="音频检测配置:", bg=COLORS['bg_panel'], fg=COLORS['accent'], font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(5,5))
            self._input("持续静音时间 (秒):", 'duration', data.get('duration', 3.0))
            self._input("静音阈值 (0.0-1.0):", 'threshold', data.get('threshold', 0.05))
            
            btn_text = "⏹ 停止监测" if self.is_monitoring_audio else "🔊 实时监测音量"
            btn_color = COLORS['danger'] if self.is_monitoring_audio else COLORS['control']
            self.monitor_btn = tk.Button(self.content, text=btn_text, command=self._toggle_audio_monitor, bg=btn_color, fg='#1a1a2e', bd=0, relief='flat', pady=5)
            self.monitor_btn.pack(fill='x', pady=5)
            
            tk.Label(self.content, text='''监测所有应用的最大音量峰值。
请播放视频以确定合适的阈值。''', bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small'], justify='left').pack(anchor='w', pady=2)
        elif ntype == 'if_file':
            tk.Label(self.content, text="文件检查配置:", bg=COLORS['bg_panel'], fg=COLORS['accent'], font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(5,5))
            self._input("文件路径 (支持变量):", 'file_path', data.get('file_path', ''))
            
            btn_frame = tk.Frame(self.content, bg=COLORS['bg_panel'])
            btn_frame.pack(fill='x', pady=5)
            self.test_file_button = tk.Button(btn_frame, text="🧪 测试文件是否存在", command=lambda: self._test_file_exist(data.get('file_path', '')), bg=COLORS['accent'], fg='#1a1a2e', bd=0, activebackground=COLORS['hover'], relief='flat', pady=5)
            self.test_file_button.pack(side='left', fill='x', expand=True)
            
            self.file_test_result = tk.Label(self.content, bg=COLORS['bg_panel'], font=FONTS['small'])
            self.file_test_result.pack(fill='x', pady=(0,5))
            
            tk.Label(self.content, text='''支持使用变量，如 {var_name} 
文件存在返回Yes，否则返回No''', bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small'], justify='left').pack(anchor='w', pady=2)
        elif ntype=='check_sensor':
                tk.Label(self.content,text="此节点类型已被移除",bg=COLORS['bg_panel'],fg=COLORS['danger']).pack(pady=10)
        elif ntype=='mouse':
            mode=data.get('mouse_mode','click');self._combo("事件类型:",'mouse_mode',list(MOUSE_MODE_MAP.values()),MOUSE_MODE_MAP.get(mode),self.on_mouse_mode_select)
            if mode=='click':self._combo("点击类型:",'click_type',list(ACTION_MAP.values()),ACTION_MAP.get(data.get('click_type','click')),self.on_action_combo_select)
            elif mode=='move':self._input("目标 X 坐标:",'x',data.get('x',0));self._input("目标 Y 坐标:",'y',data.get('y',0));self._btn("📍 拾取屏幕坐标",self._pick_pos)
            elif mode=='scroll':self._input("滚动量 (负数向下):",'amount',data.get('amount',-500))
        elif ntype=='keyboard':
            self._combo("输入类型:",'kb_mode',['text','key'],data.get('kb_mode','text'),lambda e:[self._save('kb_mode',e.widget.get()),self.load_node(node)])
            if data.get('kb_mode','text')=='text': self._input("输入文本:",'text',data.get('text',''));self._chk("模拟打字 (慢速)",'slow_type',data.get('slow_type',False));self._chk("输入后按回车",'press_enter',data.get('press_enter',False))
            else: self._input("按键 (如: enter, ctrl+c):",'key_name',data.get('key_name','enter'))
        elif ntype=='web': self._input("要打开的网址:",'url',data.get('url',''))
        
        elif ntype == 'set_var':
            tk.Label(self.content, text="批量设置变量 (每行一个):", bg=COLORS['bg_panel'], fg=COLORS['accent'], font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(5,5))
            tk.Label(self.content, text="格式: 变量名=值", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=FONTS['small']).pack(anchor='w')
            
            text_val = ""
            batch_vars = data.get('batch_vars', [])
            if not batch_vars and data.get('var_name'):
                 batch_vars = [{'name': data.get('var_name'), 'value': data.get('var_value', '')}]
            
            for item in batch_vars:
                if item.get('name'):
                    text_val += f"{item.get('name')}={item.get('value','')}\n"

            txt = tk.Text(self.content, height=8, bg=COLORS['bg_app'], fg=COLORS['fg_text'], font=FONTS['code'], bd=0, padx=5, pady=5)
            txt.insert('1.0', text_val)
            txt.pack(fill='x', pady=5)
            
            def save_batch_text():
                raw = txt.get('1.0', 'end').strip()
                new_list = []
                for line in raw.split('\n'):
                    if '=' in line:
                        parts = line.split('=', 1)
                        new_list.append({'name': parts[0].strip(), 'value': parts[1].strip()})
                self._save('batch_vars', new_list)
                self.app.log("变量设置已保存", "success")

            tk.Button(self.content, text="💾 保存并生效", command=save_batch_text, bg=COLORS['success'], fg='#1a1a2e', bd=0, relief='flat', pady=5).pack(fill='x')
            tk.Label(self.content, text="提示: 设置默认值直接写入即可 (如 count=0)。", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=('Segoe UI', 8)).pack(anchor='w', pady=5)

        elif ntype == 'var_switch':
            tk.Label(self.content, text="变量检测与分流:", bg=COLORS['bg_panel'], fg=COLORS['accent'], font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(5,5))
            
            if 'var_name' in data:
                existing_vars = self._get_all_defined_vars()
                
                tk.Label(self.content, text="变量名称:", bg=COLORS['bg_panel'], fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w')
                cb_name = ttk.Combobox(self.content, values=existing_vars)
                cb_name.set(data.get('var_name', ''))
                cb_name.pack(fill='x', pady=2)
                cb_name.bind("<<ComboboxSelected>>", lambda e: self._save('var_name', cb_name.get()))
                cb_name.bind("<KeyRelease>", lambda e: self._save('var_name', cb_name.get()))
                
                self._combo("判断条件:", 'operator', list(VAR_OP_MAP.values()), VAR_OP_MAP.get(data.get('operator','=')), 
                            lambda e: [self._save('operator', VAR_OP_REVERSE.get(e.widget.get())), self.load_node(node)])
                
                if data.get('operator','=') in ['=', '!=']:
                    self._input("目标值:", 'var_value', data.get('var_value', 'done'))
                
                tk.Label(self.content, text="说明: 单变量判断模式，支持等于、不等于、存在、不存在四种判断。", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=('Segoe UI', 8), justify='left', wraplength=260).pack(pady=5)
            else:
                tk.Label(self.content, text="批量变量分流逻辑:", bg=COLORS['bg_panel'], fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w', pady=(5,5))
                
                e_list = tk.Entry(self.content, bg=COLORS['bg_app'], fg=COLORS['fg_text'], bd=0, font=FONTS['code'])
                e_list.insert(0, data.get('var_list', ''))
                e_list.pack(fill='x', pady=2, ipady=4)
                e_list.bind("<KeyRelease>", lambda e: self._save('var_list', e_list.get()))
                e_list.bind("<FocusOut>", lambda e: self._save('var_list', e_list.get()))
                
                tk.Label(self.content, text="分流条件 (Case):", bg=COLORS['bg_panel'], fg=COLORS['fg_text'], font=FONTS['small']).pack(anchor='w', pady=(10,2))
                
                cases_frame = tk.Frame(self.content, bg=COLORS['bg_panel'])
                cases_frame.pack(fill='x')
                
                cases = data.get('cases', [])
                
                def update_cases():
                    node.outputs = [c['id'] for c in cases] + ['else']
                    node.h = PORT_START_Y + max(1, len(node.outputs)) * PORT_STEP_Y
                    node.draw()
                    self.app.editor.redraw_links()
                    self.load_node(node)
                
                def add_case():
                    new_id = f"case_{len(cases)}_{int(time.time())}"
                    cases.append({'value': 'new_val', 'id': new_id})
                    data['cases'] = cases
                    update_cases()

                def remove_case(idx):
                    del cases[idx]
                    data['cases'] = cases
                    update_cases()
                
                def update_case_val(idx, val):
                    cases[idx]['value'] = val
                    node.draw()

                for i, case in enumerate(cases):
                    row = tk.Frame(cases_frame, bg=COLORS['bg_card'], pady=2)
                    row.pack(fill='x', pady=1)
                    tk.Label(row, text=f"当所有变量==", bg=COLORS['bg_card'], fg=COLORS['fg_sub'], font=('Segoe UI', 8)).pack(side='left', padx=2)
                    
                    e_val = tk.Entry(row, bg=COLORS['bg_app'], fg=COLORS['fg_text'], bd=0, width=10)
                    e_val.insert(0, case.get('value', ''))
                    e_val.pack(side='left', fill='x', expand=True, padx=2)
                    e_val.bind("<KeyRelease>", lambda e, idx=i: update_case_val(idx, e.widget.get()))
                    
                    tk.Button(row, text="❌", command=lambda idx=i: remove_case(idx), bg=COLORS['bg_card'], fg=COLORS['danger'], bd=0, relief='flat').pack(side='right', padx=2)

                tk.Button(self.content, text="➕ 添加分流条件", command=add_case, bg=COLORS['bg_header'], fg='white', bd=0, relief='flat').pack(fill='x', pady=5)
                tk.Label(self.content, text="说明: 如果列表中所有变量的值都等于设定值，则执行对应端口。否则执行 Else。", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=('Segoe UI', 8), justify='left', wraplength=260).pack(pady=5)

        self._draw_output_help(node)

    def _draw_output_help(self, node):
        ntype = node.type
        outputs = NODE_CONFIG.get(ntype, {}).get('outputs', [])
        
        if ntype in ['sequence', 'var_switch']:
            if ntype == 'sequence': outputs = ['(数字)', 'else']
            if ntype == 'var_switch': outputs = ['(条件分支)', 'else']
        
        if not outputs: return
        help_frame = tk.LabelFrame(self.content, text="📤 输出逻辑说明", bg=COLORS['bg_panel'], fg=COLORS['fg_sub'], font=('Segoe UI', 8, 'bold'))
        help_frame.pack(fill='x', pady=(15, 5), padx=2, ipadx=5, ipady=5)
        docs = NODE_OUTPUT_DOCS.get(ntype, {})
        for port in outputs:
            display_name = PORT_TRANSLATION.get(port, port)
            description = docs.get(port, "执行后续流程")
            
            if port == '(数字)': display_name, description = "分支 1..N", "对应步骤成功(Yes/Found)时触发"
            if port == '(条件分支)': display_name, description = "Case 1..N", "所有变量均满足对应值时触发"

            icon_color = COLORS['success']
            if port in ['no', 'timeout', 'exit', 'else']: icon_color = COLORS['danger']
            if port == 'out': icon_color = COLORS['accent']
            row = tk.Frame(help_frame, bg=COLORS['bg_panel']); row.pack(fill='x', pady=2)
            tk.Label(row, text=f"● {display_name}", fg=icon_color, bg=COLORS['bg_panel'], font=('Segoe UI', 9, 'bold'), width=10, anchor='w').pack(side='left')
            tk.Label(row, text=description, fg=COLORS['fg_sub'], bg=COLORS['bg_panel'], font=('Segoe UI', 8), justify='left', wraplength=180, anchor='w').pack(side='left', fill='x', expand=True)

    def _delete_image_condition(self, image_id_to_delete):
        if not self.current_node or not image_id_to_delete: return
        images = self.current_node.data.get('images', [])
        self.current_node.data['images'] = [img for img in images if img.get('id') != image_id_to_delete]
        self.load_node(self.current_node)
    def _draw_image_preview(self,data):
        c=tk.Canvas(self.content,width=240,height=135,bg='black',highlightthickness=0,cursor="crosshair");c.pack(pady=5);c.create_image(120,67,image=data['tk_image'],anchor='center')
        rx,ry=data.get('relative_click_pos',(0.5,0.5));cx,cy=240*rx,135*ry;c.create_line(cx-10,cy,cx+10,cy,fill=COLORS['marker'],width=2);c.create_line(cx,cy-10,cx,cy+10,fill=COLORS['marker'],width=2)
        c.bind("<Button-1>",lambda e:[self._save('relative_click_pos',(max(0,min(1,e.x/240)),max(0,min(1,e.y/135)))),self.load_node(self.current_node)])
    def _chk_in_frame(self, parent, txt, key, val):
        var = tk.BooleanVar(value=val)
        tk.Checkbutton(parent, text=txt, variable=var, bg=COLORS['bg_panel'], fg=COLORS['fg_text'], selectcolor=COLORS['bg_app'], activebackground=COLORS['bg_panel'], borderwidth=0, highlightthickness=0, command=lambda: [self._save(key, var.get()), self.load_node(self.current_node)]).pack(anchor='w', pady=2)
    def on_mouse_mode_select(self,event):
        self._save('mouse_mode',MOUSE_MODE_REVERSE.get(event.widget.get())); self.load_node(self.current_node)
    def on_action_combo_select(self,event):
        self._save('click_type',ACTION_MAP_REVERSE.get(event.widget.get()))
    def _input(self,label,key,val): tk.Label(self.content,text=label,bg=COLORS['bg_panel'],fg=COLORS['fg_text'],font=FONTS['small']).pack(anchor='w',pady=(5,0));e=tk.Entry(self.content,bg=COLORS['bg_app'],fg=COLORS['fg_text'],bd=0,insertbackground='white',font=FONTS['code']);e.insert(0,str(val));e.pack(fill='x',pady=2,ipady=4);e.bind("<KeyRelease>",lambda ev:self._save(key,e.get()));e.bind("<FocusOut>",lambda ev:self._save(key,e.get()))
    def _combo(self,label,key,values,val,cmd=None):
        tk.Label(self.content,text=label,bg=COLORS['bg_panel'],fg=COLORS['fg_text'],font=FONTS['small']).pack(anchor='w',pady=(5,0));cb=ttk.Combobox(self.content,values=values,state="readonly");cb.set(val);cb.pack(fill='x',pady=2)
        if not cmd: cmd=lambda e,w=cb:self._save(key,w.get())
        cb.bind("<<ComboboxSelected>>",cmd)
    def _btn(self,txt,cmd): tk.Button(self.content,text=txt,command=cmd,bg=COLORS['accent'],fg='#1a1a2e',bd=0,activebackground=COLORS['hover'],relief='flat',pady=5).pack(fill='x',pady=5)
    def _chk(self,txt,key,val): var=tk.BooleanVar(value=val);tk.Checkbutton(self.content,text=txt,variable=var,bg=COLORS['bg_panel'],fg=COLORS['fg_text'],selectcolor=COLORS['bg_app'],activebackground=COLORS['bg_panel'],borderwidth=0,highlightthickness=0,command=lambda:self._save(key,var.get())).pack(anchor='w',pady=5)
    def _save(self,key,val): (self.current_node and self.current_node.update_data(key,val))
    def _pick_pos(self): self.app.pick_coordinate()
    def start_test_match(self):
        if not self.current_node: return
        if self.current_node.type == 'if_img' and not self.current_node.data.get('images'): return
        if self.current_node.type == 'image' and 'image' not in self.current_node.data: return
        if self.current_node.type == 'if_file':
            self._test_file_exist()
            return
        self.test_match_button.config(text="测试中...",state="disabled");self.test_result_label.config(text="")
        threading.Thread(target=self._test_match_worker,daemon=True).start()
    def _test_match_worker(self):
        node, data = self.current_node, self.current_node.data
        test_results = []
        try:
            self.app.iconify(); time.sleep(0.4)
            conf, strategy = float(data.get('confidence',0.9)), data.get('match_strategy','hybrid')
            if node.type == 'if_img':
                images_to_test = data.get('images', [])
                if not images_to_test: test_results.append(('⚠️ 无图像', False))
                else:
                    haystack = VisionEngine.capture_screen()
                    if haystack is None: test_results.append(('❌ 截图失败', False))
                    else:
                        for i, img_data in enumerate(images_to_test):
                            if not (needle := img_data.get('image')): continue
                            res, _ = VisionEngine._advanced_match(needle, haystack, conf, None, data.get('use_grayscale',True), data.get('use_multiscale',True), self.app.core.scaling_ratio, strategy)
                            test_results.append((f"条件 {i+1}", bool(res)))
            else:
                if (needle := data.get('image')):
                    res, _ = VisionEngine._advanced_match(needle, VisionEngine.capture_screen(), conf, None, data.get('use_grayscale',True), data.get('use_multiscale',True), self.app.core.scaling_ratio, strategy)
                    test_results.append(("目标图像", bool(res)))
        except Exception as e: test_results.append((f"测试出错: {e}", False))
        finally: self.app.after(0, self._update_test_ui, test_results)
    def _update_test_ui(self, results):
        self.app.deiconify();self.app.after(100,self.app.lift)
        if not results:
            self.test_result_label.config(text="❌ 未执行测试", fg=COLORS['danger']); self.test_match_button.config(bg=COLORS['accent']); return
        all_passed = all(res[1] for res in results)
        if len(results) > 1:
            summary = "✅ 所有条件通过" if all_passed else "❌ 部分条件未通过"
            details = ", ".join([f"{name}: {'✔️' if passed else '✖️'}" for name, passed in results])
            self.test_result_label.config(text=f"{summary}\n({details})", wraplength=250, justify='left', fg=COLORS['success'] if all_passed else COLORS['danger'])
        else:
            name, passed = results[0]
            self.test_result_label.config(text=f"✅ {name} 找到" if passed else f"❌ {name} 未找到",fg=COLORS['success'] if passed else COLORS['danger'])
        self.test_match_button.config(bg=COLORS['success'] if all_passed else COLORS['danger'])
        self.app.after(5000,self._reset_test_ui)
    def _reset_test_ui(self):
        if self.test_match_button and self.test_match_button.winfo_exists():
            text = "🧪 测试所有条件" if self.current_node and self.current_node.type == 'if_img' else "🧪 测试匹配"
            self.test_match_button.config(text=text, state="normal", bg=COLORS['accent'])
        if self.test_result_label and self.test_result_label.winfo_exists(): self.test_result_label.config(text="")
    
    def _test_file_exist(self):
        if not self.current_node or self.current_node.type != 'if_file':
            return
        file_path = self.current_node.data.get('file_path', '')
        file_path = self.app.core._replace_variables(file_path)
        
        try:
            if os.path.exists(file_path):
                self.test_result_label.config(text=f"✅ 文件存在: {file_path}", fg=COLORS['success'])
            else:
                self.test_result_label.config(text=f"❌ 文件不存在: {file_path}", fg=COLORS['danger'])
        except Exception as e:
            self.test_result_label.config(text=f"❌ 测试出错: {e}", fg=COLORS['danger'])
    
    def _toggle_audio_monitor(self):
        if self.is_monitoring_audio:
            self.is_monitoring_audio = False
            self.monitor_btn.config(text="🔊 实时监测音量", bg=COLORS['control'])
            self.app.log("🔊 音量监测已停止", "info")
        else:
            self.is_monitoring_audio = True
            self.monitor_btn.config(text="⏹ 停止监测", bg=COLORS['danger'])
            threading.Thread(target=self._audio_monitor_thread, daemon=True).start()

    def _audio_monitor_thread(self):
        self.app.log("🔊 开始监测所有会话音量...", "info")
        try:
            try: comtypes.CoInitialize()
            except: pass
            while self.is_monitoring_audio:
                if not self.app.property_panel.winfo_exists(): break
                vol = AudioEngine.get_max_audio_peak()
                if vol > 0.001:
                    self.app.log(f"  📊 最大音量峰值: {vol:.4f}", "info")
                time.sleep(0.5)
        except Exception as e:
            self.app.log(f"⚠️ 监测出错: {e}", "error")
            self.is_monitoring_audio = False
            if self.monitor_btn.winfo_exists():
                self.monitor_btn.config(text="🔊 实时监测音量", bg=COLORS['control'])

# --- 9. 主程序入口 ---
class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("Qflow 1.0"); self.geometry("1400x900"); self.configure(bg=COLORS['bg_app'])
        self.protocol("WM_DELETE_WINDOW", self._on_closing); self.core = AutomationCore(self.log, self); self.log_q = queue.Queue()
        self.drag_node_type, self.drag_ghost = None, None
        self._setup_styles(); self._setup_ui(); self.after(100, self._poll_log)
    def _setup_styles(self): 
        s=ttk.Style();s.theme_use('clam')
        s.configure('TCombobox',fieldbackground=COLORS['bg_card'],background=COLORS['bg_card'],foreground=COLORS['fg_text'],arrowcolor=COLORS['fg_text'],selectbackground=COLORS['bg_app'],selectforeground=COLORS['accent'],bordercolor=COLORS['bg_header'],lightcolor=COLORS['bg_card'],darkcolor=COLORS['bg_card']);s.map('TCombobox',fieldbackground=[('readonly',COLORS['bg_card'])])
        s.configure("Vertical.TScrollbar", gripcount=0, background=COLORS['bg_header'], darkcolor=COLORS['bg_app'], lightcolor=COLORS['bg_app'], troughcolor=COLORS['bg_app'], bordercolor=COLORS['bg_app'], arrowcolor='white')
    def _setup_ui(self):
        top=tk.Frame(self,bg=COLORS['bg_app'],height=60);top.pack(fill='x',pady=10,padx=20);tk.Label(top,text="Qflow 1.0",font=('Impact',24),bg=COLORS['bg_app'],fg=COLORS['accent']).pack(side='left')
        btn_box=tk.Frame(top,bg=COLORS['bg_app']);btn_box.pack(side='left',padx=30);self._flat_btn(btn_box,"📂 打开",self.load);self._flat_btn(btn_box,"💾 保存",self.save);self._flat_btn(btn_box,"🗑️ 清空",self.clear)
        
        self.run_btn_frame = tk.Frame(top, bg=COLORS['bg_app'])
        self.run_btn_frame.pack(side='right')
        
        self.btn_run = tk.Button(self.run_btn_frame, text="▶ 启动 (Default)", command=lambda: self.toggle_run(None), bg=COLORS['success'], fg='#1a1a2e', font=('Segoe UI', 12, 'bold'), padx=15, bd=0, relief='flat')
        self.btn_run.pack(side='left', fill='y')
        
        self.btn_run_menu = tk.Menubutton(self.run_btn_frame, text="▼", bg=COLORS['success'], fg='#1a1a2e', font=('Segoe UI', 10, 'bold'), padx=5, bd=0, relief='flat', direction='below')
        self.run_menu = tk.Menu(self.btn_run_menu, tearoff=0, bg=COLORS['bg_card'], fg=COLORS['fg_text'], font=FONTS['small'])
        self.run_menu.add_command(label="▶ 从头开始运行 (F5)", command=lambda: self.toggle_run(None))
        self.run_menu.add_command(label="▶ 从选中节点开始运行", command=self._run_selected)
        self.btn_run_menu.config(menu=self.run_menu)
        self.btn_run_menu.pack(side='left', fill='y', padx=(1,0))

        paned=tk.PanedWindow(self,orient='horizontal',bg=COLORS['bg_app'],sashwidth=6,sashrelief='flat',bd=0);paned.pack(fill='both',expand=True,padx=10,pady=5)
        toolbox=tk.Frame(paned,bg=COLORS['bg_sidebar'],width=200);toolbox.pack_propagate(False)
        
        self._add_group(toolbox,"事件 / 逻辑",['start','end','set_var','var_switch'])
        self._add_group(toolbox,"控制流程",['sequence','if_img','if_static','if_sound','if_file','loop','wait'])
        self._add_group(toolbox,"动作操作",['mouse','keyboard','image','web'])
        paned.add(toolbox,minsize=180)
        
        try: self.editor=FlowEditor(paned,self,splinesteps=12)
        except: self.editor=FlowEditor(paned,self)
        paned.add(self.editor,minsize=400,stretch="always")
        
        # 缩小属性面板宽度
        self.property_panel=PropertyPanel(paned,self)
        paned.add(self.property_panel,minsize=200,width=240)
        
        self.log_panel=LogPanel(self); self.log_panel.pack(side='bottom',fill='x',pady=(0,10),padx=10)
        self.editor.add_node('start',100,100);self._setup_hotkeys()

    def _flat_btn(self,p,txt,cmd): tk.Button(p,text=txt,command=cmd,bg=COLORS['bg_header'],fg=COLORS['fg_text'],bd=0,padx=15,pady=5,activebackground=COLORS['bg_panel'],relief='flat').pack(side='left',padx=5)
    def _add_group(self, p, title, items):
        tk.Label(p, text=title, bg=COLORS['bg_sidebar'], fg=COLORS['fg_sub'], font=('Segoe UI', 9, 'bold'), pady=8).pack(anchor='w', padx=10)
        for t in items:
            f = tk.Frame(p, bg=COLORS['bg_sidebar'], pady=1); f.pack(fill='x', padx=10)
            lbl = tk.Label(f, text=NODE_CONFIG[t]['title'], bg=COLORS['bg_card'], fg=COLORS['fg_text'], anchor='w', padx=10, pady=6, cursor="hand2")
            lbl.pack(fill='x', pady=1); lbl.bind("<ButtonPress-1>", lambda e, ntype=t: self.on_sidebar_drag_start(e, ntype)); lbl.bind("<B1-Motion>", self.on_sidebar_drag_motion); lbl.bind("<ButtonRelease-1>", self.on_sidebar_drag_release)
    def on_sidebar_drag_start(self,event,ntype): self.drag_node_type=ntype;self.drag_ghost=tk.Toplevel(self);self.drag_ghost.overrideredirect(True);self.drag_ghost.attributes("-topmost",True,"-alpha",0.7);tk.Label(self.drag_ghost,text=NODE_CONFIG[ntype]['title'],bg=COLORS['accent'],fg='#1a1a2e',padx=10,pady=5).pack();self.drag_ghost.geometry(f"+{event.x_root+10}+{event.y_root+10}")
    def on_sidebar_drag_motion(self,event): (self.drag_ghost and self.drag_ghost.geometry(f"+{event.x_root+10}+{event.y_root+10}"))
    def on_sidebar_drag_release(self,event):
        if self.drag_ghost: self.drag_ghost.destroy();self.drag_ghost=None
        cx_root,cy_root,cw,ch=self.editor.winfo_rootx(),self.editor.winfo_rooty(),self.editor.winfo_width(),self.editor.winfo_height()
        if cx_root<=event.x_root<=cx_root+cw and cy_root<=event.y_root<=cy_root+ch:
            canvas_x,canvas_y=event.x_root-cx_root,event.y_root-cy_root
            log_x,log_y=self.editor.canvasx(canvas_x)/self.editor.zoom,self.editor.canvasy(canvas_y)/self.editor.zoom
            if self.drag_node_type: self.editor.add_node(self.drag_node_type,round(log_x/GRID_SIZE)*GRID_SIZE,round(log_y/GRID_SIZE)*GRID_SIZE)
        self.drag_node_type=None
    def do_snip(self): self.iconify();self.update();self.after(400, lambda: self._start_snip_overlay(mode='normal'))
    def do_add_anchor(self): self.iconify();self.update();self.after(400, lambda: self._start_snip_overlay(mode='add_anchor'))
    def do_set_target(self): self.iconify();self.update();self.after(400, lambda: self._start_snip_overlay(mode='set_target'))

    def _start_snip_overlay(self, mode='normal'):
        top=tk.Toplevel(self);top.attributes("-fullscreen",True,"-alpha",0.3,"-topmost",True);top.configure(cursor="cross");c=tk.Canvas(top,bg="black",highlightthickness=0);c.pack(fill='both',expand=True);
        
        s, r = [0,0], [None]
        
        info_lbl = tk.Label(top, text="", font=('Segoe UI', 16, 'bold'), fg='white', bg='black')
        info_lbl.place(x=50, y=50)

        if mode == 'add_anchor': info_lbl.config(text="请框选一个【锚点/特征】区域 (ESC取消)", fg='#a6e3a1')
        elif mode == 'set_target': info_lbl.config(text="请框选【最终目标】区域 (ESC取消)", fg='#f38ba8')
        else: info_lbl.config(text="请框选区域 (ESC取消)")
        
        def dn(e): s[0], s[1] = e.x, e.y; color = 'green' if mode=='add_anchor' else 'red'; r[0] = c.create_rectangle(e.x, e.y, e.x, e.y, outline=color, width=2)
        def mv(e): (r[0] and c.coords(r[0], s[0], s[1], e.x, e.y))
        def up(e):
            x1, x2 = sorted((s[0], e.x)); y1, y2 = sorted((s[1], e.y))
            if x2-x1 < 5 or y2-y1 < 5: 
                if r[0]: c.delete(r[0]); r[0] = None
                return
            
            rect = (int(x1*SCALE_X), int(y1*SCALE_Y), int(x2*SCALE_X), int(y2*SCALE_Y))
            top.destroy()
            
            if mode == 'add_anchor': self.after(200, lambda: self._internal_add_anchor(rect))
            elif mode == 'set_target': self.after(200, lambda: self._internal_set_target(rect))
            else: self.after(200, lambda: self._internal_capture(rect))

        c.bind("<ButtonPress-1>",dn);c.bind("<B1-Motion>",mv);c.bind("<ButtonRelease-1>",up);
        top.bind("<Escape>", lambda e: [top.destroy(), self.deiconify()])
        self.wait_window(top)

    def _internal_capture(self, rect):
        x1, y1, x2, y2 = rect
        try:
            img=ImageGrab.grab(bbox=(x1,y1,x2,y2));self.deiconify();self.lift()
            if (node:=self.property_panel.current_node):
                if node.type == 'if_img':
                    if 'images' not in node.data: node.data['images'] = []
                    node.data['images'].append({'id': str(uuid.uuid4()),'image': img,'tk_image': ImageUtils.make_thumb(img, size=(120, 67)),'b64': ImageUtils.img_to_b64(img)})
                elif node.type == 'if_static':
                    node.update_data('roi', (x1, y1, x2, y2)); node.update_data('roi_preview', ImageUtils.make_thumb(img)); node.update_data('b64_preview', ImageUtils.img_to_b64(img))
                else:
                    node.update_data('image',img);node.update_data('tk_image',ImageUtils.make_thumb(img));node.update_data('b64',ImageUtils.img_to_b64(img))
                self.property_panel.load_node(node); self.log("🖼️ 截取成功", 'success')
        except Exception as e: self.deiconify();self.log(f"截图失败: {e}", "error")

    def _internal_add_anchor(self, rect):
        try:
            img = ImageGrab.grab(bbox=rect)
            self.deiconify(); self.lift()
            if (node := self.property_panel.current_node) and node.type == 'image':
                if 'anchors' not in node.data or not node.data['anchors']:
                     keys_to_clear = ['image', 'tk_image', 'b64', 'target_rect_x', 'target_rect_y', 'target_rect_w', 'target_rect_h']
                     for k in keys_to_clear:
                        if k in node.data:
                            del node.data[k]
                if 'anchors' not in node.data: node.data['anchors'] = []
                node.data['anchors'].append({
                    'id': str(uuid.uuid4()),
                    'image': img,
                    'b64': ImageUtils.img_to_b64(img),
                    'rect_x': rect[0], 'rect_y': rect[1]
                })
                self.property_panel.load_node(node)
                self.log(f"⚓ 锚点添加成功 (当前共 {len(node.data['anchors'])} 个)", 'success')
        except Exception as e: self.deiconify(); self.log(f"锚点截图失败: {e}", "error")

    def _internal_set_target(self, rect):
        try:
            img = ImageGrab.grab(bbox=rect)
            self.deiconify(); self.lift()
            if (node := self.property_panel.current_node) and node.type == 'image':
                node.update_data('image', img)
                node.update_data('tk_image', ImageUtils.make_thumb(img))
                node.update_data('b64', ImageUtils.img_to_b64(img))
                node.update_data('target_rect_x', rect[0])
                node.update_data('target_rect_y', rect[1])
                node.update_data('target_rect_w', rect[2] - rect[0])
                node.update_data('target_rect_h', rect[3] - rect[1])
                self.property_panel.load_node(node)
                self.log("🎯 目标区域已设定", 'success')
        except Exception as e: self.deiconify(); self.log(f"目标截图失败: {e}", "error")

    def pick_coordinate(self): self.iconify();self.update();self.after(400, self._start_coordinate_overlay)
    def _start_coordinate_overlay(self):
        top=tk.Toplevel(self);top.attributes("-fullscreen",True,"-alpha",0.1,"-topmost",True);top.configure(cursor="none");c=tk.Canvas(top,bg="white",highlightthickness=0);c.pack(fill='both',expand=True)
        w,h=top.winfo_screenwidth(),top.winfo_screenheight();h_line,v_line=c.create_line(0,0,w,0,fill="red",width=1),c.create_line(0,0,0,h,fill="red",width=1);lbl_bg=c.create_rectangle(0,0,0,0,fill="#ffffdd",outline="black");lbl=c.create_text(0,0,text="",fill="black",anchor="nw",font=("Consolas",10))
        def on_move(e): c.coords(h_line,0,e.y,w,e.y); c.coords(v_line,e.x,0,e.x,h); txt=f"X:{int(e.x*SCALE_X)}, Y:{int(e.y*SCALE_Y)}"; c.itemconfig(lbl,text=txt); bbox=c.bbox(lbl); c.coords(lbl,e.x+15,e.y+15); c.coords(lbl_bg,e.x+13,e.y+13,e.x+17+bbox[2]-bbox[0],e.y+17+bbox[3]-bbox[1])
        c.bind("<Motion>",on_move);c.bind("<Button-1>",lambda e:[top.destroy(),self.after(200,lambda:self._apply_picked_coordinate(int(e.x*SCALE_X),int(e.y*SCALE_Y)))]);c.bind("<Button-3>",lambda e:[top.destroy(),self.deiconify()]);self.wait_window(top)
    def _apply_picked_coordinate(self,x,y): self.deiconify();self.lift(); (self.property_panel.current_node and (self.property_panel.current_node.update_data('x',x),self.property_panel.current_node.update_data('y',y),self.property_panel.load_node(self.property_panel.current_node),self.log(f"📍 坐标: ({x},{y})", "info")))
    
    def _run_selected(self):
        if not self.editor.selected_node_id:
            messagebox.showwarning("提示", "请先在画布中选中一个节点！")
            return
        self.toggle_run(self.editor.selected_node_id)

    def toggle_run(self, start_id=None): 
        if self.core.running:
            self.core.stop()
            self.btn_run.config(text="▶ 启动 (Default)", bg=COLORS['success'])
            self.btn_run_menu.config(state='normal')
        else:
            self.core.load_project(self.editor.get_data())
            self.core.start(start_node_id=start_id)
            self.btn_run.config(text="⏹ 停止运行", bg=COLORS['danger'])
            self.btn_run_menu.config(state='disabled')

    def reset_ui_state(self): 
        self.core.running=False
        self.btn_run.config(text="▶ 启动 (Default)", bg=COLORS['success'])
        self.btn_run_menu.config(state='normal')
        [self.update_sensor_visual_safe(nid,False) for nid in self.core.sensor_states]
        if hasattr(self, 'log_panel') and self.log_panel:
            self.log_panel.pack(side='bottom', fill='x', pady=(0,10), padx=10)
    def update_sensor_visual_safe(self,nid,active): self.after(0,self.update_sensor_visual,nid,active)
    def update_sensor_visual(self,nid,active): (nid in self.editor.nodes and self.editor.nodes[nid].set_sensor_active(active))
    def log(self,msg, level='info'): self.log_q.put((msg, level))
    def _poll_log(self):
        while not self.log_q.empty():
            item = self.log_q.get()
            if isinstance(item, tuple): self.log_panel.add_log(item[0], item[1])
            else: self.log_panel.add_log(str(item), 'info')
        self.after(100,self._poll_log)
    def highlight_node_safe(self,nid,status=None): self.after(0,self.highlight_node,nid,status)
    
    def highlight_node(self, nid, status=None):
        self.editor.delete("hl")
        if nid and nid in self.editor.nodes and (col:=COLORS.get(f"hl_{status}")): 
            n,z=self.editor.nodes[nid],self.editor.zoom;vx,vy,vw,vh=n.x*z,n.y*z,n.w*z,n.h*z; self.editor.create_rectangle(vx-3*z,vy-3*z,vx+vw+3*z,vy+vh+3*z,outline=col,width=3*z,tags="hl")
    
    def select_node_safe(self, nid):
        self.after(0, self.editor.select_node, nid)

    def save(self):
        if (fpath:=filedialog.asksaveasfilename(defaultextension=".qflow",filetypes=[("QPass Flow Project","*.qflow")])):
            try:
                with open(fpath,'w',encoding='utf-8') as f: json.dump(self.editor.get_data(),f,indent=2,ensure_ascii=False)
                self.log(f"💾 保存成功: {os.path.basename(fpath)}", "success")
            except Exception as e: messagebox.showerror("保存失败",str(e))
    def load(self):
        if (fpath:=filedialog.askopenfilename(filetypes=[("QPass Flow Project","*.qflow"),("All files","*.*")])):
            try:
                with open(fpath,'r',encoding='utf-8') as f: self.editor.load_data(json.load(f))
                self.log(f"📂 加载成功: {os.path.basename(fpath)}", "success")
            except Exception as e: messagebox.showerror("加载失败",f"无法解析文件: {e}")
    def clear(self): (messagebox.askyesno("确认","确定要清空整个画布吗？") and (self.editor.load_data({'nodes':{},'links':[]}),self.editor.add_node('start',100,100),self.log("🗑️ 画布已清空", "warning")))
    def _setup_hotkeys(self): threading.Thread(target=lambda:keyboard.GlobalHotKeys({'<alt>+1':lambda:self.after(0,self.toggle_run, None)}).start(),daemon=True).start()
    def _on_closing(self): (not self.core.running or messagebox.askyesno("退出确认","流程正在运行中，确定要强制退出吗？")) and (self.core.stop(),self.destroy())

if __name__ == "__main__":
    app = App()
    app.mainloop()