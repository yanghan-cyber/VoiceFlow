"""
全局快捷键管理器 - 防抖增强版
修复了长按 F2 并在回调中模拟打字时，导致按键状态重置的问题。
"""

import threading
import time
import platform
import os
from typing import Callable, Dict, Optional, Set
from enum import Enum

from utils import get_logger

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    logger = get_logger("HotkeyManager")
    logger.error("keyboard库未安装，请运行: pip install keyboard")


class HotkeyType(Enum):
    PRESS = "press"
    RELEASE = "release"
    LONG_PRESS = "long_press"


class HotkeyManager:
    """全局快捷键管理器 (带松开防抖)"""

    def __init__(self):
        self.logger = get_logger("HotkeyManager")
        
        # 回调存储
        self.hotkey_callbacks: Dict[str, Dict[HotkeyType, Callable]] = {}

        self.pressed_keys: Set[str] = set()
        self.long_press_timers: Dict[str, threading.Timer] = {}
        self.long_press_triggered: Set[str] = set()
        
        # 【新增】松开防抖定时器
        # key: 物理按键名, value: Timer
        self.release_debounce_timers: Dict[str, threading.Timer] = {}
        
        self.is_listening = False
        self.listener_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def is_supported(self) -> bool:
        if not KEYBOARD_AVAILABLE: return False
        if platform.system() == 'Linux':
            try:
                if os.geteuid() != 0: return False
            except AttributeError: pass
        return True

    def add_hotkey(self, hotkey: str, callback: Callable,
                   hotkey_type: HotkeyType = HotkeyType.PRESS) -> bool:
        if not KEYBOARD_AVAILABLE: return False
        with self._lock:
            norm_key = self._normalize_hotkey(hotkey)
            if norm_key not in self.hotkey_callbacks:
                self.hotkey_callbacks[norm_key] = {}
            self.hotkey_callbacks[norm_key][hotkey_type] = callback
            self.logger.info(f"注册快捷键: {norm_key} [{hotkey_type.value}]")
            return True

    def start_listening(self) -> bool:
        if not self.is_supported() or self.is_listening: return False
        try:
            self.pressed_keys.clear()
            self.long_press_timers.clear()
            self.long_press_triggered.clear()
            self.release_debounce_timers.clear()

            keyboard.hook(self._on_keyboard_event)
            
            self.is_listening = True
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listener_thread.start()
            self.logger.info("开始监听 (防抖模式)...")
            return True
        except Exception as e:
            self.logger.error(f"启动监听失败: {e}")
            return False

    def stop_listening(self):
        if not self.is_listening: return
        self.is_listening = False
        keyboard.unhook_all()
        with self._lock:
            for t in self.long_press_timers.values(): t.cancel()
            for t in self.release_debounce_timers.values(): t.cancel()
            self.long_press_timers.clear()
            self.release_debounce_timers.clear()

    def _listen_loop(self):
        while self.is_listening: time.sleep(0.5)

    def _normalize_hotkey(self, hotkey: str) -> str:
        replacements = {
            'lctrl': 'ctrl', 'rctrl': 'ctrl', 'left ctrl': 'ctrl', 'right ctrl': 'ctrl', 'control': 'ctrl',
            'lalt': 'alt', 'ralt': 'alt', 'left alt': 'alt', 'right alt': 'alt', 'option': 'alt',
            'lshift': 'shift', 'rshift': 'shift', 'left shift': 'shift', 'right shift': 'shift',
            'lwin': 'windows', 'rwin': 'windows', 'left windows': 'windows', 'right windows': 'windows', 'win': 'windows',
            'esc': 'esc', 'escape': 'esc'
        }
        parts = hotkey.lower().replace(' ', '').split('+')
        normalized_parts = [replacements.get(p, p) for p in parts]
        return '+'.join(sorted(normalized_parts))

    def _get_current_combo(self) -> str:
        return '+'.join(sorted(self.pressed_keys))

    def _on_keyboard_event(self, event):
        try:
            raw_name = event.name.lower()
            remap = {
                'left ctrl': 'ctrl', 'right ctrl': 'ctrl', 'control': 'ctrl',
                'left shift': 'shift', 'right shift': 'shift',
                'left alt': 'alt', 'right alt': 'alt',
                'left windows': 'windows', 'right windows': 'windows', 'win': 'windows'
            }
            key_name = remap.get(raw_name, raw_name)
            
            with self._lock:
                if event.event_type == 'down':
                    self._handle_press(key_name)
                elif event.event_type == 'up':
                    self._handle_release(key_name)
        except Exception as e:
            self.logger.error(f"Event Error: {e}")

    def _handle_press(self, key_name: str):
        # 【防抖逻辑 1】如果这个键正在等待“确认松开”，说明它是抖动或重复，取消松开逻辑
        if key_name in self.release_debounce_timers:
            self.release_debounce_timers[key_name].cancel()
            del self.release_debounce_timers[key_name]
            # 既然取消了松开，说明键一直按着，不需要重新触发 Press 逻辑，直接返回
            return

        # 常规重复检测
        if key_name in self.pressed_keys:
            return
        
        self.pressed_keys.add(key_name)
        current_combo = self._get_current_combo()
        
        if current_combo in self.hotkey_callbacks:
            callbacks = self.hotkey_callbacks[current_combo]
            
            if HotkeyType.PRESS in callbacks:
                try: callbacks[HotkeyType.PRESS]()
                except Exception as e: self.logger.error(f"Press Err: {e}")

            if HotkeyType.LONG_PRESS in callbacks:
                if current_combo in self.long_press_timers:
                    self.long_press_timers[current_combo].cancel()
                
                timer = threading.Timer(
                    0.8, 
                    self._trigger_long_press, 
                    args=[current_combo, callbacks[HotkeyType.LONG_PRESS]]
                )
                timer.start()
                self.long_press_timers[current_combo] = timer

    def _handle_release(self, key_name: str):
        if key_name not in self.pressed_keys:
            return

        # 获取当前的 combo，用于稍后回调
        prev_combo = self._get_current_combo()

        # 【防抖逻辑 2】不要立即处理，而是启动一个短定时器 (50ms)
        # 如果 50ms 内 key 又被 press 了，这个 timer 会被 cancel
        if key_name in self.release_debounce_timers:
            self.release_debounce_timers[key_name].cancel()
        
        timer = threading.Timer(
            0.2, # 50ms 防抖延迟
            self._finalize_release,
            args=[key_name, prev_combo]
        )
        timer.start()
        self.release_debounce_timers[key_name] = timer

    def _finalize_release(self, key_name: str, prev_combo: str):
        """真正的松开逻辑，由定时器触发"""
        with self._lock:
            # 清理 timer 引用
            if key_name in self.release_debounce_timers:
                del self.release_debounce_timers[key_name]
            
            # 真正移除按键
            if key_name in self.pressed_keys:
                self.pressed_keys.remove(key_name)
            else:
                return # 已经被处理过了

            # 触发回调
            if prev_combo in self.hotkey_callbacks:
                callbacks = self.hotkey_callbacks[prev_combo]
                
                # 清理长按定时器
                if prev_combo in self.long_press_timers:
                    self.long_press_timers[prev_combo].cancel()
                    del self.long_press_timers[prev_combo]

                if HotkeyType.RELEASE in callbacks:
                    try: callbacks[HotkeyType.RELEASE]()
                    except Exception as e: self.logger.error(f"Release Err: {e}")

            if prev_combo in self.long_press_triggered:
                self.long_press_triggered.remove(prev_combo)

    def _trigger_long_press(self, combo: str, callback: Callable):
        with self._lock:
            current = self._get_current_combo()
            if current == combo:
                try:
                    self.logger.info(f"触发长按: {combo}")
                    callback()
                    self.long_press_triggered.add(combo)
                except Exception as e:
                    self.logger.error(f"LongPress Err: {e}")
            
            if combo in self.long_press_timers:
                del self.long_press_timers[combo]

    def get_registered_hotkeys(self) -> Dict[str, list]:
        return {k: list(v.keys()) for k, v in self.hotkey_callbacks.items()}