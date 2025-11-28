import threading
import time
import sys
from typing import Callable, Dict, Set
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
    """
    全局快捷键管理器 (状态差分版)
    核心逻辑：维护一个 active_combos 集合。
    - 每次按键变化时，检查注册的组合键状态。
    - 从 False -> True : 触发 PRESS / 启动长按定时器
    - 从 True -> False : 触发 RELEASE / 取消长按定时器
    
    解决了“同时松开两个键导致触发两次 Release”的问题，因为状态移除是原子的。
    """

    def __init__(self):
        self.logger = get_logger("HotkeyManager")
        
        # 快捷键注册表: {'ctrl+f2': {Type.PRESS: func, ...}}
        self.hotkey_callbacks: Dict[str, Dict[HotkeyType, Callable]] = {}
        
        # 当前激活的组合键集合 (防止重复触发)
        self.active_combos: Set[str] = set()
        
        # 长按定时器
        self.long_press_timers: Dict[str, threading.Timer] = {}
        
        self.is_listening = False
        self._lock = threading.Lock()

    def add_hotkey(self, hotkey: str, callback: Callable, hotkey_type: HotkeyType):
        if not KEYBOARD_AVAILABLE: return
        with self._lock:
            # 规范化键名 (如 'Ctrl + F2' -> 'ctrl+f2')
            try:
                # keyboard.parse_hotkey 返回的是 tuple，我们需要转回标准字符串
                # 或者直接用 keyboard.normalize_name (注意：normalize_name处理单个键较好，组合键最好自己标准化)
                # 这里为了简单，手动处理一下空格和大小写
                norm_key = hotkey.lower().replace(' ', '')
                # 再次利用 keyboard 自身的逻辑确保顺序一致 (ctrl+alt vs alt+ctrl)
                # 但 keyboard 没有直接暴露 normalize_hotkey_string，只要我们自己保持 consistent 即可
                if '+' in norm_key:
                    parts = sorted(norm_key.split('+'))
                    norm_key = '+'.join(parts)
            except:
                norm_key = hotkey.lower()

            if norm_key not in self.hotkey_callbacks:
                self.hotkey_callbacks[norm_key] = {}
            self.hotkey_callbacks[norm_key][hotkey_type] = callback
            self.logger.info(f"注册: {norm_key} -> {hotkey_type.value}")

    def start(self):
        """兼容旧接口"""
        self.start_listening()

    def start_listening(self):
        if not KEYBOARD_AVAILABLE or self.is_listening: return
        self.is_listening = True
        self.active_combos.clear()
        # 监听所有键盘事件
        keyboard.hook(self._on_event)
        self.logger.info("🎹 键盘监听已启动")

    def stop_listening(self):
        if not self.is_listening: return
        self.is_listening = False
        keyboard.unhook_all()
        with self._lock:
            for t in self.long_press_timers.values():
                t.cancel()
            self.long_press_timers.clear()
            self.active_combos.clear()

    def _on_event(self, event):
        """
        核心事件循环：不依赖 event.name 判断，而是直接轮询 active_combos 状态变化
        这样可以避免因 event 顺序导致的逻辑错误。
        """
        if not self.is_listening: return
        
        # 为了不阻塞钩子，快速处理。
        # 这里虽然有循环，但注册的快捷键数量通常很少（1-5个），所以开销极小。
        
        # 1. 检查是否有新激活的组合键
        for combo in self.hotkey_callbacks:
            # 使用 keyboard.is_pressed 精准判断组合键物理状态
            if keyboard.is_pressed(combo):
                if combo not in self.active_combos:
                    self._on_combo_down(combo)
        
        # 2. 检查已激活的组合键是否释放
        # 使用 list() 拷贝一份，因为循环中可能会 remove
        for combo in list(self.active_combos):
            if not keyboard.is_pressed(combo):
                self._on_combo_up(combo)

    def _on_combo_down(self, combo):
        with self._lock:
            if combo in self.active_combos: return
            self.active_combos.add(combo)
            
            callbacks = self.hotkey_callbacks.get(combo, {})
            
            # 触发 Press
            if HotkeyType.PRESS in callbacks:
                self._async_run(callbacks[HotkeyType.PRESS], f"Press-{combo}")

            # 启动长按计时
            if HotkeyType.LONG_PRESS in callbacks:
                timer = threading.Timer(
                    0.5, # 长按判定时间 0.5s
                    self._trigger_long_press,
                    args=(combo, callbacks[HotkeyType.LONG_PRESS])
                )
                timer.start()
                self.long_press_timers[combo] = timer

    def _on_combo_up(self, combo):
        with self._lock:
            if combo not in self.active_combos: return
            self.active_combos.remove(combo) # 立即移除，防止双重触发
            
            callbacks = self.hotkey_callbacks.get(combo, {})

            # 停止长按计时（如果还在跑）
            if combo in self.long_press_timers:
                self.long_press_timers[combo].cancel()
                del self.long_press_timers[combo]

            # 触发 Release
            # 注意：如果刚才触发了长按，这里依然会触发 Release (Stop)
            # 这是符合逻辑的：长按开始录音 -> 松开停止录音
            if HotkeyType.RELEASE in callbacks:
                self._async_run(callbacks[HotkeyType.RELEASE], f"Release-{combo}")

    def _trigger_long_press(self, combo, callback):
        """长按定时器触发"""
        with self._lock:
            # 双重检查：防止定时器刚好到期时，用户松手了
            if combo not in self.active_combos:
                return
            
            # 移除 timer 引用
            if combo in self.long_press_timers:
                del self.long_press_timers[combo]
            
            self._async_run(callback, f"LongPress-{combo}")

    def _async_run(self, func, name):
        """独立线程运行回调"""
        def wrapper():
            try:
                func()
            except Exception as e:
                self.logger.error(f"回调错误 [{name}]: {e}")
        threading.Thread(target=wrapper, daemon=True).start()

    # 兼容代码
    def is_supported(self): return KEYBOARD_AVAILABLE