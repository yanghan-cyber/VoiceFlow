import time
import threading
import yaml
import numpy as np
import sys

# 确保引入 keyboard，用于最后的主线程阻塞
try:
    import keyboard
except ImportError:
    sys.exit("请安装 keyboard 库")

from hotkeys.hotkey_manager import HotkeyManager, HotkeyType
from asr.core import ASRFactory
from audio.recorder import AudioRecorder
# 注意：这里假设你的 TextTyper 类在 utils.py 或 typer.py 中，请根据实际情况导入
# 假设你上面的 typer 代码保存在 typer.py 中
from utils.typer import TextTyper 
from utils import get_logger

# 获取主模块日志器
logger = get_logger("MainApp")

class VoiceInputMethod:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path

        # 1. 加载配置
        self.config = self._load_config(config_path)
        self.mode = self.config.get('app', {}).get('mode', 'stream')
        self.hotkey_config = self.config.get('app', {}).get('hotkey', {})
        logger.info(f"当前运行模式: {self.mode.upper()}")

        # 2. 初始化 ASR
        self.asr = ASRFactory.get_asr_engine(config_path)
        
        # 3. 初始化录音机
        self.recorder = AudioRecorder(sample_rate=16000)

        # 4. 初始化打字机
        self.typer = TextTyper()

        # 5. 绑定回调
        self.asr.on_partial_result = self.on_partial_text
        self.asr.on_final_result = self.on_final_text

        # 状态
        self.processing_thread = None
        self.is_running = False
        self.audio_buffer = []

    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ==========================
    # ASR 回调
    # ==========================
    
    def on_partial_text(self, text: str):
        """流式中间结果"""
        if self.mode == 'stream':
            self.typer.update_stream(text)

    def on_final_text(self, text: str):
        """流式最终结果"""
        if self.mode == 'stream':
            self.typer.commit_text(text)

    # ==========================
    # 任务控制
    # ==========================

    def start_recording_task(self):
        """【按下/长按】开始"""
        if self.mode == 'stream':
            self.typer.show_status("🎤...流式录制") 
        else:
            self.typer.show_status("🎤...离线录制")

        self.audio_buffer = [] 
        self.is_running = True
        
        self.recorder.start()
        if self.mode == 'stream':
            self.asr.start_stream()

        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()

    def stop_recording_task(self):
        """【松开】结束"""
        logger.info(f"停止录音 (线程: {threading.current_thread().name})")

        self.is_running = False
        self.recorder.stop()
        
        if self.processing_thread:
            self.processing_thread.join()
        
        if self.mode == 'stream':
            final_text = self.asr.stop_stream()
            if final_text:
                self.typer.commit_text(final_text)
            else:
                self.typer.clear_temp()
        else:
            # Offline 模式逻辑
            if not self.audio_buffer:
                self.typer.show_status("⚠️ 时间太短")
                time.sleep(1)
                self.typer.clear_temp()
                return

            self.typer.show_status("⏳ 转录中...")
            try:
                full_audio = np.concatenate(self.audio_buffer)
                text = self.asr.transcribe_offline(full_audio, sample_rate=16000)
                self.typer.clear_temp()

                if text:
                    self.typer.commit_text(text)
                    logger.info(f"识别结果：{text}")
                else:
                    self.typer.show_status("❌ 未识别到内容")
                    time.sleep(1)
                    self.typer.clear_temp()
            except Exception as e:
                logger.error(f"识别出错: {e}")
                self.typer.clear_temp()

    def _process_loop(self):
        while self.is_running:
            chunk = self.recorder.get_audio_chunk()
            if chunk is not None:
                if self.mode == 'stream':
                    self.asr.feed_audio(chunk, sample_rate=16000)
                else:
                    self.audio_buffer.append(chunk)
            else:
                time.sleep(0.005)

def main():
    try:
        app = VoiceInputMethod()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    hm = HotkeyManager()

    if not hm.is_supported(): # 这个方法是有返回值的，可以保留
        logger.error("环境不支持全局快捷键")
        return

    # 从配置读取快捷键
    key_combination = app.hotkey_config.get('key_combination', 'ctrl+f2')
    exit_key = app.hotkey_config.get('exit_key', 'esc')

    # 注册快捷键
    hm.add_hotkey(key_combination, app.start_recording_task, HotkeyType.LONG_PRESS)
    hm.add_hotkey(key_combination, app.stop_recording_task, HotkeyType.RELEASE)

    logger.info(f"语音输入法已启动 (模式: {app.mode})")
    logger.info(f"录音快捷键: {key_combination}, 退出键: {exit_key}")
    logger.info(f"请把光标放在任意输入框中，按住 {key_combination} 说话...")

    # 1. 启动监听
    hm.start_listening()

    # 2. 只有启动后才进入阻塞
    logger.info(f"按下 {exit_key} 键退出程序")
    try:
        keyboard.wait(exit_key)
    except KeyboardInterrupt:
        pass
    finally:
        hm.stop_listening()
        logger.info("程序已退出")
    
    # --- 修改重点 END ---

if __name__ == "__main__":
    main()