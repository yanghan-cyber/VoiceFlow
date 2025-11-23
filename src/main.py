import time
import threading
import yaml
import numpy as np

from hotkeys.hotkey_manager import HotkeyManager, HotkeyType
from asr.core import ASRFactory
from audio.recorder import AudioRecorder
from utils import TextTyper, get_logger

# 获取主模块日志器
logger = get_logger("MainApp")

class VoiceInputMethod:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        
        # 1. 加载配置
        self.config = self._load_config(config_path)
        self.mode = self.config.get('app', {}).get('mode', 'stream')
        logger.info(f"当前运行模式: {self.mode.upper()}")

        # 2. 初始化 ASR
        self.asr = ASRFactory.get_asr_engine(config_path)
        
        # 3. 初始化录音机
        self.recorder = AudioRecorder(sample_rate=16000)

        # 4. 初始化打字机 【新增】
        self.typer = TextTyper()

        # 5. 绑定回调 (仅流式模式会频繁触发)
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
    # ASR 回调 (核心输入逻辑)
    # ==========================
    
    def on_partial_text(self, text: str):
        """流式中间结果: 类似 '今天天' -> '今天天气'"""
        if self.mode == 'stream':
            # 调用打字机：删除旧的'今天天'，输入'今天天气'
            self.typer.update_stream(text)

    def on_final_text(self, text: str):
        """流式最终结果: VAD 检测到停顿，输出了带标点的整句"""
        if self.mode == 'stream':
            # 调用打字机：删除旧的临时buffer，输入最终带标点的文字
            # 此时这部分文字就“落地”了
            self.typer.commit_text(text)

    # ==========================
    # 任务控制 (按键触发)
    # ==========================

    def start_recording_task(self):
        """【按下F2】开始"""
        # 1. 视觉反馈：在光标处输入提示
        if self.mode == 'stream':
            self.typer.show_status("(( 🎤 ))") 
        else:
            self.typer.show_status("(( 🎤 正在录音... ))")

        # 2. 重置状态
        self.audio_buffer = [] 
        self.is_running = True
        
        # 3. 启动硬件
        self.recorder.start()
        if self.mode == 'stream':
            self.asr.start_stream()

        # 4. 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()

    def stop_recording_task(self):
        """【松开F2】结束"""
        
        # 1. 停止硬件录音
        self.is_running = False 
        self.recorder.stop()
        
        if self.processing_thread:
            self.processing_thread.join()
        
        # 2. 根据模式处理结果
        if self.mode == 'stream':
            # Stream 模式：收尾
            # 停止流，获取最后可能残留的文字
            final_text = self.asr.stop_stream()
            if final_text:
                self.typer.commit_text(final_text)
            else:
                # 如果没有最后结果，仅仅清除屏幕上的 "(( 🎤 ))" 或者残留的 partial
                self.typer.clear_temp()
                
        else:
            # Offline 模式：开始转录
            if not self.audio_buffer:
                self.typer.show_status("(( ⚠️ 时间太短 ))")
                time.sleep(1)
                self.typer.clear_temp()
                return

            # 更新状态提示
            self.typer.show_status("(( ⏳ 转录中... ))")

            # 合并音频并识别
            try:
                full_audio = np.concatenate(self.audio_buffer)
                text = self.asr.transcribe_offline(full_audio, sample_rate=16000)
                
                # 最终上屏
                if text:
                    self.typer.commit_text(text)
                else:
                    self.typer.show_status("(( ❌ 未识别到内容 ))")
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
        app = VoiceInputMethod() # 使用新的类名
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    hm = HotkeyManager()

    if not hm.is_supported():
        logger.error("环境不支持全局快捷键")
        return

    # 注册 F2
    hm.add_hotkey("f2", app.start_recording_task, HotkeyType.LONG_PRESS)
    hm.add_hotkey("f2", app.stop_recording_task, HotkeyType.RELEASE)
    
    logger.info(f"语音输入法已启动 (模式: {app.mode})")
    logger.info("请把光标放在任意输入框中，长按 F2 说话...")
    
    if hm.start_listening():
        import keyboard
        try:
            keyboard.wait('esc')
        except KeyboardInterrupt:
            pass
        finally:
            hm.stop_listening()

if __name__ == "__main__":
    main()