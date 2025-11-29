import time
import threading
import yaml
import numpy as np
import sys

# 确保引入 keyboard
try:
    import keyboard
except ImportError:
    sys.exit("请安装 keyboard 库: pip install keyboard")

from hotkeys.hotkey_manager import HotkeyManager, HotkeyType
from asr.core import ASRFactory
from audio.recorder import AudioRecorder
# 根据你的目录结构调整导入
from utils.typer import TextTyper 
from utils import get_logger
# 导入新的优化器
from llm.optimizer import LLMOptimizer

logger = get_logger("MainApp")

class VoiceInputMethod:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        
        # 1. 加载配置
        self.config = self._load_config(config_path)
        self.app_config = self.config.get('app', {})
        
        # 默认模式 (仅影响普通录音)
        self.default_mode = self.app_config.get('mode', 'stream')
        
        # 获取快捷键配置 (默认为 ctrl+f2 和 ctrl+f3)
        self.hotkeys = self.app_config.get('hotkeys', {})
        self.key_std = self.hotkeys.get('std', 'ctrl+f2')
        self.key_llm = self.hotkeys.get('llm', 'ctrl+f3')
        
        logger.info(f"运行配置: 普通模式={self.default_mode.upper()}")

        # 2. 初始化组件
        self.asr = ASRFactory.get_asr_engine(config_path)
        self.recorder = AudioRecorder(sample_rate=16000)
        self.typer = TextTyper()
        self.llm = LLMOptimizer(self.config)

        # 3. 绑定 ASR 回调
        self.asr.on_partial_result = self.on_partial_text
        self.asr.on_final_result = self.on_final_text

        # 4. 运行状态
        self.processing_thread = None
        self.is_running = False
        self.audio_buffer = []
        
        # 标记当前任务类型: 'std' (普通) 或 'llm' (AI)
        self.current_task = None 

    def _load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ==========================
    # ASR 回调 (仅用于 Stream 模式)
    # ==========================
    def on_partial_text(self, text: str):
        # 只有在普通流式模式下才实时上屏
        if self.current_task == 'std' and self.default_mode == 'stream':
            self.typer.update_stream(text)

    def on_final_text(self, text: str):
        if self.current_task == 'std' and self.default_mode == 'stream':
            self.typer.commit_text(text)

    # ==========================
    # 通用控制逻辑
    # ==========================
    def _start_capture(self):
        """启动录音硬件和线程"""
        if self.is_running: return
        
        self.audio_buffer = [] 
        self.is_running = True
        self.recorder.start()
        
        # 启动后台处理线程
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()

    def _stop_capture(self):
        """停止录音硬件"""
        self.is_running = False
        self.recorder.stop()
        if self.processing_thread:
            self.processing_thread.join()

    def _process_loop(self):
        """音频数据处理循环"""
        while self.is_running:
            chunk = self.recorder.get_audio_chunk()
            if chunk is not None:
                # 只有在 [普通任务] 且 [流式模式] 下才推流给 ASR
                # LLM 任务强制离线，所以只存 Buffer
                if self.current_task == 'std' and self.default_mode == 'stream':
                    self.asr.feed_audio(chunk, sample_rate=16000)
                else:
                    self.audio_buffer.append(chunk)
            else:
                time.sleep(0.005)

    # ==========================
    # 任务入口 1: 普通录音 (Ctrl + F2)
    # ==========================
    def start_recording_task(self):
        """开始: 遵循配置的 mode (stream/offline)"""
        if self.is_running: return
        self.current_task = 'std'
        
        status = "(( 🎤 普通录音... ))" if self.default_mode == 'stream' else "(( 🎤 离线录音... ))"
        self.typer.show_status(status)
        
        self._start_capture()
        if self.default_mode == 'stream':
            self.asr.start_stream()

    # ==========================
    # 任务入口 2: AI 润色录音 (Ctrl + F3)
    # ==========================
    def start_llm_recording_task(self):
        """开始: 强制 Offline + LLM 优化"""
        if self.is_running: return
        self.current_task = 'llm'
        
        # 提示用户进入了 AI 模式
        self.typer.show_status("(( ✨ AI 思考录音... ))")
        
        # 启动录音，但不启动 ASR 流
        self._start_capture()

    # ==========================
    # 统一结束入口 (松开按键)
    # ==========================
    def stop_any_task(self):
        """根据当前 current_task 决定如何处理结束"""
        if not self.is_running: return
        
        logger.info(f"停止任务: {self.current_task}")
        self._stop_capture() # 停止录音
        
        # 分流处理
        if self.current_task == 'std':
            self._finish_std_task()
        elif self.current_task == 'llm':
            self._finish_llm_task()
        
        self.current_task = None

    def _finish_std_task(self):
        """处理普通任务结果"""
        if self.default_mode == 'stream':
            final_text = self.asr.stop_stream()
            if final_text:
                self.typer.commit_text(final_text)
            else:
                self.typer.clear_temp()
        else:
            # 普通离线
            self._transcribe_and_paste(use_llm=False)

    def _finish_llm_task(self):
        """处理 AI 任务结果 (强制离线 + LLM)"""
        self._transcribe_and_paste(use_llm=True)

    def _transcribe_and_paste(self, use_llm=False):
        """离线转录公共逻辑"""
        if not self.audio_buffer:
            self.typer.show_status("(( ⚠️ 时间太短 ))")
            time.sleep(1)
            self.typer.clear_temp()
            return

        status_msg = "(( ✨ AI 润色中... ))" if use_llm else "(( ⏳ 转录中... ))"
        self.typer.show_status(status_msg)

        try:
            # 1. ASR 识别
            full_audio = np.concatenate(self.audio_buffer)
            text = self.asr.transcribe_offline(full_audio, sample_rate=16000)
            
            if not text:
                self.typer.show_status("(( ❌ 未识别到内容 ))")
                time.sleep(1)
                self.typer.clear_temp()
                return

            # 2. (可选) LLM 优化
            final_text = text
            if use_llm:
                final_text = self.llm.optimize(text)

            # 3. 上屏
            self.typer.clear_temp()
            self.typer.commit_text(final_text)
            
        except Exception as e:
            logger.error(f"处理出错: {e}")
            self.typer.clear_temp()


def main():
    try:
        app = VoiceInputMethod()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    hm = HotkeyManager()
    if not hm.is_supported():
        return

    # 注册快捷键
    # 任务 1: 普通 (Ctrl+F2)
    hm.add_hotkey(app.key_std, app.start_recording_task, HotkeyType.LONG_PRESS)
    hm.add_hotkey(app.key_std, app.stop_any_task, HotkeyType.RELEASE)
    
    # 任务 2: AI (Ctrl+F3)
    hm.add_hotkey(app.key_llm, app.start_llm_recording_task, HotkeyType.LONG_PRESS)
    hm.add_hotkey(app.key_llm, app.stop_any_task, HotkeyType.RELEASE)
    
    logger.info("============== 系统就绪 ==============")
    logger.info(f"1. 普通模式: 按住 [{app.key_std}] 说话")
    logger.info(f"2. AI 模式 : 按住 [{app.key_llm}] 说话 (自动润色)")
    logger.info("======================================")
    
    hm.start_listening()
    
    try:
        keyboard.wait('esc')
    except KeyboardInterrupt:
        pass
    finally:
        hm.stop_listening()
        logger.info("程序已退出")

if __name__ == "__main__":
    main()