import os
import sys
import time
import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sys.stderr.write("错误: 请运行 'pip install sherpa-onnx'\n")
    sys.exit(1)

from .core import ASRBase
from utils import get_logger

# 获取SenseVoice ASR模块日志器
logger = get_logger("SherpaSenseVoiceASR")


class SherpaSenseVoiceASR(ASRBase):
    def __init__(self, config: dict):
        """
        初始化 SenseVoice ASR (非流式模型 + VAD 实现伪流式)
        :param config: 包含模型路径、VAD配置、热词配置的字典
        """
        super().__init__()

        # ==================================================
        # 1. 初始化 SenseVoice 识别器 (自带标点与ITN)
        # ==================================================
        model_path = config.get("model")
        tokens_path = config.get("tokens")

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SenseVoice模型文件缺失: {model_path}")
        if not tokens_path or not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Tokens文件缺失: {tokens_path}")

        logger.info("正在加载 SenseVoice 模型 (集成标点/ITN)...")

        # 提取热词参数
        hr_dict_dir = config.get("hr_dict_dir", "")
        hr_rule_fsts = config.get("hr_rule_fsts", "")
        hr_lexicon = config.get("hr_lexicon", "")

        if hr_dict_dir and not os.path.exists(hr_dict_dir):
            logger.warning(f"⚠️ 警告: 热词字典目录不存在: {hr_dict_dir}")

        try:
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_path,
                tokens=tokens_path,
                num_threads=config.get("num_threads", 1),
                use_itn=config.get("use_itn", True),
                debug=config.get("debug", False),
                language=config.get("language", "auto"),
                provider=config.get("provider", "cpu"),
                hr_dict_dir=hr_dict_dir,
                hr_rule_fsts=hr_rule_fsts,
                hr_lexicon=hr_lexicon,
            )
        except Exception as e:
            logger.error(f"❌ SenseVoice 初始化失败: {e}")
            raise e

        # ==================================================
        # 2. 初始化 VAD (语音活动检测)
        # ==================================================
        vad_params = config.get("vad", {})
        vad_model_path = vad_params.get("model")

        self.vad = None
        self.vad_window_size = 512

        # 【修改点 1】: VAD 初始化逻辑优化，支持降级
        if not vad_model_path or not os.path.exists(vad_model_path):
            logger.warning("⚠️ 警告: 未找到 VAD 模型，系统将自动降级为 [Offline模式]")
            logger.info("   (在录音过程中不会有实时文字，只有停止录音后才会输出结果)")
        else:
            vad_type = vad_params.get("type", "ten_vad")
            logger.info(f"正在加载 VAD 模型 ({vad_type})...")

            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.sample_rate = 16000

            threshold = vad_params.get("threshold", 0.5)
            min_silence = vad_params.get("min_silence_duration", 0.1)
            min_speech = vad_params.get("min_speech_duration", 0.25)
            max_speech = vad_params.get("max_speech_duration", 8.0)
            window_size = vad_params.get("window_size", 512)

            # 配置参数填充
            if vad_type == "silero_vad":
                vad_config.silero_vad.model = vad_model_path
                vad_config.silero_vad.threshold = threshold
                vad_config.silero_vad.min_silence_duration = min_silence
                vad_config.silero_vad.min_speech_duration = min_speech
                vad_config.silero_vad.max_speech_duration = max_speech
                vad_config.silero_vad.window_size = window_size
            elif vad_type == "ten_vad":
                vad_config.ten_vad.model = vad_model_path
                vad_config.ten_vad.threshold = threshold
                vad_config.ten_vad.min_silence_duration = min_silence
                vad_config.ten_vad.min_speech_duration = min_speech
                vad_config.ten_vad.max_speech_duration = max_speech
                vad_config.ten_vad.window_size = window_size
            else:
                logger.warning(f"⚠️ 未知 VAD 类型: {vad_type}，尝试使用 Silero")
                vad_config.silero_vad.model = vad_model_path

            try:
                self.vad = sherpa_onnx.VoiceActivityDetector(
                    vad_config, buffer_size_in_seconds=100
                )
                if vad_type == "silero_vad":
                    self.vad_window_size = vad_config.silero_vad.window_size
                elif vad_type == "ten_vad":
                    self.vad_window_size = vad_config.ten_vad.window_size
                else:
                    self.vad_window_size = window_size
            except Exception as e:
                logger.error(f"❌ VAD 初始化失败: {e}")
                self.vad = None

        # 运行时状态变量
        self.buffer = np.array([], dtype=np.float32)
        self.started = False
        self.started_time = 0

    def start_stream(self):
        """重置流状态"""
        if self.vad:
            self.vad.reset()
        self.buffer = np.array([], dtype=np.float32)
        self.started = False
        self.started_time = 0

    def feed_audio(self, samples: np.ndarray, sample_rate: int):
        """
        处理音频流
        情况A (有VAD): 伪流式逻辑，VAD切分 -> SenseVoice识别 -> 实时回调
        情况B (无VAD): 纯缓冲逻辑，只存不识 -> 等待 stop_stream
        """
        
        # 【修改点 2】: 降级处理逻辑
        if not self.vad:
            # 没有 VAD，我们只能把数据存起来，不做任何切分
            self.buffer = np.concatenate([self.buffer, samples])
            return

        # --- 以下是有 VAD 时的正常逻辑 ---

        self.vad.accept_waveform(samples)

        # 维护 buffer 用于 partial decode
        self.buffer = np.concatenate([self.buffer, samples])
        max_len = 16000 * 2
        if not self.started and len(self.buffer) > max_len:
            self.buffer = self.buffer[-max_len:]

        # 检测开始
        if self.vad.is_speech_detected() and not self.started:
            self.started = True
            self.started_time = time.time()

        # 实时回显 (Partial)
        if self.started:
            if len(self.buffer) > 0 and (time.time() - self.started_time > 0.25):
                self.started_time = time.time()

                s = self.recognizer.create_stream()
                s.accept_waveform(sample_rate, self.buffer)
                self.recognizer.decode_stream(s)

                text = s.result.text.strip()
                if text and self.on_partial_result:
                    self.on_partial_result(text)

        # 处理 VAD 切分出的完整句子 (Final)
        while not self.vad.empty():
            segment = self.vad.front.samples
            self.vad.pop()

            s = self.recognizer.create_stream()
            s.accept_waveform(sample_rate, segment)
            self.recognizer.decode_stream(s)

            raw_text = s.result.text.strip()

            if raw_text:
                if self.on_final_result:
                    self.on_final_result(raw_text)

            self.buffer = np.array([], dtype=np.float32)
            self.started = False

    def stop_stream(self) -> str:
        """
        强制停止
        情况A (有VAD): 识别 VAD 缓存中剩余的尾音
        情况B (无VAD): 识别整个 buffer (即整个录音段)
        """
        result = ""

        # 【修改点 3】: 降级模式的结束处理
        if not self.vad:
            # 离线模式：一次性识别所有累积的音频
            if len(self.buffer) > 0:
                logger.info("⏳ 正在进行离线识别...")
                s = self.recognizer.create_stream()
                s.accept_waveform(16000, self.buffer)
                self.recognizer.decode_stream(s)
                result = s.result.text.strip()
            
            # 清理状态
            self.start_stream()
            return result

        # --- 以下是有 VAD 时的结束逻辑 ---
        if self.started and len(self.buffer) > 0:
            s = self.recognizer.create_stream()
            s.accept_waveform(16000, self.buffer)
            self.recognizer.decode_stream(s)

            raw_text = s.result.text.strip()
            if raw_text:
                result = raw_text

        self.start_stream()
        return result

    def transcribe_offline(self, samples: np.ndarray, sample_rate: int) -> str:
        """非流式识别"""
        s = self.recognizer.create_stream()
        s.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(s)

        return s.result.text.strip()