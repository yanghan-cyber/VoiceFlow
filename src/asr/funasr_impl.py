import numpy as np
import os
import time
from typing import Optional
from funasr import AutoModel

try:
    import torch
except ImportError:
    import sys
    sys.stderr.write("错误: 请运行 'pip install torch'\n")
    sys.exit(1)

try:
    import sherpa_onnx
except ImportError:
    import sys
    sys.stderr.write("错误: 请运行 'pip install sherpa-onnx'\n")
    sys.exit(1)

from .core import ASRBase
from utils import get_logger

logger = get_logger("FunASRImpl")


class FunASRASR(ASRBase):
    """
    FunASR ASR 实现类
    采用与 SherpaSenseVoiceASR 类似的架构：
    - 外部VAD用于流式切分
    - FunASR模型用于识别
    - 支持降级到离线模式
    """

    def __init__(self, config: dict):
        """
        :param config: FunASR 配置参数
        {
            "model_dir": "FunAudioLLM/Fun-ASR-Nano-2512",  # 模型目录
            "device": "cuda:0",         # 设备
            "sample_rate": 16000,     # 采样率
            "enable_vad": True,       # 是否启用VAD（用于流式）
            "vad_model": "fsmn-vad",  # VAD模型
            "vad_kwargs": {}          # VAD参数
        }
        """
        super().__init__()

        # 基础配置
        self.sample_rate = config.get("sample_rate", 16000)
        model_dir = config.get("model_dir", "FunAudioLLM/Fun-ASR-Nano-2512")
        device = config.get("device", "cuda:0")
        enable_vad = config.get("enable_vad", True)

        logger.info(f"正在初始化 FunASR 模型: {model_dir}")
        logger.info(f"设备: {device}")

        # 初始化 FunASR 模型（固定参数）
        try:
            model_config = {
                "model": model_dir,
                "trust_remote_code": True,
                "remote_code": "./src/asr/utils/model.py",  # 固定路径
                "vad_model": "fsmn-vad",
                "vad-kwargs": {"max_single_segment_time": 30000},  # VAD参数"
                "device": device,
            }

            self.model = AutoModel(**model_config)
            logger.info("✅ FunASR 模型加载成功")
        except Exception as e:
            logger.error(f"❌ FunASR 模型加载失败: {e}")
            raise e

        # 初始化 VAD（用于流式识别）
        self.vad = None
        self.vad_window_size = 512
        self._init_vad(config, enable_vad)

        # 运行时状态变量
        self.buffer = np.array([], dtype=np.float32)
        self.started = False
        self.started_time = 0
        self.is_streaming = False

    def _init_vad(self, config: dict, enable_vad: bool):
        """初始化 VAD"""
        if not enable_vad:
            logger.warning("⚠️ 警告: VAD 被禁用，系统将自动降级为 [Offline模式]")
            logger.info("   (在录音过程中不会有实时文字，只有停止录音后才会输出结果)")
            return

        vad_params = config.get("vad_kwargs", {})
        vad_model = config.get("vad_model", "ten-vad")

        # VAD 模型路径 - 只支持实际存在的模型
        if vad_model == "silero-vad":
            vad_model_path = "./ckpts/vad/silero_vad.onnx"
        elif vad_model == "ten-vad":
            vad_model_path = "./ckpts/vad/ten-vad.onnx"
        else:
            # 默认使用 ten-vad
            vad_model_path = "./ckpts/vad/ten-vad.onnx"
            vad_model = "ten-vad"

        if not os.path.exists(vad_model_path):
            logger.warning(f"⚠️ 警告: 未找到 VAD 模型 {vad_model_path}，系统将降级为 [Offline模式]")
            return

        logger.info(f"正在加载 VAD 模型 ({vad_model})...")

        try:
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.sample_rate = self.sample_rate

            # VAD 参数配置
            threshold = vad_params.get("threshold", 0.5)
            min_silence = vad_params.get("min_silence_duration", 0.1)
            min_speech = vad_params.get("min_speech_duration", 0.25)
            max_speech = vad_params.get("max_speech_duration", 8.0)
            window_size = vad_params.get("window_size", 512)

            if vad_model == "silero-vad":
                # Silero VAD 配置
                vad_config.silero_vad.model = vad_model_path
                vad_config.silero_vad.threshold = threshold
                vad_config.silero_vad.min_silence_duration = min_silence
                vad_config.silero_vad.min_speech_duration = min_speech
                vad_config.silero_vad.max_speech_duration = max_speech
                vad_config.silero_vad.window_size = window_size
            else:
                # ten-vad 配置
                vad_config.ten_vad.model = vad_model_path
                vad_config.ten_vad.threshold = threshold
                vad_config.ten_vad.min_silence_duration = min_silence
                vad_config.ten_vad.min_speech_duration = min_speech
                vad_config.ten_vad.max_speech_duration = max_speech
                vad_config.ten_vad.window_size = window_size

            self.vad = sherpa_onnx.VoiceActivityDetector(
                vad_config, buffer_size_in_seconds=100
            )
            self.vad_window_size = window_size
            logger.info("✅ VAD 模型加载成功")

        except Exception as e:
            logger.error(f"❌ VAD 初始化失败: {e}")
            self.vad = None

    def start_stream(self):
        """重置流状态"""
        if self.vad:
            self.vad.reset()
        self.buffer = np.array([], dtype=np.float32)
        self.started = False
        self.started_time = 0
        self.is_streaming = True

    def feed_audio(self, samples: np.ndarray, sample_rate: int):
        """
        处理音频流
        情况A (有VAD): 伪流式逻辑，VAD切分 -> FunASR识别 -> 实时回调
        情况B (无VAD): 纯缓冲逻辑，只存不识 -> 等待 stop_stream
        """
        # 重采样到目标采样率
        if sample_rate != self.sample_rate:
            samples = self._resample(samples, sample_rate, self.sample_rate)

        # 【降级处理逻辑】
        if not self.vad:
            # 没有 VAD，我们只能把数据存起来，不做任何切分
            self.buffer = np.concatenate([self.buffer, samples])
            return

        # --- 以下是有 VAD 时的正常逻辑 ---

        self.vad.accept_waveform(samples)

        # 维护 buffer 用于 partial decode
        self.buffer = np.concatenate([self.buffer, samples])
        max_len = 16000 * 2  # 最多保留2秒音频用于部分识别
        if not self.started and len(self.buffer) > max_len:
            self.buffer = self.buffer[-max_len:]

        # 检测开始
        if self.vad.is_speech_detected() and not self.started:
            self.started = True
            self.started_time = time.time()

        # 实时回显 (Partial) - 每0.25秒识别一次当前buffer
        if self.started:
            if len(self.buffer) > 0 and (time.time() - self.started_time > 0.25):
                self.started_time = time.time()

                # 使用 FunASR 进行部分识别
                partial_text = self._process_audio_chunk(self.buffer, is_partial=True)
                if partial_text and self.on_partial_result:
                    self.on_partial_result(partial_text)

        # 处理 VAD 切分出的完整句子 (Final)
        while not self.vad.empty():
            segment = self.vad.front.samples
            self.vad.pop()

            # 确保segment是numpy数组而不是list
            if isinstance(segment, list):
                segment = np.array(segment, dtype=np.float32)

            # 识别这个语音段
            text = self._process_audio_chunk(segment, is_partial=False)
            if text and self.on_final_result:
                self.on_final_result(text)

            # 重置状态
            self.buffer = np.array([], dtype=np.float32)
            self.started = False

    def stop_stream(self) -> str:
        """
        强制停止
        情况A (有VAD): 识别 VAD 缓存中剩余的尾音
        情况B (无VAD): 识别整个 buffer (即整个录音段)
        """
        result = ""

        # 【降级模式的结束处理】
        if not self.vad:
            # 离线模式：一次性识别所有累积的音频
            if len(self.buffer) > 0:
                logger.info("⏳ 正在进行离线识别...")
                result = self._process_audio_chunk(self.buffer, is_partial=False)

            # 清理状态
            self.start_stream()
            return result

        # --- 以下是有 VAD 时的结束逻辑 ---
        if self.started and len(self.buffer) > 0:
            # 识别剩余的音频
            result = self._process_audio_chunk(self.buffer, is_partial=False)

        self.start_stream()
        return result

    def transcribe_offline(self, samples: np.ndarray, sample_rate: int) -> str:
        """离线识别完整音频"""
        logger.debug("FunASR 离线识别")

        # 重采样到目标采样率
        if sample_rate != self.sample_rate:
            samples = self._resample(samples, sample_rate, self.sample_rate)

        # 确保数据类型正确
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # 将 numpy 数组转换为 PyTorch Tensor
        audio_tensor = torch.from_numpy(samples).float()

        # 使用 FunASR 进行离线识别
        result = self.model.generate(input=audio_tensor, batch_size=1)

        if result and len(result) > 0:
            if isinstance(result[0], dict):
                return result[0].get("text", "")
            elif isinstance(result[0], str):
                return result[0]

        return ""

    def _resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """简单的重采样实现"""
        if orig_sr == target_sr:
            return samples

        # 使用 scipy 进行重采样
        try:
            import scipy.signal
            return scipy.signal.resample(samples, int(len(samples) * target_sr / orig_sr))
        except ImportError:
            # 简单的线性插值作为备选
            import numpy as np
            x_old = np.arange(len(samples))
            x_new = np.arange(0, len(samples), orig_sr / target_sr)
            return np.interp(x_new, x_old, samples)

    def _process_audio_chunk(self, audio_chunk: np.ndarray, is_partial: bool = False) -> str:
        """处理音频块"""
        try:
            # 确保输入是numpy数组
            if isinstance(audio_chunk, list):
                audio_chunk = np.array(audio_chunk, dtype=np.float32)
            elif not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk, dtype=np.float32)

            # 使用 FunASR 进行识别
            # 注意：FunASR 的 generate 方法不支持真正的流式缓存
            # 所以我们每次都要重新识别整个音频块

            # 将 numpy 数组转换为 PyTorch Tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()

            result = self.model.generate(
                input=audio_tensor,
                batch_size=1
            )

            if result and len(result) > 0:
                # 提取文本结果
                if isinstance(result[0], dict):
                    return result[0].get("text", "")
                elif isinstance(result[0], str):
                    return result[0]

            return ""

        except Exception as e:
            logger.error(f"音频块处理失败: {e}")
            return ""