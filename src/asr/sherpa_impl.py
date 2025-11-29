# src/asr/sherpa_impl.py

import os
import sys
import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sys.stderr.write("错误: 请运行 'pip install sherpa-onnx'\n")
    sys.exit(1)

from .core import ASRBase
from utils import get_logger

# 获取ASR模块日志器
logger = get_logger("SherpaOnnxASR")


class SherpaOnnxASR(ASRBase):
    def __init__(self, config: dict, punct_config: dict = None):
        """
        :param config: ASR 模型参数 (Paraformer)
        :param punct_config: 标点模型参数
        """
        super().__init__()

        # 1. 初始化 ASR (保持不变)
        required_files = [
            config.get("tokens"),
            config.get("encoder"),
            config.get("decoder"),
        ]
        for f in required_files:
            if f and not os.path.exists(f):
                raise FileNotFoundError(f"ASR模型文件未找到: {f}")

        logger.info("正在加载 Sherpa-Onnx ASR 模型...")
        try:
            # 为参数添加默认值（这些已从 config.yaml 中移除）
            config.setdefault("num_threads", 1)
            config.setdefault("provider", "cpu")
            config.setdefault("device", 0)
            config.setdefault("sample_rate", 16000)
            config.setdefault("feature_dim", 80)
            config.setdefault("enable_endpoint_detection", True)
            config.setdefault("rule1_min_trailing_silence", 2.4)
            config.setdefault("rule2_min_trailing_silence", 1.2)
            config.setdefault("rule3_min_utterance_length", 30)
            config.setdefault("decoding_method", "greedy_search")
            config.setdefault("debug", False)

            self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(**config)
        except Exception as e:
            logger.error(f"❌ ASR 初始化失败: {e}")
            raise e

        # 2. 初始化标点模型 (新增)
        self.punct = None
        if punct_config and punct_config.get("enabled", False):
            logger.info("正在加载标点恢复模型...")
            punct_model_path = punct_config.get("model")
            if not os.path.exists(punct_model_path):
                logger.warning(f"⚠️ 警告: 标点模型文件不存在: {punct_model_path}，将跳过标点恢复。")
            else:
                try:
                    # 构建标点配置对象
                    punct_cfg = sherpa_onnx.OfflinePunctuationConfig(
                        model=sherpa_onnx.OfflinePunctuationModelConfig(
                            ct_transformer=punct_model_path,
                            num_threads=punct_config.get("num_threads", 1),
                            provider=punct_config.get("provider", "cpu"),
                        ),
                    )
                    self.punct = sherpa_onnx.OfflinePunctuation(punct_cfg)
                    logger.info("✅ 标点模型加载完毕")
                except Exception as e:
                    logger.error(f"❌ 标点模型初始化失败: {e}")

        self.stream = None
        logger.info("✅ ASR 引擎就绪")

    def _add_punctuation(self, text: str) -> str:
        """内部辅助函数：给文本加标点"""
        if self.punct and text and len(text.strip()) > 0:
            return self.punct.add_punctuation(text)
        return text

    def start_stream(self):
        if self.stream is not None:
            self.stream = None
        self.stream = self.recognizer.create_stream()

    def feed_audio(self, samples: np.ndarray, sample_rate: int):
        if self.stream is None:
            return

        self.stream.accept_waveform(sample_rate, samples)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        # 获取当前识别结果 (可能是半句话)
        text = self.recognizer.get_result(self.stream)

        # 流式中间结果：通常不加标点，因为会跳变
        if text:
            if self.on_partial_result:
                self.on_partial_result(text)

        # 【核心修改】端点检测：一句话结束了
        if self.recognizer.is_endpoint(self.stream):
            if text:
                # 1. 加标点
                text_with_punct = self._add_punctuation(text)

                # 2. 回调最终结果
                if self.on_final_result:
                    self.on_final_result(text_with_punct)

            # 3. 重置流，准备下一句
            self.recognizer.reset(self.stream)

    def stop_stream(self) -> str:
        """停止流并处理剩余尾音"""
        result = ""
        if self.stream:
            raw_text = self.recognizer.get_result(self.stream)
            if raw_text:
                # 停止时，肯定也是一句话的结束，需要加标点
                result = self._add_punctuation(raw_text)
            self.stream = None
        return result

    def transcribe_offline(self, samples: np.ndarray, sample_rate: int) -> str:
        """非流式识别"""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

        raw_text = self.recognizer.get_result(stream)
        # 离线识别结果直接加标点
        return self._add_punctuation(raw_text)
