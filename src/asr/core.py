import abc
import yaml
import os
from typing import Callable, Optional
import numpy as np

# ASRBase 保持不变 ...
class ASRBase(abc.ABC):
    def __init__(self):
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[str], None]] = None

    @abc.abstractmethod
    def start_stream(self): pass

    @abc.abstractmethod
    def feed_audio(self, samples: np.ndarray, sample_rate: int): pass

    @abc.abstractmethod
    def stop_stream(self) -> str: pass

    @abc.abstractmethod
    def transcribe_offline(self, samples: np.ndarray, sample_rate: int) -> str: pass

class ASRFactory:
    @staticmethod
    def _load_config(config_path: str) -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_asr_engine(config_path: str = "config.yaml") -> ASRBase:
        full_config = ASRFactory._load_config(config_path)
        asr_config = full_config.get('asr', {})
        hotwords = asr_config.get('hotwords', [])
        engine_type = asr_config.get('active_engine', 'sherpa_onnx')

        if engine_type == "sherpa_onnx":
            sherpa_cfg = asr_config.get('sherpa_onnx', {})
            # 获取模型类型: paraformer 或 sense_voice
            model_type = sherpa_cfg.get('model_type', 'paraformer')

            if model_type == 'paraformer':
                from .sherpa_impl import SherpaOnnxASR
                paraformer_params = sherpa_cfg.get('paraformer', {})
                return SherpaOnnxASR(config=paraformer_params, hotwords=hotwords)

            elif model_type == 'sense_voice':
                from .sherpa_sense_voice_impl import SherpaSenseVoiceASR
                sense_params = sherpa_cfg.get('sense_voice', {})
                return SherpaSenseVoiceASR(config=sense_params)

            else:
                raise ValueError(f"不支持的 Sherpa 模型类型: {model_type}")
        else:
            raise ValueError(f"未知的 ASR 引擎: {engine_type}")