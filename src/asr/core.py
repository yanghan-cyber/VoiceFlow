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
    def _get_advanced_config() -> dict:
        """获取高级配置（如果存在）"""
        advanced_path = "config.advanced.yaml"
        if os.path.exists(advanced_path):
            with open(advanced_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    @staticmethod
    def _merge_configs(main_config: dict, advanced_config: dict) -> dict:
        """合并主配置和高级配置"""
        if not advanced_config:
            return main_config

        # 使用高级配置中的模型路径和详细参数
        merged = main_config.copy()

        # 合并 ASR 配置
        if 'asr' in advanced_config:
            merged['asr'] = advanced_config['asr']

        return merged

    @staticmethod
    def get_asr_engine(config_path: str = "config.yaml") -> ASRBase:
        """
        根据配置创建 ASR 引擎
        现在直接根据 app.asr_model 选择对应的实现
        """
        full_config = ASRFactory._load_config(config_path)
        advanced_config = ASRFactory._get_advanced_config()
        config = ASRFactory._merge_configs(full_config, advanced_config)

        # 从主配置获取模型类型
        app_config = config.get('app', {})
        asr_model = app_config.get('asr_model', 'paraformer')

        # 获取热词配置（如果有）
        asr_config = config.get('asr', {})
        hotwords_config = asr_config.get('hotwords', {})

        if asr_model == "paraformer":
            from .sherpa_impl import SherpaOnnxASR
            # 优先使用高级配置中的详细参数
            if 'asr' in config and 'models' in config['asr'] and 'paraformer' in config['asr']['models']:
                advanced_paraformer = config['asr']['models']['paraformer']
                # 转换配置格式：从 streaming 中提取模型路径
                paraformer_params = advanced_paraformer.get('streaming', {}).copy()
                # 添加标点配置
                if 'punctuation' in advanced_paraformer:
                    paraformer_params['punctuation'] = advanced_paraformer['punctuation']
                # 添加性能参数
                if 'performance' in config['asr']:
                    performance = config['asr']['performance']
                    for key in ['num_threads', 'provider', 'device', 'sample_rate', 'feature_dim',
                               'enable_endpoint_detection', 'rule1_min_trailing_silence',
                               'rule2_min_trailing_silence', 'rule3_min_utterance_length',
                               'decoding_method', 'debug']:
                        if key in performance:
                            paraformer_params[key] = performance[key]
            else:
                # 使用默认配置
                paraformer_params = {
                    'tokens': "./ckpts/sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt",
                    'encoder': "./ckpts/sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx",
                    'decoder': "./ckpts/sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx",
                    'punctuation': {
                        'enabled': True,
                        'model': "./ckpts/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"
                    }
                }
            # 添加热词配置（只添加必要的参数）
            if hotwords_config:
                # 只添加热词相关的参数，不要覆盖ASR核心参数
                hotword_params = {}
                for key in ['hr_dict_dir', 'hr_lexicon', 'rule_fsts', 'rule_fars']:
                    if key in hotwords_config:
                        hotword_params[key] = hotwords_config[key]
                paraformer_params.update(hotword_params)
            return SherpaOnnxASR(config=paraformer_params)

        elif asr_model == "sense_voice":
            from .sherpa_sense_voice_impl import SherpaSenseVoiceASR
            # 优先使用高级配置中的详细参数
            if 'asr' in config and 'models' in config['asr'] and 'sense_voice' in config['asr']['models']:
                sense_params = config['asr']['models']['sense_voice'].copy()
                # 添加性能参数
                if 'performance' in config['asr']:
                    performance = config['asr']['performance']
                    for key in ['num_threads', 'provider', 'device', 'debug']:
                        if key in performance:
                            sense_params[key] = performance[key]
            else:
                # 使用默认配置
                sense_params = {
                    'model': "./ckpts/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
                    'tokens': "./ckpts/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
                    'language': "auto",
                    'use_itn': True,
                    'vad': {
                        'model': "./ckpts/vad/ten-vad.onnx"
                    }
                }
            # 添加热词配置（只添加必要的参数）
            if hotwords_config:
                # 只添加热词相关的参数，不要覆盖ASR核心参数
                hotword_params = {}
                for key in ['hr_dict_dir', 'hr_lexicon', 'rule_fsts', 'rule_fars']:
                    if key in hotwords_config:
                        hotword_params[key] = hotwords_config[key]
                sense_params.update(hotword_params)
            return SherpaSenseVoiceASR(config=sense_params)

        else:
            raise ValueError(f"不支持的 ASR 模型类型: {asr_model}")