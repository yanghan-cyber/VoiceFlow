import threading
import queue
import sys
import numpy as np
try:
    import sounddevice as sd
except ImportError:
    sys.stderr.write("错误: 请运行 'pip install sounddevice numpy'\n")
    sys.exit(1)

from utils import get_logger

# 获取音频模块日志器
logger = get_logger("AudioRecorder")

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=0.1):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None

    def start(self):
        """开始录音"""
        if self.is_recording:
            return
        
        self.is_recording = True
        # 清空旧队列
        while not self.audio_queue.empty():
            self.audio_queue.get()

        # 启动 sounddevice 流
        # channels=1 (单声道), dtype='float32' (ASR通常需要)
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        self.stream.start()
        logger.info("🎤 麦克风已开启")

    def stop(self):
        """停止录音"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("🛑 麦克风已关闭")

    def _audio_callback(self, indata, frames, time, status):
        """此函数在后台线程运行"""
        if status:
            logger.error(f"Audio Error: {status}")
        
        if self.is_recording:
            # 必须拷贝数据，因为 indata 是复用的 buffer
            self.audio_queue.put(indata.copy().reshape(-1))

    def get_audio_chunk(self):
        """非阻塞获取音频块，如果没有数据返回None"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None