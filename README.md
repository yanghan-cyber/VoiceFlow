# 🎤 语音输入法 (Voice Input Method)

一个基于 Sherpa-ONNX 的实时语音输入法，支持流式和非流式两种模式，可以全局使用快捷键进行语音转文字输入。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ 特性

- 🎯 **双模式支持**：流式实时识别 + 离线整句识别
- 🌍 **多语言支持**：中文、英文、日语、韩语、粤语
- ⌨️ **全局热键**：F2 长按即可开始语音输入
- 🔄 **实时流式**：边说边显示，智能增量更新
- 📝 **标点恢复**：自动添加标点符号
- 🖥️ **跨平台**：Windows、macOS、Linux 支持

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 麦克风设备

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/voiceinput.git
cd voiceinput

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载模型

#### 下载模型

```bash
# 创建模型目录
mkdir -p ckpts
mkdir -p ckpts/vad
mkdir -p ckpts/hr-files

# 下载 ASR 模型
cd ckpts
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

# 下载 Paraformer 模型（推荐流式模式使用）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

# 下载标点恢复模型
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

# 下载 VAD 模型
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx -O vad/ten-vad.onnx

# 下载热词文件
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
tar xf dict.tar.bz2 -C hr-files/
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst -O hr-files/replace.fst
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt -O hr-files/lexicon.txt

cd ..
```

### 4. 配置

编辑 `config.yaml` 文件，确保模型路径正确：

```yaml
app:
  mode: "stream"  # "stream" 流式模式 或 "offline" 离线模式

asr:
  active_engine: "sherpa_onnx"
  sherpa_onnx:
    model_type: "paraformer"  # 或 "sense_voice"
    # ... 其他配置项
```

### 5. 运行

```bash
python src/main.py
```

运行后，将光标放在任意输入框中，**长按 F2 键**开始语音输入。

## 📖 使用说明

### 基本操作

1. **启动程序**：运行 `python src/main.py`
2. **开始输入**：长按 F2 键，看到 `(( 🎤 ))` 提示后开始说话
3. **结束输入**：松开 F2 键，文字将自动输入到光标位置
4. **退出程序**：按 ESC 键退出

### 模式选择

#### 流式模式 (`stream`)

- 特点：边说边识别，实时显示文字
- 适用：长文本输入，需要实时反馈的场景
- 体验：类似 Siri 的实时语音转文字

#### 离线模式 (`offline`)

- 特点：录音完成后整体识别
- 适用：短文本输入，网络不稳定的情况
- 体验：传统录音笔模式

## 🔧 配置说明

### ASR 模型配置

#### Paraformer 模型（推荐流式使用）

```yaml
sherpa_onnx:
  model_type: "paraformer"
  paraformer:
    tokens: "./ckpts/sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt"
    encoder: "./ckpts/sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx"
    decoder: "./ckpts/sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx"
    # ... 其他配置
```

#### SenseVoice 模型（多语言支持）

```yaml
sherpa_onnx:
  model_type: "sense_voice"
  sense_voice:
    model: "./ckpts/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx"
    tokens: "./ckpts/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"
    language: "auto"  # 自动检测语言
    use_itn: true     # 逆文本标准化
    # ... 其他配置
```

### VAD 配置

```yaml
vad:
  type: "ten_vad"        # 或 "silero_vad"
  model: "./ckpts/vad/ten-vad.onnx"
  threshold: 0.5
  min_silence_duration: 0.1
  min_speech_duration: 0.25
```

## 🎯 热词配置

想要提高特定领域词汇的识别准确率？可以配置热词来优化识别效果。

详细的热词提取和配置方法请参考：

- **官方文档**：[Sherpa-ONNX Hotword Replacer](https://k2-fsa.github.io/sherpa/onnx/homophone-replacer/index.html)

🛠️ 开发说明

### 项目结构

```
voiceinput/
├── src/
│   ├── asr/                 # ASR 引擎
│   │   ├── core.py         # 核心接口和工厂类
│   │   ├── sherpa_impl.py  # Paraformer 实现
│   │   └── sherpa_sense_voice_impl.py
│   ├── audio/              # 音频处理
│   │   └── recorder.py
│   ├── hotkeys/            # 热键管理
│   │   └── hotkey_manager.py
│   ├── utils/              # 工具类
│   │   └── typer.py        # 智能文本输入
│   └── main.py             # 主程序
├── config.yaml             # 配置文件
├── ckpts/                  # 模型文件目录
├── examples/               # 示例脚本
└── download.md             # 下载说明
```

### 自定义开发

#### 添加新的 ASR 引擎

1. 继承 `ASRBase` 类
2. 实现抽象方法
3. 在 `ASRFactory` 中注册新引擎

```python
class CustomASR(ASRBase):
    def start_stream(self):
        # 实现流式开始
        pass

    def feed_audio(self, samples, sample_rate):
        # 实现音频输入
        pass

    def stop_stream(self) -> str:
        # 实现流式停止
        pass

    def transcribe_offline(self, samples, sample_rate) -> str:
        # 实现离线识别
        pass
```

## ❓ 常见问题

### Q: Linux 下权限不足？

A: 需要管理员权限来监听全局热键，请使用 `sudo` 运行程序。

### Q: 识别准确率不高？

A:

1. 检查麦克风质量和环境噪音
2. 尝试不同的 ASR 模型
3. 配置领域相关的热词
4. 调整 VAD 参数

### Q: 热键不生效？

A:

1. 检查是否有其他程序占用 F2 键
2. 确认程序有足够的系统权限
3. 在 Linux 上可能需要 root 权限

### Q: 模型下载失败？

A: 可以尝试以下镜像源或手动下载：

- 使用代理或 VPN
- 从 GitHub Releases 页面手动下载
- 使用国内镜像源

## 📝 更新日志

### v1.0.0

- 初始版本发布
- 支持流式和离线两种模式
- 集成 Paraformer 和 SenseVoice 模型
- 支持热词配置
- 全局热键功能

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发环境搭建

```bash
# 克隆项目
git clone https://github.com/yourusername/voiceinput.git
cd voiceinput

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 运行测试
python src/main.py
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) - 优秀的语音识别框架
- [Paraformer](https://github.com/alibaba-damo-academy/FunASR) - 阿里巴巴开源的语音识别模型
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 多语言语音识别模型

如果这个项目对你有帮助，请给个 ⭐ Star！
