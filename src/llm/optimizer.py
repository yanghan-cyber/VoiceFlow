import os
from openai import OpenAI
from utils import get_logger

logger = get_logger("LLMOptimizer")

class LLMOptimizer:
    def __init__(self, config: dict):
        """
        初始化 LLM 优化器
        :param config: 完整的配置字典
        """
        llm_config = config.get("llm", {})
        
        # 1. 获取基础配置
        self.api_key = llm_config.get("api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.base_url = llm_config.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        self.model_name = llm_config.get("model", "gpt-3.5-turbo")
        self.temperature = llm_config.get("temperature", 0.3)
        
        # 2. 自动提取热词 (从 ASR 配置中获取，实现联动)
        # 尝试路径: asr -> sherpa_onnx -> sense_voice -> hotwords
        self.hotwords = []
        try:
            asr_cfg = config.get("asr", {})
            if "sherpa_onnx" in asr_cfg:
                sv_cfg = asr_cfg.get("sherpa_onnx", {}).get("sense_voice", {})
                self.hotwords.extend(sv_cfg.get("hotwords", []))
        except Exception:
            pass
        
        # 也可以从 llm 配置里额外读取专门给 LLM 看的热词
        self.hotwords.extend(llm_config.get("hotwords", []))
        self.hotwords = list(set(self.hotwords)) # 去重

        # 3. 构建高阶 System Prompt
        # 如果配置文件里没有写死 system_prompt，则自动构建一个更智能的
        custom_prompt = llm_config.get("system_prompt", "")
        if custom_prompt:
            self.system_prompt = custom_prompt
        else:
            self.system_prompt = self._build_system_prompt()
        
        logger.info(f"LLM 初始化 | Model: {self.model_name} | Temp: {self.temperature}")
        if self.hotwords:
            logger.info(f"LLM 已感知热词: {len(self.hotwords)} 个")
        
        # 4. 初始化客户端
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=15.0 # 润色稍微多给点时间
            )
        except Exception as e:
            logger.error(f"OpenAI Client 初始化失败: {e}")
            self.client = None

    def _build_system_prompt(self) -> str:
        """根据配置自动组装 System Prompt"""
        
        # 热词部分提示
        hotwords_instruction = ""
        if self.hotwords:
            hotwords_str = "、".join(self.hotwords)
            hotwords_instruction = f"\n2. **热词感知**：上下文中可能包含以下特定领域热词：[{hotwords_str}]。如果识别文本中出现读音相近的词，请优先修正为上述热词。"

        prompt = (
            "你是一个专业的语音输入法智能纠错助手。你的输入源是语音识别（ASR）生成的原始文本。"
            "\n请严格遵守以下指令进行处理："
            "\n1. **场景认知**：这是语音转文字的内容，可能包含同音错别字、口语语气词（如“呃”、“那个”）或断句错误。"
            f"{hotwords_instruction}"
            "\n3. **核心任务**："
            "\n   - 纠正错别字和标点符号错误。"
            "\n   - 优化表达逻辑，使其通顺流畅。"
            "\n   - **绝对禁止**改变原文的核心语义和意图。"
            "\n4. **输出规范**：仅输出润色后的纯文本结果。**严禁**包含任何解释、前缀（如“修正后：”）、引号或无关的寒暄。"
        )
        return prompt

    def optimize(self, text: str) -> str:
        """
        执行文本优化
        """
        if not self.client or not text or len(text.strip()) < 1:
            return text

        try:
            # 构造消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
            
            # 发起请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048, 
            )
            
            # 提取结果
            content = response.choices[0].message.content.strip()
            if not content:
                return text
            logger.info(f"LLM 优化: [原] {text} -> [新] {content}")
            return content

        except Exception as e:
            logger.error(f"LLM 请求失败: {e}")
            return text # 失败返回原文