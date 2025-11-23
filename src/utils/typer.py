import sys
import time
try:
    import keyboard
except ImportError:
    sys.stderr.write("错误: 请运行 'pip install keyboard'\n")
    sys.exit(1)

# 获取日志器
from .log_utils import get_logger
logger = get_logger("TextTyper")

class TextTyper:
    def __init__(self):
        # 记录当前屏幕上可以通过 Backspace 撤回的文字内容
        self.current_content = "" 

    def _get_common_prefix_len(self, str1: str, str2: str) -> int:
        """计算两个字符串的公共前缀长度"""
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return i
        return min_len

    def show_status(self, text: str):
        """
        显示状态提示 (如 '(( 🎤 ))')
        状态提示通常是完全替换，所以直接全删全写
        """
        self.clear_temp() # 先清除旧的
        keyboard.write(text)
        self.current_content = text

    def update_stream(self, new_text: str):
        """
        流式更新文字 (智能增量更新)
        """
        if new_text == self.current_content:
            return

        # 1. 计算公共前缀长度
        # 例如: old="ABC", new="ABD" -> prefix="AB" (len=2)
        common_len = self._get_common_prefix_len(self.current_content, new_text)
        
        # 2. 计算需要删除的字符数
        # delete_count = len("ABC") - 2 = 1 (删除 'C')
        delete_count = len(self.current_content) - common_len
        
        # 3. 计算需要输入的字符数
        # input_text = "ABD"[2:] = "D"
        input_text = new_text[common_len:]

        # 4. 执行操作
        if delete_count > 0:
            for _ in range(delete_count):
                keyboard.send('backspace')
        
        if input_text:
            # delay=0.005 让字有一个极短的间隔出现，模拟打字感
            keyboard.write(input_text, delay=0.005)

        # 5. 更新内部状态
        self.current_content = new_text

    def commit_text(self, text: str):
        """
        提交最终结果 (确定上屏)
        通常这步是修正一下最后的标点，然后清空内部状态
        """
        # 复用 update_stream 的逻辑把文字修正成最终形态
        self.update_stream(text)
        
        # 提交后，这段文字就不归我们管了（不能再自动删除了）
        self.current_content = ""

    def clear_temp(self):
        """
        清除当前的临时内容 (全删)
        """
        if len(self.current_content) > 0:
            for _ in range(len(self.current_content)):
                keyboard.send('backspace')
            self.current_content = ""