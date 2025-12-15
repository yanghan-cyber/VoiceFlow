import sys
import time
try:
    import keyboard
    import pyclip
except ImportError as e:
    missing = str(e).split()[-1]
    sys.stderr.write(f"错误: 请运行 'pip install {missing}'\n")
    sys.exit(1)

from .log_utils import get_logger
logger = get_logger("TextTyper")

class TextTyper:
    def __init__(self):
        # 记录当前屏幕上可以通过定位删除的文字内容
        self.current_content = ""

    def _get_common_prefix_len(self, str1: str, str2: str) -> int:
        """计算两个字符串的公共前缀长度"""
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return i
        return min_len

    def _paste_with_retry(self, text: str, retries=3):
        """带重试机制的粘贴操作，防止剪贴板由于占用而失败"""
        for i in range(retries):
            try:
                # # 1. 保存旧剪贴板
                # try:
                #     original_clipboard = pyclip.paste(text=False)
                # except Exception:
                #     original_clipboard = None
                
                # 2. 写入新内容
                # 注意：某些系统下，copy需要一点时间生效
                pyclip.copy(text)
                time.sleep(0.05) 

                # 3. 模拟粘贴
                keyboard.send('ctrl+v')
                time.sleep(0.05) # 等待粘贴动作完成

                # # 4. 恢复旧剪贴板 (可选，为了不干扰用户体验)
                # if original_clipboard:
                #     # pyclip.paste 返回 bytes，copy 需要 str (utf-8) 或 bytes
                #     content = original_clipboard
                #     if isinstance(content, bytes):
                #         try:
                #             content = content.decode('utf-8')
                #         except:
                #             pass # 如果是二进制数据，直接传bytes通常也可以，或者就放弃恢复
                #     pyclip.copy(content)
                
                return # 成功则退出
            except Exception as e:
                logger.warning(f"粘贴失败 (尝试 {i+1}/{retries}): {e}")
                time.sleep(0.1)
        
        # 如果重试都失败了，降级为模拟打字
        logger.error("剪贴板粘贴彻底失败，降级为模拟按键")
        keyboard.write(text)

    def show_status(self, text: str):
        """显示状态提示 (覆盖模式)"""
        self.clear_temp()
        self._paste_with_retry(text)
        self.current_content = text

    def update_stream(self, new_text: str):
        """流式更新 (增量模式)"""
        if new_text == self.current_content:
            return

        common_len = self._get_common_prefix_len(self.current_content, new_text)
        delete_count = len(self.current_content) - common_len
        input_text = new_text[common_len:]

        # 1. 删除旧字符
        if delete_count > 0:
            for _ in range(delete_count):
                keyboard.send('backspace')

        # 2. 粘贴新字符
        if input_text:
            self._paste_with_retry(input_text)

        self.current_content = new_text

    def commit_text(self, text: str):
        """提交最终结果"""
        self.update_stream(text)
        self.current_content = "" # 清空记录，表示这就常驻屏幕了

    def clear_temp(self):
        """清除临时内容"""
        if len(self.current_content) > 0:
            for _ in range(len(self.current_content)):
                keyboard.send('backspace')
            self.current_content = ""