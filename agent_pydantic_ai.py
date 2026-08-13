import os
import subprocess
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

# 使用SDK换模型只需更改Provider 类和模型名。以下是3个常见例子。
# # DeepSeek
# from pydantic_ai.providers.deepseek import DeepSeekProvider
# model = OpenAIChatModel('deepseek-v4-flash',provider=DeepSeekProvider(api_key=API_KEY))

# # OpenAI
# from pydantic_ai.providers.openai import OpenAIProvider
# model = OpenAIChatModel('gpt-5.5',provider=OpenAIProvider(api_key=API_KEY))

# # Anthropic
# from pydantic_ai.models.anthropic import AnthropicModel
# from pydantic_ai.providers.anthropic import AnthropicProvider
# model = AnthropicModel('claude-sonnet-4-6',provider=AnthropicProvider(api_key=API_KEY))

# 加载 .env 文件中的变量到环境变量中
load_dotenv()

# 从环境变量读取，避免把 API Key 硬编码到代码里
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("请先设置环境变量 API_KEY")

# 显式构造 model + provider，后续想换其他模型
# 只需换模型名称和对应的 Provider 类即可
# 默认使用deepseek官方，若是连接自己的中转站则还需添加BASE_URL
model = OpenAIChatModel(
    'deepseek-v4-flash',
    provider=DeepSeekProvider(api_key=API_KEY),
)

# Agent(.) 相当于把你手写版的tools、tool_functions、SYSTEM_PROMPT等等内容装进一个对象
agent = Agent(
    model,
    # instructions 就是手写版里的 System Prompt，SDK 自动放到上下文开头
    instructions="你是一个编程助手。你可以读写文件和执行命令来帮用户完成编程任务。\n"
                 "工作流程：先理解需求，写代码，然后运行验证。如果有错误就修复并重新运行，直到确认正确。",
)

# --- 工具定义 ---
# @agent.tool_plain用于普通函数，不需要 Agent 的 RunContext。@agent.tool用于函数需要 RunContext。
@agent.tool_plain
def read_file(path: str) -> str:
    """读取指定文件的内容。

    Args:
        path: 文件路径
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"

@agent.tool_plain
def write_file(path: str, content: str) -> str:
    """将内容写入指定文件。

    Args:
        path: 文件路径
        content: 要写入的内容
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}"

@agent.tool_plain
def run_command(command: str) -> str:
    """执行一条 shell 命令并返回输出。

    Args:
        command: 要执行的命令
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, errors="replace", timeout=10
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[错误] {result.stderr}"
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return "[错误] 命令执行超时（10秒）"

# 不再需要手写版的tool_functions了

# --- 启动 Agent ---
# message_history 跨轮次保留上下文，让 Agent 记得之前做过什么
history = []
print("编程 Agent 已启动，输入任务开始，输入 q 退出\n")

while True:
    user_input = input("你: ")
    if user_input.strip() == "q":
        break

    # run_sync 内部跑 Agent 循环：调模型、执行工具、把结果喂回上下文，直到模型给出最终回复
    result = agent.run_sync(user_input, message_history=history) #手写版的内层 Agent 循环被这一行代码代替了
    # history本身返回包含旧历史 + 本轮新增消息的完整对话历史，下一轮模型就能看到之前的所有操作
    history = result.all_messages()
    print(f"AI: {result.output}\n")