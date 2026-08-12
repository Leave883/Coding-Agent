import json
import os
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的变量到环境变量中
load_dotenv()

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
if not API_KEY or not BASE_URL:
    raise RuntimeError("请先设置环境变量 API_KEY 和 BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 工具描述 ---，依旧使用Json Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定文件（会覆盖已有内容）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "执行一条 shell 命令并返回输出结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"]
            }
        }
    }
]

# --- 工具实现 ---
def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}"

# subprocess.run 是 Python 标准库里执行外部命令的函数，shell=True 让它通过 shell 解释器执行命令字符串，
# capture_output=True 捕获命令的输出和错误信息，text=True 让返回结果是字符串而不是原始字节。
# 此处给了模型很大的的 Shell 执行权限，工程使用时shell=True要谨慎使用。
def run_command(command):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, errors="replace", timeout=10
        )
        output = result.stdout
        # shell 惯例：返回码 0 表示成功，非 0 表示出错
        if result.returncode != 0:
            output += f"\n[错误] {result.stderr}"
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return "[错误] 命令执行超时（10秒）"

tool_functions = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

# --- Agent 主循环 ---
SYSTEM_PROMPT = """你是一个编程助手。你可以读写文件和执行命令来帮用户完成编程任务。
工作流程：先理解需求，写代码，然后运行验证。如果发现错误就修复并重新运行，直到确认正确。"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
print("编程 Agent 已启动，输入任务开始，输入 q 退出\n")

while True:
    user_input = input("你: ")
    if user_input.strip() == "q":
        break

    messages.append({"role": "user", "content": user_input})

    # Agent 循环
    while True:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tools,
            # 开启思考模式，让模型每一步决策前先推理
            extra_body={"reasoning_effort": "high"}
        )
        assistant_message = response.choices[0].message

        # 不管这轮是否调用工具，assistant 消息都要先追加到上下文
        messages.append(assistant_message)

        # assistant 消息可以同时带文本和工具调用，这轮模型说的话先显示出来
        if assistant_message.content:
            print(f"AI: {assistant_message.content}\n")

        # 模型不再请求工具调用，说明它准备好回复了，退出循环
        if not assistant_message.tool_calls:
            break

        # 模型可能请求多个工具，依次执行
        for tool_call in assistant_message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            func = tool_functions[tool_call.function.name]
            result = func(**args)
            # 打印工具调用过程，方便观察 Agent 的行为
            print(f"  [工具] {tool_call.function.name}({args})")
            # 只打印前 200 个字符，避免大文件输出刷屏
            print(f"  [结果] {result[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })