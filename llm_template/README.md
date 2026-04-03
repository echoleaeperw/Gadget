# LLM Template — 多后端 LLM 统一接口模板

将此文件夹复制到任意 Python 项目即可使用。

## 支持的 LLM Provider

| Provider | 模型示例 | 说明 |
|----------|---------|------|
| `deepseek` | deepseek-chat (V3) | 国内直连，高性价比 |
| `deepseek-r1` | deepseek-reasoner | 推理模型，慢但深度思考 |
| `openai` | gpt-4o, gpt-4-turbo | OpenAI 官方 API |
| `claude` | claude-sonnet-4 | Anthropic Claude 系列 |
| `gemini-flash` | gemini-2.0-flash | Google，1M 上下文，免费 |
| `gemini-pro` | gemini-2.5-pro | Google，2M 上下文 |
| `siliconflow` | GLM-5, Qwen | 硅基流动，国产模型聚合 |
| `poe` | 任意 POE Bot | 代理访问，需翻墙 |

## 快速开始

### 1. 配置

```bash
cp llm_config.template.json llm_config.json
# 编辑 llm_config.json，填入 API Key
```

### 2. 安装依赖

```bash
pip install langchain langchain-openai langchain-core pydantic
# 如需 Claude: pip install langchain-anthropic
# 如需 POE:    pip install fastapi-poe httpx
```

### 3. 使用

```python
# 方式一: 直接创建 LLM
from llm_template import create_llm
from langchain_core.messages import HumanMessage

llm = create_llm(provider="deepseek", temperature=0.2)
resp = llm.invoke([HumanMessage(content="什么是强化学习?")])
print(resp.content)

# 方式二: 继承 BaseAgent
from llm_template import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage

class QAAgent(BaseAgent):
    def ask(self, question: str) -> str:
        msgs = [
            SystemMessage(content="你是一个 AI 专家。"),
            HumanMessage(content=question),
        ]
        return self.llm.invoke(msgs).content

agent = QAAgent(provider="deepseek", temperature=0)
print(agent.ask("Transformer 的自注意力机制原理?"))

# 方式三: 流式输出
from llm_template import BaseAgent
from langchain_core.messages import HumanMessage

agent = BaseAgent(provider="gemini-flash")
full_text = agent.stream_response([HumanMessage(content="写一首诗")])
```

### 4. 解析 LLM 的 JSON 响应

```python
from llm_template import RobustJSONParser

raw = '''
Here is my analysis:
```json
{"reasoning": "...", "risk_weights": {"ttc": 3.0, "adv_crash": 7.0}}
```
'''
result = RobustJSONParser.extract_json_from_response(raw)
print(result)  # {"reasoning": "...", "risk_weights": {...}}
```

## 文件结构

```
llm_template/
├── __init__.py                  # 导出 BaseAgent, create_llm, RobustJSONParser
├── llm_factory.py               # LLM 工厂 + BaseAgent 基类
├── json_parser.py               # 鲁棒 JSON 解析器
├── llm_config.template.json     # 配置模板 (复制后填入 Key)
└── README.md                    # 本文件
```

## 切换 Provider

只需修改 `llm_config.json` 中的 `default_provider`，或在代码中指定 `provider` 参数：

```python
llm_ds = create_llm(provider="deepseek")
llm_gpt = create_llm(provider="openai")
llm_claude = create_llm(provider="claude")
```
