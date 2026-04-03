"""
Multi-Provider LLM Factory — 统一多 LLM 后端接口模板

支持后端:
  - DeepSeek (V3 / R1)
  - OpenAI (GPT-4o 等)
  - Anthropic / Claude
  - Google Gemini (Flash / Pro)
  - SiliconFlow (GLM 等国产模型)
  - POE (代理访问多种模型)

用法:
    from llm_template import BaseAgent, create_llm

    # 方式一: 直接创建 LLM 实例
    llm = create_llm(provider="deepseek", temperature=0.2)
    resp = llm.invoke([HumanMessage(content="Hello")])

    # 方式二: 继承 BaseAgent 构建自定义 Agent
    class MyAgent(BaseAgent):
        def run(self, prompt):
            from langchain_core.messages import HumanMessage
            return self.llm.invoke([HumanMessage(content=prompt)]).content

    agent = MyAgent(provider="deepseek", temperature=0)
    print(agent.run("什么是 CVAE?"))

配置:
    在同级目录或项目 configs/ 下放置 llm_config.json，格式见 llm_config.template.json
"""

import asyncio
import os
import json
from typing import Dict, Any, Optional

from pydantic.v1 import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import fastapi_poe as fp
    POE_AVAILABLE = True
except ImportError:
    POE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# POE wrapper (LangChain-compatible sync interface over async POE API)
# ─────────────────────────────────────────────────────────────────────────────

class _ChatPoe:
    """Minimal LangChain-compatible wrapper for POE API."""

    def __init__(self, api_key: str, bot_name: str,
                 temperature: Optional[float] = None,
                 proxy_url: Optional[str] = None):
        self.api_key = api_key
        self.bot_name = bot_name
        self.temperature = temperature
        self.proxy_url = proxy_url or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

    @staticmethod
    def _to_poe(messages) -> list:
        result = []
        for msg in messages:
            cls = type(msg).__name__
            if cls == "SystemMessage":
                role = "system"
            elif cls in ("HumanMessage", "human"):
                role = "user"
            elif cls in ("AIMessage", "ai"):
                role = "bot"
            else:
                role = "user"
            result.append(fp.ProtocolMessage(role=role, content=msg.content))
        return result

    def _make_session(self):
        try:
            import httpx
            if self.proxy_url:
                return httpx.AsyncClient(proxies={"https://": self.proxy_url,
                                                   "http://":  self.proxy_url})
            return httpx.AsyncClient()
        except ImportError:
            return None

    async def _acollect(self, messages) -> str:
        full = ""
        poe_msgs = self._to_poe(messages)
        session = self._make_session()
        kwargs = dict(messages=poe_msgs, bot_name=self.bot_name, api_key=self.api_key)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if session is not None:
            kwargs["session"] = session
        try:
            async for partial in fp.get_bot_response(**kwargs):
                if hasattr(partial, "text"):
                    full += partial.text
        finally:
            if session is not None:
                await session.aclose()
        return full

    def invoke(self, messages):
        content = asyncio.run(self._acollect(messages))

        class _Resp:
            def __init__(self, c):
                self.content = c
        return _Resp(content)

    def stream(self, messages):
        poe_msgs = self._to_poe(messages)

        async def _gen():
            session = self._make_session()
            kwargs = dict(messages=poe_msgs, bot_name=self.bot_name, api_key=self.api_key)
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if session is not None:
                kwargs["session"] = session
            try:
                async for partial in fp.get_bot_response(**kwargs):
                    yield partial
            finally:
                if session is not None:
                    await session.aclose()

        loop = asyncio.new_event_loop()
        agen = _gen()

        class _Chunk:
            def __init__(self, c):
                self.content = c
        try:
            while True:
                partial = loop.run_until_complete(agen.__anext__())
                yield _Chunk(getattr(partial, "text", ""))
        except StopAsyncIteration:
            pass
        finally:
            loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────────────────

class FullResponseCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.full_response += token


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "llm_config.json"),
    os.path.join(os.path.dirname(__file__), "..", "configs", "llm_config.json"),
    os.path.join(os.getcwd(), "configs", "llm_config.json"),
    os.path.join(os.getcwd(), "llm_config.json"),
]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load llm_config.json. Search order:
      1. Explicit config_path
      2. Same directory as this file
      3. ../configs/llm_config.json
      4. ./configs/llm_config.json
      5. ./llm_config.json
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    for p in _CONFIG_SEARCH_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)

    raise FileNotFoundError(
        "llm_config.json not found. Copy llm_config.template.json → llm_config.json and fill in API keys."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_llm(
    provider: Optional[str] = None,
    temperature: float = 0,
    verbose: bool = False,
    config_path: Optional[str] = None,
):
    """
    Create an LLM instance by provider name.

    Args:
        provider:    "deepseek" / "openai" / "claude" / "gemini-flash" / "siliconflow" / "poe" / ...
                     If None, uses default_provider from config.
        temperature: Generation temperature.
        verbose:     If True, attach streaming stdout callback.
        config_path: Explicit path to llm_config.json.

    Returns:
        A LangChain ChatModel (or compatible wrapper) with invoke() / stream() methods.
    """
    config = load_config(config_path)
    effective_provider = provider or config.get("default_provider", "deepseek")

    callbacks = []
    if verbose:
        callbacks.append(StreamingStdOutCallbackHandler())

    provider_config = config.get("providers", {}).get(effective_provider, {})
    api_key = provider_config.get("api_key") or os.getenv(f"{effective_provider.upper()}_API_KEY")
    base_url = provider_config.get("base_url")
    max_tokens = provider_config.get("max_tokens", 8192)

    if not api_key:
        raise ValueError(
            f"API Key not found for '{effective_provider}'. "
            f"Set it in llm_config.json or as env var {effective_provider.upper()}_API_KEY."
        )

    if effective_provider == "siliconflow":
        return ChatOpenAI(
            model=provider_config.get("model", "Pro/zai-org/GLM-4.7"),
            temperature=temperature,
            api_key=SecretStr(api_key),
            base_url=provider_config.get("base_url", "https://api.siliconflow.cn/v1"),
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

    elif effective_provider in ("openai", "gemini-flash", "gemini-pro"):
        return ChatOpenAI(
            model=provider_config.get("model", "gpt-4-turbo"),
            temperature=temperature,
            api_key=SecretStr(api_key),
            base_url=base_url,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

    elif effective_provider in ("deepseek", "deepseek-r1"):
        model_name = provider_config.get("model", "deepseek-chat")
        extra = {}
        if model_name == "deepseek-reasoner":
            extra["temperature"] = 1
        else:
            extra["temperature"] = temperature
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=provider_config.get("base_url", "https://api.deepseek.com/v1"),
            max_tokens=max_tokens,
            callbacks=callbacks,
            **extra,
        )

    elif effective_provider in ("anthropic", "claude"):
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("pip install langchain-anthropic")
        model_name = provider_config.get("model", "claude-3-sonnet-20240229")
        anthropic_base_url = provider_config.get("base_url", "https://api.anthropic.com")
        if anthropic_base_url.endswith("/v1"):
            anthropic_base_url = anthropic_base_url[:-3]
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=SecretStr(api_key),
            base_url=anthropic_base_url,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

    elif effective_provider == "poe":
        if not POE_AVAILABLE:
            raise ValueError("pip install fastapi-poe")
        return _ChatPoe(
            api_key=api_key,
            bot_name=provider_config.get("bot_name", "Claude-3.7-Sonnet"),
            temperature=temperature if temperature != 0 else None,
            proxy_url=provider_config.get("proxy_url"),
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {effective_provider}")


# ─────────────────────────────────────────────────────────────────────────────
# BaseAgent — inherit to build custom agents
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Base class for LLM-powered agents.

    Usage:
        class MyAgent(BaseAgent):
            def run(self, prompt: str) -> str:
                from langchain_core.messages import SystemMessage, HumanMessage
                msgs = [SystemMessage(content="You are helpful."),
                        HumanMessage(content=prompt)]
                return self.llm.invoke(msgs).content

        agent = MyAgent(provider="deepseek")
        print(agent.run("Hello"))
    """

    def __init__(
        self,
        temperature: float = 0,
        verbose: bool = False,
        provider: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        self.verbose = verbose
        self.config = load_config(config_path)
        self.llm = create_llm(
            provider=provider,
            temperature=temperature,
            verbose=verbose,
            config_path=config_path,
        )

    def stream_response(self, messages) -> str:
        """Stream LLM response and return full text."""
        full = ""
        for chunk in self.llm.stream(messages):
            piece = chunk.content if hasattr(chunk, "content") else str(chunk)
            if isinstance(piece, str) and piece:
                full += piece
                print(piece, end="", flush=True)
        print()
        return full
