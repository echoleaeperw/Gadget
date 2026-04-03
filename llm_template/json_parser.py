"""
RobustJSONParser — 从 LLM 响应中提取 JSON 的鲁棒解析器

处理以下常见情况:
  - ```json ... ``` 代码块
  - 裸 JSON 文本
  - 不完整 JSON (截断)
  - 混杂在自然语言中的 JSON 片段
"""

import json
import re
from typing import Dict, Any, Optional


class RobustJSONParser:

    @staticmethod
    def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 响应文本中提取 JSON dict，失败返回 None。"""
        if not text or not isinstance(text, str):
            return None

        cleaned = RobustJSONParser._preprocess(text)

        for extractor in (
            RobustJSONParser._from_code_block,
            RobustJSONParser._from_bare_json,
            RobustJSONParser._from_incomplete,
        ):
            result = extractor(cleaned)
            if result:
                return result
        return None

    @staticmethod
    def _preprocess(text: str) -> str:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass
        return text

    @staticmethod
    def _from_code_block(text: str) -> Optional[Dict]:
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"`(\{.*?\})`",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _from_bare_json(text: str) -> Optional[Dict]:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _from_incomplete(text: str) -> Optional[Dict]:
        """尝试从截断文本中重建 key-value 对。"""
        m = re.search(
            r'"reasoning":\s*"([^"]*)".*?"risk_weights":\s*(\{[^}]*\})',
            text, re.DOTALL,
        )
        if m:
            try:
                return {"reasoning": m.group(1), "risk_weights": json.loads(m.group(2))}
            except json.JSONDecodeError:
                pass

        m2 = re.search(r'"risk_weights":\s*(\{[^}]*\})', text, re.DOTALL)
        if m2:
            try:
                return {"reasoning": "reconstructed", "risk_weights": json.loads(m2.group(1))}
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def validate_and_normalize(data: Dict[str, Any]) -> Dict[str, Any]:
        """确保 risk_weights 中的值都是 float。"""
        if not isinstance(data, dict):
            return {"error": "not a dict"}
        if "risk_weights" not in data:
            data = {"risk_weights": data}

        normalized = {}
        for k, v in data.get("risk_weights", {}).items():
            try:
                normalized[k] = float(v)
            except (ValueError, TypeError):
                continue
        data["risk_weights"] = normalized
        return data
