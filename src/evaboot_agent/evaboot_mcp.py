from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any


class EvabootMCPError(RuntimeError):
    pass


class EvabootMCPAdapter:
    """Thin client for Evaboot's hosted MCP tools; it does not scrape LinkedIn."""

    endpoint = "https://mcp.evaboot.com/mcp"

    def __init__(self) -> None:
        self.api_key = os.getenv("EVABOOT_API_KEY", "").strip()
        if not self.api_key:
            raise EvabootMCPError("Set EVABOOT_API_KEY to use the optional Evaboot MCP adapter.")

    async def _invoke(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise EvabootMCPError("Install the optional dependency with `pip install -e '.[live]'`.") from exc

        async with streamable_http_client(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=25,
            sse_read_timeout=35,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                if name == "__list_tools__":
                    result = await session.list_tools()
                    return result.tools
                result = await session.call_tool(name, arguments or {})
                if getattr(result, "isError", False):
                    detail = " ".join(getattr(item, "text", "") for item in result.content)
                    raise EvabootMCPError(detail[:500] or f"Evaboot MCP tool {name} failed")
                structured = getattr(result, "structuredContent", None)
                if structured is not None:
                    return structured
                texts = [getattr(item, "text", "") for item in getattr(result, "content", [])]
                raw = "\n".join(text for text in texts if text)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"text": raw}

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        try:
            return asyncio.run(self._invoke(name, arguments))
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" in str(exc):
                raise EvabootMCPError("The live MCP adapter must run from a synchronous worker thread.") from exc
            raise

    def tool_metadata(self) -> dict[str, Any]:
        tools = self.invoke("__list_tools__")
        return {tool.name: getattr(tool, "inputSchema", {}) for tool in tools}

    @staticmethod
    def _find_key(schema: dict[str, Any], names: tuple[str, ...]) -> str:
        properties = schema.get("properties", {})
        for name in names:
            if name in properties:
                return name
        required = schema.get("required", [])
        string_required = [name for name in required if properties.get(name, {}).get("type") == "string"]
        if len(string_required) == 1:
            return string_required[0]
        raise EvabootMCPError("Could not map this Evaboot MCP tool's current input schema safely.")

    def build_search(self, prompt: str) -> dict[str, Any]:
        metadata = self.tool_metadata()
        tool = "build_sn_search_from_prompt"
        if tool not in metadata:
            raise EvabootMCPError("The connected Evaboot server does not expose build_sn_search_from_prompt.")
        key = self._find_key(metadata[tool], ("prompt", "search_prompt", "query", "natural_language_query"))
        result = self.invoke(tool, {key: prompt})
        return result if isinstance(result, dict) else {"search": result}

    def preflight_quota(self, max_results: int) -> dict[str, Any]:
        result = self.invoke("get_daily_export_quota", {})
        flattened: list[tuple[str, int]] = []

        def visit(value: Any, prefix: str = "") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{prefix}.{key}" if prefix else str(key))
            elif isinstance(value, list):
                for item in value:
                    visit(item, prefix)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flattened.append((prefix.lower(), int(value)))

        visit(result)
        remaining = [value for key, value in flattened if any(word in key for word in ("remaining", "available", "quota"))]
        credits = [value for key, value in flattened if "credit" in key and any(word in key for word in ("balance", "remaining", "available"))]
        if not remaining or not credits:
            raise EvabootMCPError("Quota response could not be verified; export was not started.")
        if min(remaining) < max_results or min(credits) < max_results:
            raise EvabootMCPError("Evaboot export quota or credit balance is below the requested lead cap.")
        return {"quota_verified": True, "available_quota": min(remaining), "available_credits": min(credits)}

    def start_export(self, search: dict[str, Any], max_results: int) -> dict[str, Any]:
        self.preflight_quota(max_results)
        metadata = self.tool_metadata()
        tool = "export_sn_search_or_list"
        if tool not in metadata:
            raise EvabootMCPError("The connected Evaboot server does not expose export_sn_search_or_list.")
        schema = metadata[tool]
        props = schema.get("properties", {})
        args: dict[str, Any] = {}
        url = _find_string(search, ("search_url", "sales_navigator_url", "url"))
        url_key = self._find_key(schema, ("search_url", "url", "sales_navigator_url"))
        args[url_key] = url
        max_key = next((key for key in ("max_results", "limit", "max_leads") if key in props), None)
        if not max_key:
            raise EvabootMCPError("Evaboot export input schema has no bounded max_results field.")
        args[max_key] = max_results
        if "enrich_email" in props:
            args["enrich_email"] = False
        result = self.invoke(tool, args)
        if not isinstance(result, dict):
            raise EvabootMCPError("Evaboot did not return a structured export job.")
        return result

    def get_export(self, job_id: str) -> Any:
        metadata = self.tool_metadata()
        tool = "get_export"
        if tool not in metadata:
            raise EvabootMCPError("The connected Evaboot server does not expose get_export.")
        key = self._find_key(metadata[tool], ("extraction_id", "export_id", "id"))
        return self.invoke(tool, {key: job_id})

    def verify_email(self, email: str) -> Any:
        metadata = self.tool_metadata()
        tool = "verify_email_single"
        if tool not in metadata:
            raise EvabootMCPError("The connected Evaboot server does not expose verify_email_single.")
        key = self._find_key(metadata[tool], ("email", "email_address"))
        return self.invoke(tool, {key: email})


def _find_string(value: Any, keys: tuple[str, ...]) -> str:
    if isinstance(value, dict):
        for key in keys:
            if isinstance(value.get(key), str) and value[key].startswith("http"):
                return value[key]
        for child in value.values():
            found = _find_string(child, keys)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_string(child, keys)
            if found:
                return found
    elif isinstance(value, str) and re.match(r"https?://", value):
        return value
    return ""
