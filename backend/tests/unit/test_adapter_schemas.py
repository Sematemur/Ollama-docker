import pytest
from tools.adapters import TavilyDirectAdapter


def test_tavily_adapter_has_to_langchain_tool():
    adapter = TavilyDirectAdapter(api_key="test-key")
    assert hasattr(adapter, "to_langchain_tool")
    assert callable(adapter.to_langchain_tool)


def test_tavily_langchain_tool_has_correct_name():
    adapter = TavilyDirectAdapter(api_key="test-key")
    tool = adapter.to_langchain_tool()
    assert tool.name == "tavily_search"


def test_tavily_langchain_tool_has_query_in_schema():
    adapter = TavilyDirectAdapter(api_key="test-key")
    tool = adapter.to_langchain_tool()
    schema = tool.args_schema.model_json_schema()
    assert "query" in schema["properties"]


def test_tavily_langchain_tool_description_mentions_web_search():
    adapter = TavilyDirectAdapter(api_key="test-key")
    tool = adapter.to_langchain_tool()
    assert "web" in tool.description.lower() or "search" in tool.description.lower()
