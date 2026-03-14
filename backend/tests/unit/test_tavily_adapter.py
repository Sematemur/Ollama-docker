from tools.adapters import TavilyDirectAdapter


def test_tavily_readiness_error_when_api_key_missing():
    adapter = TavilyDirectAdapter(api_key=None)
    err = adapter.readiness_error()
    assert err is not None
    assert "Tavily API key is not configured" in err

