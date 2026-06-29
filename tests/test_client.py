from citemd.generate.client import FakeLLMClient, LLMClient


def test_fake_client_records_calls_and_satisfies_protocol():
    client = FakeLLMClient("hello", model="m")
    assert isinstance(client, LLMClient)
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out == "hello"
    assert client.calls == [[{"role": "user", "content": "hi"}]]


def test_fake_client_callable_responder():
    client = FakeLLMClient(lambda msgs: msgs[-1]["content"].upper())
    assert client.complete([{"role": "user", "content": "abc"}]) == "ABC"
