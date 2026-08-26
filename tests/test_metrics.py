from backend.metrics import _parse_smi_line, parse_tokens_per_sec, ram_snapshot, stats_tokens_per_sec


def test_parse_smi_line():
    row = _parse_smi_line("0, NVIDIA GeForce RTX 5080, 2341, 16303, 12, 45")
    assert row is not None
    assert row["name"] == "NVIDIA GeForce RTX 5080"
    assert row["used_mb"] == 2341
    assert row["total_gb"] == round(16303 / 1024, 2)
    assert row["util_percent"] == 12
    assert row["temperature_c"] == 45


def test_parse_llama_cpp_tokens_per_sec():
    line = "eval time = 1234.56 ms / 100 tokens ( 12.35 ms per token, 81.05 tokens per second)"
    assert parse_tokens_per_sec(line) == 81.05


def test_parse_tok_s_shorthand():
    assert parse_tokens_per_sec("decode: 42.5 tok/s running") == 42.5


def test_stats_payload():
    assert stats_tokens_per_sec({"tokens_per_second": 33.1}) == 33.1
    assert stats_tokens_per_sec({"performance": {"decode_throughput": 12}}) == 12.0


def test_ram_snapshot_keys():
    ram = ram_snapshot()
    assert ram["total_gb"] > 0
    assert 0 <= ram["percent"] <= 100
