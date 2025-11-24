from pathlib import Path

from enemane_ai.analyzer import PRESET_PROMPT
from enemane_ai.app import analyze_files


class DummyLLM:
    def __init__(self) -> None:
        self.calls = 0

    def comment_on_graph(self, image, prompt):  # type: ignore[no-untyped-def]
        self.calls += 1
        return "graph"

    def comment_on_text(self, text, prompt):  # type: ignore[no-untyped-def]
        self.calls += 1
        return "text"


def test_analyze_files_skips_comment_generation_for_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "temperature.csv"
    csv_path.write_text("date,temp\n2024-01-01,8.5\n", encoding="utf-8")
    llm = DummyLLM()

    results = analyze_files([csv_path], prompt=PRESET_PROMPT, llm=llm)

    assert llm.calls == 0
    assert len(results) == 1
    result = results[0]
    assert result.image_data is None
    assert result.text is not None
    assert "2024-01-01,8.5" in result.text
    assert result.comment == ""
