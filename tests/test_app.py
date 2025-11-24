from pathlib import Path

from PIL import Image

from enemane_ai.analyzer import PRESET_PROMPT
from enemane_ai.app import (
    AnalyzedGraph,
    ResultRow,
    analyze_files,
    build_result_rows,
    export_table_csv,
    parse_structured_comment,
)


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


def test_analyze_files_extracts_titles_from_structured_llm(tmp_path: Path) -> None:
    img_path = tmp_path / "plot.png"
    Image.new("RGB", (10, 10), "white").save(img_path)

    class StructuredLLM(DummyLLM):
        def comment_on_graph(self, image, prompt):  # type: ignore[no-untyped-def]
            self.calls += 1
            return (
                '{"image_title": "レポートA",'
                ' "item_name": "月次売上グラフ",'
                ' "comment": "1) 増加傾向 2) 好調 3) 外れ値に注意"}'
            )

    llm = StructuredLLM()

    results = analyze_files([img_path], prompt="prompt", llm=llm)

    assert llm.calls == 1
    result = results[0]
    assert result.image_title == "レポートA"
    assert result.item_name == "月次売上グラフ"
    assert "増加傾向" in result.comment


def test_parse_structured_comment_handles_code_fence() -> None:
    raw = """```json
{"image_title": "トップ", "item_name": "PV推移", "comment": "1) ..."}
```"""

    image_title, item_name, comment = parse_structured_comment(raw, fallback_label="fallback")

    assert image_title == "トップ"
    assert item_name == "PV推移"
    assert "1)" in comment


def test_build_result_rows_sets_item_name_and_default_comment() -> None:
    analyzed = [
        AnalyzedGraph(label="chart.png", comment="いい感じ", image_data=b"raw"),
    ]

    rows = build_result_rows(analyzed)

    assert rows == [
        ResultRow(image_title="chart.png", item_name="グラフ画像", comment="いい感じ"),
    ]


def test_build_result_rows_ignores_csv_entries() -> None:
    analyzed = [
        AnalyzedGraph(label="chart.png", comment="comment", image_data=b"raw"),
        AnalyzedGraph(label="temperature.csv", comment="", text="csv rows"),
    ]

    rows = build_result_rows(analyzed)

    assert len(rows) == 1
    assert rows[0].image_title == "chart.png"


def test_build_result_rows_strips_markdown() -> None:
    analyzed = [
        AnalyzedGraph(
            label="**Bold Title**",
            image_title="**Bold Title**",
            item_name="*Italic Item*",
            comment="Check [Link](http://example.com) and `Code`\n- List",
            image_data=b"raw",
        ),
    ]

    rows = build_result_rows(analyzed)

    assert rows[0].image_title == "Bold Title"
    assert rows[0].item_name == "Italic Item"
    # Newlines are preserved by strip_markdown
    assert "Check Link and Code" in rows[0].comment
    assert "List" in rows[0].comment
    assert "**" not in rows[0].comment
    assert "[" not in rows[0].comment  # Links removed


def test_export_table_csv_outputs_headers_and_rows() -> None:
    rows = [
        ResultRow(image_title="plot", item_name="グラフ画像", comment="トレンド良好"),
        ResultRow(
            image_title="temp.csv",
            item_name="テキスト/CSV",
            comment="CSV/テキストはコメント生成を省略しています。",
        ),
    ]

    csv_bytes = export_table_csv(rows)

    text = csv_bytes.decode("utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "画像内タイトル,項目名,AIで生成したコメント"
    assert "plot,グラフ画像,トレンド良好" in lines[1]
