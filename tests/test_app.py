from pathlib import Path

from PIL import Image

from enemane_ai.analyzer import PRESET_PROMPT
from enemane_ai.app import (
    AnalyzedGraph,
    CalendarAnalysisRow,
    OutputRow,
    ResultRow,
    analyze_files,
    build_result_rows,
    export_calendar_analysis_csv,
    export_table_csv,
    export_target_format_csv,
    parse_calendar_analysis_response,
    parse_multi_item_response,
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

    text = csv_bytes.decode("cp932")
    lines = text.strip().splitlines()
    assert lines[0] == "画像内タイトル,項目名,AIで生成したコメント"
    assert "plot,グラフ画像,トレンド良好" in lines[1]


def test_parse_multi_item_response_handles_json_array() -> None:
    raw = """```json
[
  {"graph_name": "電力使用状況", "item_name": "最大電力[kW]", "comment": "コメント1"},
  {"graph_name": "電力使用状況", "item_name": "電力使用量[kWh]", "comment": "コメント2"}
]
```"""

    results = parse_multi_item_response(raw, fallback_graph_name="fallback")

    assert len(results) == 2
    assert results[0] == ("電力使用状況", "最大電力[kW]", "コメント1")
    assert results[1] == ("電力使用状況", "電力使用量[kWh]", "コメント2")


def test_parse_multi_item_response_handles_single_object() -> None:
    raw = '{"graph_name": "グラフA", "item_name": "項目X", "comment": "テストコメント"}'

    results = parse_multi_item_response(raw, fallback_graph_name="fallback")

    assert len(results) == 1
    assert results[0] == ("グラフA", "項目X", "テストコメント")


def test_parse_multi_item_response_falls_back_on_invalid_json() -> None:
    raw = "This is not JSON"

    results = parse_multi_item_response(raw, fallback_graph_name="fallback.png")

    assert len(results) == 1
    assert results[0][0] == "fallback.png"
    assert results[0][1] == ""
    assert "This is not JSON" in results[0][2]


def test_parse_multi_item_response_uses_fallback_for_missing_graph_name() -> None:
    raw = '[{"item_name": "項目A", "comment": "コメント"}]'

    results = parse_multi_item_response(raw, fallback_graph_name="default.png")

    assert len(results) == 1
    assert results[0][0] == "default.png"
    assert results[0][1] == "項目A"


def test_export_target_format_csv_has_bom_and_correct_headers() -> None:
    rows = [
        OutputRow(graph_name="グラフ1", item_name="最大電力[kW]", ai_comment="コメント1"),
        OutputRow(graph_name="グラフ1", item_name="電力使用量[kWh]", ai_comment="コメント2"),
    ]

    csv_bytes = export_target_format_csv(rows)

    # BOM付きUTF-8を確認
    assert csv_bytes.startswith(b"\xef\xbb\xbf")

    text = csv_bytes.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0] == "対応するグラフ名,項目名,生成するAIコメント"
    assert "グラフ1,最大電力[kW],コメント1" in lines[1]
    assert "グラフ1,電力使用量[kWh],コメント2" in lines[2]


def test_parse_calendar_analysis_response_handles_json_array() -> None:
    raw = """```json
[
  {"item": "全体傾向", "analysis": "10月は電力使用量が減少傾向"},
  {"item": "最大需要日の確認", "analysis": "22日(火)が最大需要を記録"}
]
```"""

    results = parse_calendar_analysis_response(raw)

    assert len(results) == 2
    assert results[0].item == "全体傾向"
    assert "10月" in results[0].analysis
    assert results[1].item == "最大需要日の確認"
    assert "22日" in results[1].analysis


def test_parse_calendar_analysis_response_handles_single_object() -> None:
    raw = '{"item": "全体傾向", "analysis": "テスト分析"}'

    results = parse_calendar_analysis_response(raw)

    assert len(results) == 1
    assert results[0].item == "全体傾向"
    assert results[0].analysis == "テスト分析"


def test_parse_calendar_analysis_response_falls_back_on_invalid_json() -> None:
    raw = "This is not JSON"

    results = parse_calendar_analysis_response(raw)

    assert len(results) == 1
    assert results[0].item == "分析結果"
    assert "This is not JSON" in results[0].analysis


def test_parse_calendar_analysis_response_strips_markdown() -> None:
    raw = """```json
[
  {"item": "**全体傾向**", "analysis": "電力使用量は*減少*傾向"}
]
```"""

    results = parse_calendar_analysis_response(raw)

    assert results[0].item == "全体傾向"
    assert "**" not in results[0].item
    assert "*" not in results[0].analysis


def test_export_calendar_analysis_csv_has_bom_and_correct_headers() -> None:
    rows = [
        CalendarAnalysisRow(item="全体傾向", analysis="10月は減少傾向"),
        CalendarAnalysisRow(item="最大需要日の確認", analysis="22日が最大"),
    ]

    csv_bytes = export_calendar_analysis_csv(rows)

    # BOM付きUTF-8を確認
    assert csv_bytes.startswith(b"\xef\xbb\xbf")

    text = csv_bytes.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0] == "項目,事実+仮説"
    assert "全体傾向,10月は減少傾向" in lines[1]
    assert "最大需要日の確認,22日が最大" in lines[2]
