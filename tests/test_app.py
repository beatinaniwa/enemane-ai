from pathlib import Path

from PIL import Image

from enemane_ai.analyzer import PRESET_PROMPT
from enemane_ai.app import (
    AnalyzedGraph,
    ResultRow,
    analyze_files,
    build_image_prompt,
    build_result_rows,
    export_table_csv,
    parse_structured_comment,
)


class DummyLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt: str | None = None

    def comment_on_graph(self, image, prompt):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_prompt = prompt
        return "graph"

    def comment_on_text(self, text, prompt):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_prompt = prompt
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


def test_analyze_files_includes_comparison_context(tmp_path: Path) -> None:
    img_path = tmp_path / "plot.png"
    Image.new("RGB", (10, 10), "white").save(img_path)
    llm = DummyLLM()

    analyze_files(
        [img_path],
        prompt="base prompt",
        llm=llm,
        comparison_context="前年比較: 2023年=100, 2024年=120",
    )

    assert llm.calls == 1
    assert llm.last_prompt is not None
    assert "前年比較: 2023年=100, 2024年=120" in llm.last_prompt


def test_build_image_prompt_mentions_monthly_energy_rule() -> None:
    prompt = build_image_prompt("base")

    assert "電力使用量 [kWh] について <月間電力使用量>" in prompt
    assert "前年比較" in prompt
    assert "月報YYYYMM.csv" in prompt


def test_analyze_files_generates_two_items_for_power_usage_status_chart(tmp_path: Path) -> None:
    """「電力使用状況」のグラフでは最大電力と電力使用量の2つを生成."""
    img_path = tmp_path / "status.png"
    Image.new("RGB", (10, 10), "white").save(img_path)

    class UsageStatusLLM(DummyLLM):
        def comment_on_graph(self, image, prompt):  # type: ignore[no-untyped-def]
            self.calls += 1
            # 最初の呼び出しで電力使用状況を返す (ベースの判定用)
            # 2回目以降は最大電力/電力使用量のコメントを生成
            if self.calls == 1:
                return (
                    '{"image_title": "電力使用状況",'
                    ' "item_name": "概要グラフ", "comment": "正常に稼働中"}'
                )
            elif self.calls == 2:
                return (
                    '{"image_title": "電力使用状況",'
                    ' "item_name": "最大電力[kW]", "comment": "最大電力は安定推移"}'
                )
            else:
                return (
                    '{"image_title": "電力使用状況",'
                    ' "item_name": "電力使用量[kWh]", "comment": "使用量は前年並み"}'
                )

    llm = UsageStatusLLM()

    results = analyze_files([img_path], prompt="prompt", llm=llm)

    # 「電力使用状況」では最大電力と電力使用量の2エントリーが生成される
    assert llm.calls == 3  # ベース判定用 + 最大電力 + 電力使用量
    assert len(results) == 2
    max_power = results[0]
    energy = results[1]
    assert max_power.item_name == "最大電力[kW]"
    assert energy.item_name == "電力使用量[kWh]"
    assert max_power.image_title == "電力使用状況"
    assert energy.image_title == "電力使用状況"


def test_build_image_prompt_includes_forced_item_and_focus() -> None:
    prompt = build_image_prompt("base", forced_item_name="回路別内訳", focus="回路別内訳(構成比)")

    assert "回路別内訳" in prompt
    assert "構成比" in prompt


def test_analyze_files_enforces_item_name_for_monthly_energy_chart(tmp_path: Path) -> None:
    img_path = tmp_path / "energy.png"
    Image.new("RGB", (10, 10), "white").save(img_path)

    class MonthlyLLM(DummyLLM):
        def comment_on_graph(self, image, prompt):  # type: ignore[no-untyped-def]
            self.calls += 1
            # 1回目: ベース判定用、2回目: 前年比較専用、3回目: 回路別内訳専用
            if self.calls == 1:
                return (
                    '{"image_title": "電力使用量 [kWh] について <月間電力使用量>",'
                    ' "item_name": "もとの項目", "comment": "ベースコメント"}'
                )
            elif self.calls == 2:
                return (
                    '{"image_title": "電力使用量 [kWh] について <月間電力使用量>",'
                    ' "item_name": "前年比較", "comment": "前年比 10% 増加"}'
                )
            else:
                return (
                    '{"image_title": "電力使用量 [kWh] について <月間電力使用量>",'
                    ' "item_name": "回路別内訳", "comment": "回路別トップ3"}'
                )

    llm = MonthlyLLM()

    results = analyze_files([img_path], prompt="prompt", llm=llm)

    # ベース判定用 + 前年比較専用 + 回路別内訳専用 = 3回
    assert llm.calls == 3
    assert len(results) == 2
    yoy = results[0]
    assert yoy.item_name == "前年比較"
    assert "前年比" in yoy.comment


def test_analyze_files_adds_circuit_breakdown_entry(tmp_path: Path) -> None:
    img_path = tmp_path / "energy.png"
    Image.new("RGB", (10, 10), "white").save(img_path)

    class DualLLM(DummyLLM):
        def comment_on_graph(self, image, prompt):  # type: ignore[no-untyped-def]
            self.calls += 1
            # 1回目: ベース判定用、2回目: 前年比較専用、3回目: 回路別内訳専用
            if self.calls == 1:
                return (
                    '{"image_title": "電力使用量 [kWh] について <月間電力使用量>",'
                    ' "item_name": "もとの項目", "comment": "ベースコメント"}'
                )
            elif self.calls == 2:
                return (
                    '{"image_title": "電力使用量 [kWh] について <月間電力使用量>",'
                    ' "item_name": "前年比較", "comment": "前年比 10% 増加"}'
                )
            else:
                return (
                    '{"image_title": "電力使用量 [kWh] について <月間電力使用量>",'
                    ' "item_name": "回路別内訳", "comment": "回路別トップ3"}'
                )

    llm = DualLLM()

    results = analyze_files([img_path], prompt="prompt", llm=llm)

    # ベース判定用 + 前年比較専用 + 回路別内訳専用 = 3回
    assert llm.calls == 3
    assert len(results) == 2
    yoy, breakdown = results
    assert yoy.item_name == "前年比較"
    assert "前年比" in yoy.comment
    assert breakdown.item_name == "回路別内訳"
    assert "回路別" in breakdown.comment


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


def test_parse_structured_comment_strips_inline_json_noise() -> None:
    raw = (
        "LLM output with preface\n"
        '{"image_title": "タイトル", "item_name": "項目名", "comment": "1) コメント"}'
    )

    image_title, item_name, comment = parse_structured_comment(raw, fallback_label="fallback")

    assert image_title == "タイトル"
    assert item_name == "項目名"
    assert comment == "1) コメント"


def test_parse_structured_comment_drops_json_when_comment_missing() -> None:
    raw = (
        "Here is the structure\n"
        '{"image_title": "タイトルだけ", "item_name": "項目だけ"}\n'
        "文章としてはここを表示"
    )

    image_title, item_name, comment = parse_structured_comment(raw, fallback_label="fallback")

    assert image_title == "タイトルだけ"
    assert item_name == "項目だけ"
    assert "{" not in comment
    assert "文章としてはここを表示" in comment


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

    # Shift_JIS (CP932) でエンコードされている
    text = csv_bytes.decode("cp932")
    lines = text.strip().splitlines()
    assert lines[0] == "画像内タイトル,項目名,AIで生成したコメント"
    assert "plot,グラフ画像,トレンド良好" in lines[1]
