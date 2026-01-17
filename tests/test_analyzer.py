from pathlib import Path

import pytest
from google.genai import types as genai_types
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pytest import MonkeyPatch

from enemane_ai import analyzer
from enemane_ai.analyzer import (
    PRESET_PROMPT,
    DailyPowerSummary,
    MonthlyPowerCalendarData,
    MonthlyReportData,
    MonthlyTemperatureSummary,
    PeakDayCircuitData,
    PeakDayPowerData,
    analyze_image,
    analyze_text,
    build_peak_day_comparison_context,
    build_power_calendar_context,
    build_power_calendar_extended_context,
    build_supplementary_context,
    collect_graph_entries,
    judge_article_relevance,
    parse_monthly_report_csv,
    parse_peak_day_power_csv,
    parse_power_30min_csv,
    parse_temperature_csv_for_comparison,
)


class DummyLLM:
    def __init__(self, response: str = "analysis"):
        self.response = response
        self.calls = 0
        self.last_prompt: str | None = None
        self.last_text: str | None = None

    def comment_on_graph(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        return self.response

    def comment_on_text(self, text: str, prompt: str) -> str:
        self.calls += 1
        self.last_prompt = prompt
        self.last_text = text
        return self.response


def test_analyze_image_mentions_brightness_levels() -> None:
    llm = DummyLLM("LLM response")
    image = Image.new("RGB", (24, 24), "white")

    comment = analyze_image(image, prompt="custom prompt", llm=llm)

    assert comment == "LLM response"
    assert llm.calls == 1
    assert llm.last_prompt == "custom prompt"


def test_analyze_text_uses_llm() -> None:
    llm = DummyLLM("text response")

    comment = analyze_text("2024-01-01,8.5", prompt="csv prompt", llm=llm)

    assert comment == "text response"
    assert llm.calls == 1
    assert llm.last_prompt == "csv prompt"
    assert llm.last_text == "2024-01-01,8.5"


def test_analyze_text_does_not_leak_prompt_on_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = analyze_text("text content", prompt="top-secret prompt")

    assert "top-secret prompt" not in result
    assert "失敗しました" in result


def test_collect_graph_entries_from_png_and_pdf(tmp_path: Path) -> None:
    png_path = tmp_path / "plot.png"
    base = Image.new("RGB", (80, 60), "white")
    draw = ImageDraw.Draw(base)
    draw.line((10, 50, 70, 10), fill="black", width=3)
    base.save(png_path)

    pdf_path = tmp_path / "doc.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    with pdf_path.open("wb") as fp:
        writer.write(fp)

    entries = collect_graph_entries([png_path, pdf_path])

    labels = {entry.display_label for entry in entries}
    assert len(entries) == 2
    assert any("plot.png" in label for label in labels)
    assert any("doc.pdf#1" in label for label in labels)
    assert all(entry.image is not None and entry.image.size[0] > 0 for entry in entries)


def test_collect_graph_entries_from_temperature_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "temperature.csv"
    csv_path.write_text(
        "date,temp_c\n" "2024-01-01,8.5\n" "2024-01-02,12.3\n" "2024-01-03,5.0\n",
        encoding="utf-8",
    )

    entries = collect_graph_entries([csv_path])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.display_label == "temperature.csv"
    assert entry.image is None
    assert entry.text is not None
    assert "2024-01-02,12.3" in entry.text


def test_collect_graph_entries_from_shift_jis_temperature_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "maebashi_shift_jis.csv"
    csv_path.write_bytes(
        (
            "ダウンロードした時刻：2025/10/08 12:19:10\r\n"  # noqa: RUF001
            "\r\n"
            ",前橋,前橋\r\n"
            "年月日時,気温(℃),品質情報\r\n"
            "2023/10/1 1:00:00,22.9,8\r\n"
            "2023/10/1 2:00:00,22.3,8\r\n"
        ).encode("cp932")
    )

    entries = collect_graph_entries([csv_path])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.display_label == "maebashi_shift_jis.csv"
    assert entry.image is None
    assert entry.text is not None
    assert "2023/10/1 1:00:00,22.9" in entry.text
    assert "2023/10/1 2:00:00,22.3" in entry.text


def test_gemini_model_uses_env_key(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    class CallLog:
        def __init__(self) -> None:
            self.api_key: str | None = None
            self.model: str | None = None
            self.contents: list[genai_types.Part | str] | None = None

    calls = CallLog()

    class FakeModelAPI:
        @staticmethod
        def generate_content(model: str, contents: list[genai_types.Part | str]) -> object:
            calls.model = model
            calls.contents = contents

            class Response:
                text = "Gemini says hello"

            return Response()

    class FakeClient:
        def __init__(self, api_key: str):
            calls.api_key = api_key

        models = FakeModelAPI()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(analyzer, "genai", type("GenaiWrapper", (), {"Client": FakeClient}))

    model = analyzer.GeminiGraphLanguageModel.from_env()

    img_path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), "black").save(img_path)
    with Image.open(img_path) as img:
        comment = model.comment_on_graph(img, PRESET_PROMPT)

    assert comment == "Gemini says hello"
    assert calls.api_key == "fake-key"
    assert calls.model == analyzer.DEFAULT_MODEL_NAME
    assert calls.contents is not None
    assert len(calls.contents) == 2
    assert calls.contents[0] == PRESET_PROMPT
    image_part = calls.contents[1]
    assert isinstance(image_part, genai_types.Part)
    inline_data = image_part.inline_data
    assert inline_data is not None
    assert inline_data.mime_type == "image/png"

    text_comment = model.comment_on_text("raw csv content", PRESET_PROMPT)
    assert text_comment == "Gemini says hello"
    assert calls.contents == [PRESET_PROMPT, "raw csv content"]


def test_parse_monthly_report_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "月報202310.csv"
    csv_path.write_text(
        "(1) 参照元「月間エネルギー使用実績と最大電力の推移」,,,\n"
        ",10/1(日),10/2(月),10/3(火)\n"
        "最大電力[kW],27.52,35.14,34.18\n"
        "1F事務所SR_電灯,100.47,97,99.53\n"
        "受電電力,330.95,465.97,458.73\n",
        encoding="utf-8",
    )

    report = parse_monthly_report_csv(csv_path)

    assert report.month_label == "2023年10月"
    assert report.max_power_daily == [27.52, 35.14, 34.18]
    assert report.max_power_monthly == 35.14
    assert report.total_power_daily == [330.95, 465.97, 458.73]
    assert report.total_power_monthly == 1255.65
    assert "1F事務所SR_電灯" in report.circuits
    assert report.circuits["1F事務所SR_電灯"] == [100.47, 97, 99.53]


def test_parse_temperature_csv_for_comparison(tmp_path: Path) -> None:
    csv_path = tmp_path / "temp.csv"
    csv_path.write_text(
        "年月日時,気温(℃),品質\n"
        "2023/10/1 1:00:00,22.9,8\n"
        "2023/10/1 2:00:00,20.0,8\n"
        "2024/10/1 1:00:00,25.0,8\n"
        "2024/10/1 2:00:00,23.0,8\n",
        encoding="utf-8",
    )

    prev, curr = parse_temperature_csv_for_comparison(csv_path)

    assert prev.year_month == "2023-10"
    assert prev.max_temp == 22.9
    assert prev.min_temp == 20.0
    assert prev.avg_temp == 21.45

    assert curr.year_month == "2024-10"
    assert curr.max_temp == 25.0
    assert curr.min_temp == 23.0
    assert curr.avg_temp == 24.0


def test_parse_temperature_csv_for_comparison_with_next_month_data(
    tmp_path: Path,
) -> None:
    """月末翌日のデータ(1件)が含まれる場合でも同じ月を正しく比較する."""
    csv_path = tmp_path / "temp.csv"
    csv_path.write_text(
        "年月日時,気温(℃),品質\n"
        "2023/10/1 1:00:00,22.9,8\n"
        "2023/10/1 2:00:00,20.0,8\n"
        "2023/11/1 0:00:00,14.2,8\n"  # 月末翌日データ(1件)
        "2024/10/1 1:00:00,25.0,8\n"
        "2024/10/1 2:00:00,23.0,8\n"
        "2024/11/1 0:00:00,13.1,8\n",  # 月末翌日データ(1件)
        encoding="utf-8",
    )

    prev, curr = parse_temperature_csv_for_comparison(csv_path)

    # 同じ月(10月)を比較すること
    assert prev.year_month == "2023-10"
    assert curr.year_month == "2024-10"
    # 11月の1件データは無視される
    assert prev.max_temp == 22.9
    assert curr.max_temp == 25.0


def test_parse_temperature_csv_for_comparison_no_matching_month(
    tmp_path: Path,
) -> None:
    """同じ月の前年・当年データがない場合はエラー."""
    csv_path = tmp_path / "temp.csv"
    csv_path.write_text(
        "年月日時,気温(℃),品質\n"
        "2023/10/1 1:00:00,22.9,8\n"
        "2024/11/1 1:00:00,25.0,8\n",  # 異なる月
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="同じ月の前年・当年データが見つかりません"):
        parse_temperature_csv_for_comparison(csv_path)


def test_build_supplementary_context_with_data() -> None:
    report = MonthlyReportData(
        month_label="2023年10月",
        max_power_daily=[30.0, 35.0, 32.0],
        circuits={"1F電灯": [100.0, 110.0, 105.0], "2F電灯": [50.0, 55.0, 52.0]},
        total_power_daily=[300.0, 350.0, 320.0],
    )
    prev = MonthlyTemperatureSummary(
        year_month="2023-10", max_temp=27.0, min_temp=10.0, avg_temp=18.0
    )
    curr = MonthlyTemperatureSummary(
        year_month="2024-10", max_temp=30.0, min_temp=12.0, avg_temp=20.0
    )

    context = build_supplementary_context(report, (prev, curr))

    assert "2023年10月" in context
    assert "35.0 kW" in context
    assert "970 kWh" in context
    assert "1F電灯" in context
    assert "2024年10月" in context
    assert "+3.0℃" in context
    assert "+2.0℃" in context


def test_build_supplementary_context_with_none() -> None:
    context = build_supplementary_context(None, None)
    assert context == ""


def test_parse_power_30min_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "PDU_30min_202410.csv"
    csv_path.write_bytes(
        (
            "計測対象,コントローラ1,,,,,,,\n"
            ",機器31-1,,,,,,,,\n"
            ",受電電力量,,,,,,,,\n"
            ",kWh,,,,,,,,\n"
            "2024-10-01 00:00,4.29,,,,,,,\n"
            "2024-10-01 00:30,4.04,,,,,,,\n"
            "2024-10-01 01:00,4.08,,,,,,,\n"
            "2024-10-05 00:00,3.50,,,,,,,\n"  # 土曜日
            "2024-10-05 00:30,3.60,,,,,,,\n"
            "2024-10-06 00:00,3.20,,,,,,,\n"  # 日曜日
            "2024-10-06 00:30,3.30,,,,,,,\n"
        ).encode("cp932")
    )

    data = parse_power_30min_csv(csv_path)

    assert data.year_month == "2024年10月"
    assert len(data.daily_summaries) == 3
    # 10/1のデータ確認
    day1 = data.daily_summaries[0]
    assert day1.date == "2024-10-01"
    assert day1.day_of_week == "火"
    assert day1.total_kwh == 4.29 + 4.04 + 4.08
    assert day1.max_kwh == 4.29
    assert day1.max_time == "00:00"
    # 最大電力使用量日 (1日の合計が最大 = 10/1)
    assert "1日" in data.max_usage_day
    assert data.max_usage_kwh == 4.29 + 4.04 + 4.08
    # 最大需要電力日 (30分ピークが最大 = 10/1の4.29kW)
    assert "1日" in data.max_demand_day
    assert data.max_demand_kw == 4.29
    # 平日/休日平均
    assert data.weekday_avg_kwh > 0
    assert data.weekend_avg_kwh > 0


def test_parse_power_30min_csv_no_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("header1,header2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="有効な30分電力データがありません"):
        parse_power_30min_csv(csv_path)


def test_build_power_calendar_context() -> None:
    data = MonthlyPowerCalendarData(
        year_month="2024年10月",
        daily_summaries=[
            DailyPowerSummary(
                date="2024-10-01",
                day_of_week="火",
                total_kwh=300.0,
                max_kwh=21.5,
                max_time="14:00",
            ),
            DailyPowerSummary(
                date="2024-10-05",
                day_of_week="土",
                total_kwh=100.0,
                max_kwh=8.0,
                max_time="10:00",
            ),
        ],
        total_monthly_kwh=8500.0,
        max_usage_day="1日(火)",
        max_usage_kwh=300.0,
        max_demand_day="1日(火)",
        max_demand_kw=21.5,
        weekday_avg_kwh=290.0,
        weekend_avg_kwh=95.0,
    )

    context = build_power_calendar_context(data)

    assert "2024年10月" in context
    assert "8,500.0 kWh" in context
    assert "最大電力使用量日: 1日(火)" in context
    assert "300.0 kWh" in context
    assert "最大需要電力日: 1日(火)" in context
    assert "21.5 kW" in context
    assert "290.0 kWh/日" in context
    assert "95.0 kWh/日" in context
    assert "上位5日" in context


def test_build_power_calendar_extended_context_current_only() -> None:
    """当年データのみのコンテキスト構築."""
    curr_power = MonthlyPowerCalendarData(
        year_month="2024年10月",
        daily_summaries=[
            DailyPowerSummary(
                date="2024-10-01",
                day_of_week="火",
                total_kwh=300.0,
                max_kwh=21.5,
                max_time="14:00",
            ),
        ],
        total_monthly_kwh=9000.0,
        max_usage_day="1日(火)",
        max_usage_kwh=300.0,
        max_demand_day="1日(火)",
        max_demand_kw=21.5,
        weekday_avg_kwh=300.0,
        weekend_avg_kwh=100.0,
    )

    context = build_power_calendar_extended_context(curr_power=curr_power)

    assert "当年30分間隔電力データ" in context
    assert "2024年10月" in context
    assert "9,000.0 kWh" in context
    assert "前年30分間隔電力データ" not in context
    assert "前年比較" not in context
    assert "気温データ" not in context


def test_build_power_calendar_extended_context_with_prev_power() -> None:
    """前年電力データありのコンテキスト構築."""
    curr_power = MonthlyPowerCalendarData(
        year_month="2024年10月",
        daily_summaries=[],
        total_monthly_kwh=9000.0,
        max_usage_day="1日(火)",
        max_usage_kwh=300.0,
        max_demand_day="1日(火)",
        max_demand_kw=21.5,
        weekday_avg_kwh=300.0,
        weekend_avg_kwh=100.0,
    )
    prev_power = MonthlyPowerCalendarData(
        year_month="2023年10月",
        daily_summaries=[],
        total_monthly_kwh=8500.0,
        max_usage_day="1日(火)",
        max_usage_kwh=280.0,
        max_demand_day="1日(火)",
        max_demand_kw=20.0,
        weekday_avg_kwh=280.0,
        weekend_avg_kwh=95.0,
    )

    context = build_power_calendar_extended_context(
        curr_power=curr_power,
        prev_power=prev_power,
    )

    # 当年データの確認
    assert "当年30分間隔電力データ" in context
    assert "2024年10月" in context
    assert "9,000.0 kWh" in context

    # 前年データの確認
    assert "前年30分間隔電力データ" in context
    assert "2023年10月" in context
    assert "8,500.0 kWh" in context

    # 前年比較の確認
    assert "前年比較" in context
    assert "+500.0 kWh" in context
    assert "+5.9%" in context


def test_build_power_calendar_extended_context_with_temperature() -> None:
    """気温データありのコンテキスト構築."""
    curr_power = MonthlyPowerCalendarData(
        year_month="2024年10月",
        daily_summaries=[],
        total_monthly_kwh=9000.0,
        max_usage_day="1日(火)",
        max_usage_kwh=300.0,
        max_demand_day="1日(火)",
        max_demand_kw=21.5,
        weekday_avg_kwh=300.0,
        weekend_avg_kwh=100.0,
    )
    prev_temp = MonthlyTemperatureSummary(
        year_month="2023-10",
        max_temp=28.0,
        min_temp=13.0,
        avg_temp=20.0,
    )
    curr_temp = MonthlyTemperatureSummary(
        year_month="2024-10",
        max_temp=30.0,
        min_temp=15.0,
        avg_temp=22.0,
    )

    context = build_power_calendar_extended_context(
        curr_power=curr_power,
        temperature=(prev_temp, curr_temp),
    )

    # 気温データの確認
    assert "気温データ" in context
    assert "2023年10月" in context
    assert "2024年10月" in context
    assert "最高28.0" in context
    assert "最高30.0" in context
    assert "+2.0℃" in context  # 平均気温差


def test_build_power_calendar_extended_context_full() -> None:
    """全データを含むコンテキスト構築."""
    curr_power = MonthlyPowerCalendarData(
        year_month="2024年10月",
        daily_summaries=[
            DailyPowerSummary(
                date="2024-10-01",
                day_of_week="火",
                total_kwh=300.0,
                max_kwh=21.5,
                max_time="14:00",
            ),
        ],
        total_monthly_kwh=9000.0,
        max_usage_day="1日(火)",
        max_usage_kwh=300.0,
        max_demand_day="1日(火)",
        max_demand_kw=21.5,
        weekday_avg_kwh=300.0,
        weekend_avg_kwh=100.0,
    )
    prev_power = MonthlyPowerCalendarData(
        year_month="2023年10月",
        daily_summaries=[],
        total_monthly_kwh=8500.0,
        max_usage_day="1日(火)",
        max_usage_kwh=280.0,
        max_demand_day="1日(火)",
        max_demand_kw=20.0,
        weekday_avg_kwh=280.0,
        weekend_avg_kwh=95.0,
    )
    prev_temp = MonthlyTemperatureSummary(
        year_month="2023-10",
        max_temp=28.0,
        min_temp=13.0,
        avg_temp=20.0,
    )
    curr_temp = MonthlyTemperatureSummary(
        year_month="2024-10",
        max_temp=30.0,
        min_temp=15.0,
        avg_temp=22.0,
    )

    context = build_power_calendar_extended_context(
        curr_power=curr_power,
        prev_power=prev_power,
        temperature=(prev_temp, curr_temp),
    )

    # 当年電力データ
    assert "当年30分間隔電力データ" in context
    assert "9,000.0 kWh" in context

    # 前年電力データ
    assert "前年30分間隔電力データ" in context
    assert "8,500.0 kWh" in context

    # 前年比較
    assert "前年比較" in context
    assert "+500.0 kWh" in context

    # 気温データ
    assert "気温データ" in context
    assert "+2.0℃" in context

    # 上位5日
    assert "当年電力使用量 上位5日" in context


# =============================================================================
# 記事適切性判定機能のテスト
# =============================================================================


def test_judge_article_relevance_returns_true_for_relevant() -> None:
    """関連記事は is_relevant=True を返す."""
    llm = DummyLLM('{"is_relevant": true, "reason": "オフィスの省エネに関連"}')
    result = judge_article_relevance(
        content="オフィスビルの空調省エネ対策について詳しく解説します。",
        title="オフィス空調の省エネ",
        url="https://example.com/article",
        building_type="オフィス",
        llm=llm,
    )
    assert result.is_relevant is True
    assert "省エネ" in result.reason
    assert result.url == "https://example.com/article"
    assert result.title == "オフィス空調の省エネ"


def test_judge_article_relevance_returns_false_for_irrelevant() -> None:
    """無関係記事は is_relevant=False を返す."""
    llm = DummyLLM('{"is_relevant": false, "reason": "建物タイプと無関係"}')
    result = judge_article_relevance(
        content="自動車のカスタマイズについて...",
        title="車のチューニング",
        url="https://example.com/cars",
        building_type="介護福祉施設",
        llm=llm,
    )
    assert result.is_relevant is False
    assert "無関係" in result.reason


def test_judge_article_relevance_handles_json_error() -> None:
    """JSONパースエラー時はフォールスルー (True)."""
    llm = DummyLLM("invalid json response")
    result = judge_article_relevance(
        content="何らかの記事内容",
        title="Test",
        url="https://example.com",
        building_type="工場",
        llm=llm,
    )
    # パース失敗時は適切と判定してフォールスルー
    assert result.is_relevant is True
    assert "判定不能" in result.reason


def test_judge_article_relevance_handles_code_fence() -> None:
    """コードフェンス付きJSON応答を正しくパースする."""
    llm = DummyLLM('```json\n{"is_relevant": true, "reason": "適切な記事"}\n```')
    result = judge_article_relevance(
        content="省エネに関する記事",
        title="省エネ記事",
        url="https://example.com/energy",
        building_type="オフィス",
        llm=llm,
    )
    assert result.is_relevant is True
    assert result.reason == "適切な記事"


def test_judge_article_relevance_truncates_long_content() -> None:
    """長いコンテンツは3000文字に切り詰められる."""
    long_content = "あ" * 5000
    llm = DummyLLM('{"is_relevant": true, "reason": "OK"}')
    result = judge_article_relevance(
        content=long_content,
        title="Test",
        url="https://example.com",
        building_type="オフィス",
        llm=llm,
    )
    # プロンプトに埋め込まれたコンテンツが3000文字に切り詰められていることを確認
    # (テキストは空文字列で渡される - 二重送信防止のため)
    assert llm.last_text == ""
    assert llm.last_prompt is not None
    # プロンプト内に切り詰められたコンテンツ (3000文字の"あ") が含まれている
    truncated = "あ" * 3000
    assert truncated in llm.last_prompt
    # 5000文字のフルコンテンツは含まれていない
    assert long_content not in llm.last_prompt
    assert result.is_relevant is True


# =============================================================================
# グラフ用データ(4) (最大デマンド発生日データ) のテスト
# =============================================================================


def test_parse_peak_day_power_csv(tmp_path: Path) -> None:
    """グラフ用データ(4) CSVをパースできること."""
    csv_path = tmp_path / "月報202410.xlsx - グラフ用データ(4).csv"
    csv_path.write_text(
        "(4) 詳細データ(単位:kWh)\n"
        "2024/10/02,0:00,0:30,1:00,1:30\n"
        "受電電力,4.18,4.18,4.03,3.95\n"
        "1F事務所SR_電灯,1.20,1.15,1.10,1.05\n",
        encoding="utf-8",
    )

    data = parse_peak_day_power_csv(csv_path)

    assert data.peak_date == "2024/10/02"
    assert len(data.time_slots) == 4
    assert data.time_slots[0] == "0:00"
    assert data.time_slots[1] == "0:30"
    assert len(data.circuits) == 2
    assert data.circuits[0].circuit_name == "受電電力"
    assert data.circuits[0].values[0] == 4.18
    assert data.circuits[1].circuit_name == "1F事務所SR_電灯"


def test_parse_peak_day_power_csv_shift_jis(tmp_path: Path) -> None:
    """Shift_JISエンコーディングのCSVをパースできること."""
    csv_path = tmp_path / "test.csv"
    csv_path.write_bytes(
        ("(4) 詳細データ\n" "2024/10/02,0:00,0:30\n" "受電電力,4.18,4.20\n").encode("cp932")
    )

    data = parse_peak_day_power_csv(csv_path)

    assert data.peak_date == "2024/10/02"
    assert data.circuits[0].circuit_name == "受電電力"
    assert data.circuits[0].values[0] == 4.18


def test_parse_peak_day_power_csv_invalid_format(tmp_path: Path) -> None:
    """フォーマットが不正な場合はエラー."""
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("header only\n", encoding="utf-8")

    with pytest.raises(ValueError, match="フォーマットが不正です"):
        parse_peak_day_power_csv(csv_path)


def test_parse_peak_day_power_csv_invalid_date(tmp_path: Path) -> None:
    """日付フォーマットが不正な場合はエラー."""
    csv_path = tmp_path / "invalid_date.csv"
    csv_path.write_text(
        "(4) 詳細データ\n" "invalid_date,0:00,0:30\n" "受電電力,4.18,4.20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="日付フォーマットが不正です"):
        parse_peak_day_power_csv(csv_path)


def test_peak_day_power_data_properties() -> None:
    """PeakDayPowerDataのプロパティが正しく動作すること."""
    data = PeakDayPowerData(
        peak_date="2024/10/02",
        time_slots=["0:00", "0:30", "1:00", "1:30"],
        circuits=[
            PeakDayCircuitData(
                circuit_name="受電電力",
                values=[4.0, 5.0, 4.5, 4.2],  # max at index 1 (0:30)
            ),
            PeakDayCircuitData(
                circuit_name="1F電灯",
                values=[1.0, 1.5, 1.2, 1.1],
            ),
        ],
    )

    # ピーク時刻のテスト (受電電力の最大値のインデックス)
    assert data.peak_time == "0:30"
    # ピーク電力のテスト (kWh/30min -> kW: *2)
    assert data.peak_power_kw == 10.0  # 5.0 * 2

    # ピーク時刻の回路別電力値
    peak_circuits = data.get_circuit_power_at_peak()
    assert peak_circuits["受電電力"] == 5.0
    assert peak_circuits["1F電灯"] == 1.5


def test_peak_day_power_data_empty_circuits() -> None:
    """回路データが空の場合のプロパティ動作."""
    data = PeakDayPowerData(
        peak_date="2024/10/02",
        time_slots=["0:00", "0:30"],
        circuits=[],
    )

    assert data.peak_time == ""
    assert data.peak_power_kw == 0.0
    assert data.get_circuit_power_at_peak() == {}


def test_build_peak_day_comparison_context_current_only() -> None:
    """当年データのみのコンテキスト構築."""
    curr = PeakDayPowerData(
        peak_date="2024/10/02",
        time_slots=["14:00", "14:30"],
        circuits=[
            PeakDayCircuitData("受電電力", [40.0, 42.0]),
            PeakDayCircuitData("1F電灯", [10.0, 11.0]),
        ],
    )

    context = build_peak_day_comparison_context(curr)

    assert "当年最大デマンド発生日データ" in context
    assert "2024/10/02" in context
    assert "14:30" in context  # peak time
    assert "84.0 kW" in context  # 42.0 * 2
    assert "前年" not in context


def test_build_peak_day_comparison_context_with_prev() -> None:
    """前年データありのコンテキスト構築."""
    curr = PeakDayPowerData(
        peak_date="2024/10/02",
        time_slots=["14:00", "14:30"],
        circuits=[
            PeakDayCircuitData("受電電力", [40.0, 42.0]),
            PeakDayCircuitData("1F電灯", [10.0, 12.0]),
        ],
    )
    prev = PeakDayPowerData(
        peak_date="2023/10/19",
        time_slots=["14:00", "14:30"],
        circuits=[
            PeakDayCircuitData("受電電力", [38.0, 40.0]),
            PeakDayCircuitData("1F電灯", [9.0, 10.0]),
        ],
    )

    context = build_peak_day_comparison_context(curr, prev)

    assert "当年最大デマンド発生日データ" in context
    assert "前年最大デマンド発生日データ" in context
    assert "前年比較" in context
    assert "+4.0 kW" in context  # 84 - 80
    assert "2023/10/19" in context
    assert "2024/10/02" in context


def test_build_peak_day_comparison_context_none() -> None:
    """データなしの場合は空文字を返す."""
    context = build_peak_day_comparison_context(None)
    assert context == ""


def test_build_peak_day_comparison_context_time_change() -> None:
    """ピーク発生時刻が変化した場合の表示."""
    curr = PeakDayPowerData(
        peak_date="2024/10/02",
        time_slots=["14:00", "14:30", "15:00"],
        circuits=[
            PeakDayCircuitData("受電電力", [40.0, 42.0, 45.0]),  # peak at 15:00
        ],
    )
    prev = PeakDayPowerData(
        peak_date="2023/10/19",
        time_slots=["14:00", "14:30", "15:00"],
        circuits=[
            PeakDayCircuitData("受電電力", [40.0, 43.0, 42.0]),  # peak at 14:30
        ],
    )

    context = build_peak_day_comparison_context(curr, prev)

    # 発生時刻の変化を確認
    assert "14:30 → 15:00" in context


def test_build_peak_day_comparison_context_same_time() -> None:
    """ピーク発生時刻が同一の場合の表示."""
    curr = PeakDayPowerData(
        peak_date="2024/10/02",
        time_slots=["14:00", "14:30"],
        circuits=[
            PeakDayCircuitData("受電電力", [40.0, 42.0]),  # peak at 14:30
        ],
    )
    prev = PeakDayPowerData(
        peak_date="2023/10/19",
        time_slots=["14:00", "14:30"],
        circuits=[
            PeakDayCircuitData("受電電力", [38.0, 40.0]),  # peak at 14:30
        ],
    )

    context = build_peak_day_comparison_context(curr, prev)

    assert "同一 (14:30)" in context
