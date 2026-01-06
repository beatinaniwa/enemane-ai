"""Google SpreadSheet を使用した記事キャッシュ機能。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import gspread


@dataclass
class ArticleCacheRow:
    """キャッシュされた記事データ。"""

    theme: str
    title: str
    content: str
    image: str
    link: str


@dataclass
class CacheConfig:
    """キャッシュ設定。"""

    spreadsheet_id: str
    sheet_name: str = "article_cache"


CACHE_HEADER_ROW = [
    "cache_key",
    "theme",
    "building_types",
    "title",
    "content",
    "image",
    "link",
    "created_at",
]


def generate_cache_key(theme: str, building_types: list[str]) -> str:
    """テーマと建物タイプからキャッシュキーを生成。

    建物タイプはソートしてから結合することで、順序に依存しないキーを生成。

    Args:
        theme: テーマ名
        building_types: 建物タイプのリスト

    Returns:
        キャッシュキー文字列
    """
    sorted_types = sorted(building_types)
    types_str = "_".join(sorted_types)
    return f"{theme}_{types_str}"


def get_gspread_client() -> "gspread.Client | None":
    """Streamlit secretsからサービスアカウント認証情報を取得し、gspreadクライアントを生成。

    Returns:
        gspreadクライアント、認証失敗時はNone
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds_dict = st.secrets.get("gcp_service_account")
        if not creds_dict:
            return None

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
        return gspread.authorize(credentials)
    except Exception:
        return None


def ensure_header_row(sheet: "gspread.Worksheet") -> None:
    """シートにヘッダー行が存在しない場合は追加する。

    Args:
        sheet: gspreadワークシート
    """
    # 1行目を取得
    first_row = sheet.row_values(1)

    # ヘッダーが存在しない (空または異なる) 場合は追加
    if not first_row or first_row[0] != CACHE_HEADER_ROW[0]:
        sheet.insert_row(CACHE_HEADER_ROW, 1)


def get_cached_articles(
    theme: str,
    building_types: list[str],
    client: "gspread.Client",
    config: CacheConfig,
) -> list[ArticleCacheRow] | None:
    """キャッシュから記事を取得。

    Args:
        theme: テーマ名
        building_types: 建物タイプのリスト
        client: gspreadクライアント
        config: キャッシュ設定

    Returns:
        キャッシュされた記事のリスト、キャッシュミス時はNone
    """
    cache_key = generate_cache_key(theme, building_types)

    try:
        sheet = client.open_by_key(config.spreadsheet_id).worksheet(config.sheet_name)
        all_records = sheet.get_all_records()

        cached_rows = [row for row in all_records if row.get("cache_key") == cache_key]

        if not cached_rows:
            return None

        return [
            ArticleCacheRow(
                theme=str(row.get("theme", "")),
                title=str(row.get("title", "")),
                content=str(row.get("content", "")),
                image=str(row.get("image", "")),
                link=str(row.get("link", "")),
            )
            for row in cached_rows
        ]
    except Exception:
        return None


def save_articles_to_cache(
    theme: str,
    building_types: list[str],
    articles: list[ArticleCacheRow],
    client: "gspread.Client",
    config: CacheConfig,
) -> bool:
    """記事をキャッシュに保存。既存の同一キャッシュキーのデータは削除してから保存。

    Args:
        theme: テーマ名
        building_types: 建物タイプのリスト
        articles: 保存する記事のリスト
        client: gspreadクライアント
        config: キャッシュ設定

    Returns:
        保存成功時True、失敗時False
    """
    cache_key = generate_cache_key(theme, building_types)
    now = datetime.now(timezone.utc)
    building_types_str = ",".join(sorted(building_types))

    try:
        sheet = client.open_by_key(config.spreadsheet_id).worksheet(config.sheet_name)

        # ヘッダー行を確保
        ensure_header_row(sheet)

        # 既存データを削除
        delete_cached_articles(cache_key, sheet)

        # 新規データを追加
        rows_to_add = [
            [
                cache_key,
                theme,
                building_types_str,
                article.title,
                article.content,
                article.image,
                article.link,
                now.isoformat(),
            ]
            for article in articles
        ]

        if rows_to_add:
            from gspread.utils import ValueInputOption

            sheet.append_rows(rows_to_add, value_input_option=ValueInputOption.user_entered)

        return True
    except Exception:
        return False


def delete_cached_articles(cache_key: str, sheet: "gspread.Worksheet") -> int:
    """指定キャッシュキーのデータを削除。

    Args:
        cache_key: 削除対象のキャッシュキー
        sheet: gspreadワークシート

    Returns:
        削除した行数
    """
    try:
        all_records = sheet.get_all_records()

        # 削除対象の行番号を特定 (ヘッダー行を考慮して+2)
        rows_to_delete = [
            i + 2 for i, row in enumerate(all_records) if row.get("cache_key") == cache_key
        ]

        # 後ろから削除 (行番号がずれないように)
        for row_num in reversed(rows_to_delete):
            sheet.delete_rows(row_num)

        return len(rows_to_delete)
    except Exception:
        return 0
