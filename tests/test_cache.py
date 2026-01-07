"""cache.py のユニットテスト。gspreadをモックしてテスト。"""

from unittest.mock import MagicMock

from enemane_ai.cache import (
    CACHE_HEADER_ROW,
    ArticleCacheRow,
    CacheConfig,
    delete_cached_articles,
    ensure_header_row,
    generate_cache_key,
    get_cached_articles,
    save_articles_to_cache,
)


class TestGenerateCacheKey:
    """キャッシュキー生成のテスト。"""

    def test_generates_key_with_theme_and_building_type(self) -> None:
        key = generate_cache_key("法令改正", "オフィス")
        assert key == "法令改正_オフィス"

    def test_different_building_types_generate_different_keys(self) -> None:
        key1 = generate_cache_key("法令改正", "オフィス")
        key2 = generate_cache_key("法令改正", "工場")
        assert key1 != key2
        assert key1 == "法令改正_オフィス"
        assert key2 == "法令改正_工場"

    def test_different_themes_generate_different_keys(self) -> None:
        key1 = generate_cache_key("法令改正", "オフィス")
        key2 = generate_cache_key("他社事例", "オフィス")
        assert key1 != key2


class TestEnsureHeaderRow:
    """ヘッダー行確保のテスト。"""

    def test_adds_header_when_sheet_is_empty(self) -> None:
        """空のシートにヘッダーを追加する。"""
        mock_sheet = MagicMock()
        mock_sheet.row_values.return_value = []

        ensure_header_row(mock_sheet)

        mock_sheet.insert_row.assert_called_once_with(CACHE_HEADER_ROW, 1)

    def test_adds_header_when_first_row_is_different(self) -> None:
        """1行目が異なる場合はヘッダーを追加する。"""
        mock_sheet = MagicMock()
        mock_sheet.row_values.return_value = ["different_value", "other"]

        ensure_header_row(mock_sheet)

        mock_sheet.insert_row.assert_called_once_with(CACHE_HEADER_ROW, 1)

    def test_does_not_add_header_when_already_exists(self) -> None:
        """ヘッダーが既に存在する場合は追加しない。"""
        mock_sheet = MagicMock()
        mock_sheet.row_values.return_value = CACHE_HEADER_ROW

        ensure_header_row(mock_sheet)

        mock_sheet.insert_row.assert_not_called()


class TestGetCachedArticles:
    """キャッシュ取得のテスト。"""

    def test_cache_hit_returns_articles(self) -> None:
        """キャッシュがあれば記事を返す。"""
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_client.open_by_key.return_value.worksheet.return_value = mock_sheet

        mock_sheet.get_all_records.return_value = [
            {
                "cache_key": "法令改正_オフィス",
                "theme": "法令改正",
                "building_types": "オフィス",
                "title": "テスト記事",
                "content": "要約文",
                "image": "https://example.com/img.jpg",
                "link": "https://example.com",
                "created_at": "2026-01-06T10:00:00+00:00",
            }
        ]

        config = CacheConfig(spreadsheet_id="test-id")
        result = get_cached_articles("法令改正", "オフィス", mock_client, config)

        assert result is not None
        assert len(result) == 1
        assert result[0].title == "テスト記事"
        assert result[0].content == "要約文"
        assert result[0].theme == "法令改正"

    def test_cache_hit_returns_multiple_articles(self) -> None:
        """同じキャッシュキーで複数記事を返す。"""
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_client.open_by_key.return_value.worksheet.return_value = mock_sheet

        mock_sheet.get_all_records.return_value = [
            {
                "cache_key": "法令改正_オフィス",
                "theme": "法令改正",
                "title": "記事1",
                "content": "要約1",
                "image": "",
                "link": "https://example.com/1",
            },
            {
                "cache_key": "法令改正_オフィス",
                "theme": "法令改正",
                "title": "記事2",
                "content": "要約2",
                "image": "",
                "link": "https://example.com/2",
            },
            {
                "cache_key": "法令改正_オフィス",
                "theme": "法令改正",
                "title": "記事3",
                "content": "要約3",
                "image": "",
                "link": "https://example.com/3",
            },
        ]

        config = CacheConfig(spreadsheet_id="test-id")
        result = get_cached_articles("法令改正", "オフィス", mock_client, config)

        assert result is not None
        assert len(result) == 3

    def test_cache_miss_returns_none(self) -> None:
        """キャッシュがなければNoneを返す。"""
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_client.open_by_key.return_value.worksheet.return_value = mock_sheet
        mock_sheet.get_all_records.return_value = []

        config = CacheConfig(spreadsheet_id="test-id")
        result = get_cached_articles("法令改正", "オフィス", mock_client, config)

        assert result is None

    def test_different_cache_key_returns_none(self) -> None:
        """異なるキャッシュキーはNoneを返す。"""
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_client.open_by_key.return_value.worksheet.return_value = mock_sheet

        mock_sheet.get_all_records.return_value = [
            {
                "cache_key": "法令改正_工場",
                "theme": "法令改正",
                "title": "別の記事",
                "content": "別の要約",
                "image": "",
                "link": "https://example.com",
            }
        ]

        config = CacheConfig(spreadsheet_id="test-id")
        result = get_cached_articles("法令改正", "オフィス", mock_client, config)

        assert result is None

    def test_exception_propagates(self) -> None:
        """例外発生時は呼び出し元に伝播する。"""
        import pytest

        mock_client = MagicMock()
        mock_client.open_by_key.side_effect = Exception("Connection error")

        config = CacheConfig(spreadsheet_id="test-id")

        with pytest.raises(Exception, match="Connection error"):
            get_cached_articles("法令改正", "オフィス", mock_client, config)


class TestSaveArticlesToCache:
    """キャッシュ保存のテスト。"""

    def test_saves_articles_with_correct_format(self) -> None:
        """記事が正しいフォーマットで保存される。"""
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_client.open_by_key.return_value.worksheet.return_value = mock_sheet
        mock_sheet.get_all_records.return_value = []

        articles = [
            ArticleCacheRow(
                theme="法令改正",
                title="テスト",
                content="要約",
                image="https://example.com/img.jpg",
                link="https://example.com",
            )
        ]

        config = CacheConfig(spreadsheet_id="test-id")
        save_articles_to_cache("法令改正", "オフィス", articles, mock_client, config)

        mock_sheet.append_rows.assert_called_once()

        # 保存されたデータの検証
        call_args = mock_sheet.append_rows.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0][0] == "法令改正_オフィス"  # cache_key
        assert call_args[0][1] == "法令改正"  # theme
        assert call_args[0][2] == "オフィス"  # building_type
        assert call_args[0][3] == "テスト"  # title
        assert call_args[0][4] == "要約"  # content

    def test_saves_multiple_articles(self) -> None:
        """複数記事を保存できる。"""
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_client.open_by_key.return_value.worksheet.return_value = mock_sheet
        mock_sheet.get_all_records.return_value = []

        articles = [
            ArticleCacheRow(
                theme="法令改正",
                title=f"記事{i}",
                content=f"要約{i}",
                image="",
                link=f"https://example.com/{i}",
            )
            for i in range(3)
        ]

        config = CacheConfig(spreadsheet_id="test-id")
        save_articles_to_cache("法令改正", "オフィス", articles, mock_client, config)

        call_args = mock_sheet.append_rows.call_args[0][0]
        assert len(call_args) == 3

    def test_deletes_existing_cache_before_save(self) -> None:
        """保存前に既存キャッシュを削除する。"""
        mock_client = MagicMock()
        mock_sheet = MagicMock()
        mock_client.open_by_key.return_value.worksheet.return_value = mock_sheet
        mock_sheet.get_all_records.return_value = [
            {"cache_key": "法令改正_オフィス", "title": "古い記事"},
        ]

        articles = [
            ArticleCacheRow(
                theme="法令改正",
                title="新しい記事",
                content="新しい要約",
                image="",
                link="https://example.com",
            )
        ]

        config = CacheConfig(spreadsheet_id="test-id")
        save_articles_to_cache("法令改正", "オフィス", articles, mock_client, config)

        # 削除が呼ばれたことを確認
        mock_sheet.delete_rows.assert_called()

    def test_exception_propagates(self) -> None:
        """例外発生時は呼び出し元に伝播する。"""
        import pytest

        mock_client = MagicMock()
        mock_client.open_by_key.side_effect = Exception("Connection error")

        articles = [
            ArticleCacheRow(
                theme="法令改正",
                title="テスト",
                content="要約",
                image="",
                link="https://example.com",
            )
        ]

        config = CacheConfig(spreadsheet_id="test-id")

        with pytest.raises(Exception, match="Connection error"):
            save_articles_to_cache("法令改正", "オフィス", articles, mock_client, config)


class TestDeleteCachedArticles:
    """キャッシュ削除のテスト。"""

    def test_deletes_matching_rows(self) -> None:
        """一致するキャッシュキーの行を削除する。"""
        mock_sheet = MagicMock()
        mock_sheet.get_all_records.return_value = [
            {"cache_key": "法令改正_オフィス"},
            {"cache_key": "法令改正_工場"},
            {"cache_key": "法令改正_オフィス"},
        ]

        deleted = delete_cached_articles("法令改正_オフィス", mock_sheet)

        assert deleted == 2
        # 後ろから削除されることを確認 (行番号: 4, 2)
        assert mock_sheet.delete_rows.call_count == 2

    def test_no_matching_rows(self) -> None:
        """一致する行がない場合は0を返す。"""
        mock_sheet = MagicMock()
        mock_sheet.get_all_records.return_value = [
            {"cache_key": "法令改正_工場"},
        ]

        deleted = delete_cached_articles("法令改正_オフィス", mock_sheet)

        assert deleted == 0
        mock_sheet.delete_rows.assert_not_called()

    def test_exception_returns_zero(self) -> None:
        """例外発生時は0を返す。"""
        mock_sheet = MagicMock()
        mock_sheet.get_all_records.side_effect = Exception("Error")

        deleted = delete_cached_articles("法令改正_オフィス", mock_sheet)

        assert deleted == 0
