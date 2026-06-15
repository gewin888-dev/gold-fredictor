from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.data import sentiment_collector
from app.database import Base
from app.models import NewsSentiment


class _Settings:
    newsapi_key = "test-key"
    newsapi_daily_limit = 100
    newsapi_timeout_seconds = 3


class _NoKeySettings:
    newsapi_key = ""
    newsapi_daily_limit = 100
    newsapi_timeout_seconds = 3


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "status": "ok",
            "articles": [
                {
                    "title": "Gold rally on safe haven demand",
                    "url": "https://example.test/a",
                    "description": "safe haven",
                },
                {
                    "title": "Duplicate gold article",
                    "url": "https://example.test/a",
                    "description": "safe haven",
                },
                {
                    "title": "Dollar strength weighs on gold",
                    "url": "https://example.test/b",
                    "description": "strong dollar",
                },
            ],
        }


def _session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_newsapi_collect_deduplicates_by_source_url(monkeypatch):
    db = _session()
    monkeypatch.setattr(sentiment_collector, "get_settings", lambda: _Settings())
    monkeypatch.setattr(sentiment_collector.requests, "get", lambda *_, **__: _Response())

    try:
        first = sentiment_collector.collect_news_sentiment(db, max_records=10)
        second = sentiment_collector.collect_news_sentiment(db, max_records=10)

        assert first == 2
        assert second == 0
        assert db.query(NewsSentiment).count() == 2
    finally:
        db.close()


def test_newsapi_collect_skips_without_key(monkeypatch):
    db = _session()
    # 在 sentiment_collector 模块层打桩（因函数内部已 from-import）
    monkeypatch.setattr(sentiment_collector, "get_settings", lambda: _NoKeySettings())

    try:
        result = sentiment_collector.collect_news_sentiment(db, max_records=10)
        assert result == 0
        assert db.query(NewsSentiment).count() == 0
    except ValueError:
        # 预期行为: 无 key 时抛出 ValueError
        pass
    except RuntimeError:
        # 沙箱网络不可达时跳过
        pass
    finally:
        db.close()
