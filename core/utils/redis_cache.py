import json
import hashlib
import functools
import inspect
import os
from typing import Callable, Any, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import re
import redis

class L1Cache:
    def __init__(self, redis_client: redis.Redis, prefix: str = "l1cache:", ttl: Optional[int] = None):
        """
        L1Cache 类，用于 Redis L1 缓存，支持 decorator 包装函数。
        
        :param redis_client: 已初始化的 redis.Redis 实例
        :param prefix: 缓存 key 前缀，默认 "l1cache:"
        :param ttl: 默认过期时间（秒），None 表示不设置
        """
        self.redis = redis_client
        self.prefix = prefix
        self.default_ttl = ttl

    def __call__(self, ttl: Optional[int] = None) -> Callable:
        return self.cache(ttl=ttl)

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _normalize_query_like_text(self, value: str) -> str:
        return self._normalize_text(value).lower()

    def _normalize_url(self, value: str) -> str:
        url = self._normalize_text(value)
        parsed = urlsplit(url)

        if not parsed.scheme and not parsed.netloc:
            return url

        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower()

        port = parsed.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        user_info = ""
        if parsed.username:
            user_info = parsed.username
            if parsed.password:
                user_info += f":{parsed.password}"
            user_info += "@"

        netloc = f"{user_info}{host}"
        if port is not None:
            netloc += f":{port}"

        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")

        tracking_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "gclid",
            "fbclid",
            "msclkid",
        }
        query_items = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() not in tracking_params
        ]
        query_items.sort(key=lambda item: (item[0], item[1]))
        query = urlencode(query_items, doseq=True)

        return urlunsplit((scheme, netloc, path, query, ""))

    def _normalize_value(self, value: Any, param_name: Optional[str] = None) -> Any:
        lowered_name = (param_name or "").lower()

        if isinstance(value, str):
            if any(token in lowered_name for token in ("url", "uri", "link", "site", "web")):
                return self._normalize_url(value)

            if any(
                token in lowered_name
                for token in ("query", "question", "purpose", "keyword", "search", "location", "city", "place")
            ):
                return self._normalize_query_like_text(value)

            return self._normalize_text(value)

        if isinstance(value, dict):
            normalized_dict = {}
            for key in sorted(value.keys(), key=lambda k: str(k)):
                key_name = key if isinstance(key, str) else None
                normalized_dict[key] = self._normalize_value(value[key], key_name)
            return normalized_dict

        if isinstance(value, list):
            return [self._normalize_value(item, param_name) for item in value]

        if isinstance(value, tuple):
            return tuple(self._normalize_value(item, param_name) for item in value)

        if isinstance(value, set):
            normalized_items = [self._normalize_value(item, param_name) for item in value]
            return sorted(normalized_items, key=lambda item: json.dumps(item, sort_keys=True, default=str, ensure_ascii=False))

        return value

    def _build_cache_key(self, func: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        try:
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            normalized_payload = {
                name: self._normalize_value(value, name)
                for name, value in bound.arguments.items()
                if name not in {"self", "cls"}
            }
        except Exception:
            normalized_payload = {
                "args": self._normalize_value(args),
                "kwargs": self._normalize_value(kwargs),
            }

        payload = json.dumps(
            normalized_payload,
            sort_keys=True,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        key_hash = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return f"{self.prefix}{func.__name__}:{key_hash}"

    def cache(self, ttl: Optional[int] = None) -> Callable:
        """
        Decorator，用于缓存函数结果。
        
        :param ttl: 该函数的过期时间，优先级高于默认 ttl
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                cache_key = self._build_cache_key(func, args, kwargs)
                
                # 尝试从 Redis 获取
                cached = self.redis.get(cache_key)
                if cached is not None:
                    try:
                        return json.loads(cached)
                    except json.JSONDecodeError:
                        # 缓存损坏，删除并重新计算
                        self.redis.delete(cache_key)
                
                # 缓存 miss，执行函数
                result = func(*args, **kwargs)
                
                # 存入 Redis
                ttl_to_use = ttl or self.default_ttl
                if ttl_to_use:
                    self.redis.setex(cache_key, ttl_to_use, json.dumps(result, default=str, ensure_ascii=False))
                else:
                    self.redis.set(cache_key, json.dumps(result, default=str, ensure_ascii=False))
                
                return result
            
            return wrapper
        
        return decorator

# 使用示例（基于你的连接）
r = redis.Redis.from_url(os.environ["REDIS_URL"])

l1cache = L1Cache(r, prefix="app:", ttl=3600 * 24)
