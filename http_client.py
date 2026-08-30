import json
import asyncio
import urllib.request
import urllib.error

async def async_post_json(url: str, payload: dict, headers: dict = None, timeout: float = 4.0) -> tuple:
    """
    High-compatibility async HTTP POST function.
    Works with standard Python urllib or aiohttp if available.
    Returns (status_code: int, data: dict or str)
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers or {"Content-Type": "application/json"}, timeout=timeout) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = await resp.text()
                return resp.status, data
    except ImportError:
        def _sync_post():
            req_headers = {"Content-Type": "application/json"}
            if headers:
                req_headers.update(headers)
            body_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    res_body = response.read().decode("utf-8")
                    try:
                        return response.status, json.loads(res_body)
                    except Exception:
                        return response.status, res_body
            except urllib.error.HTTPError as e:
                return e.code, {}
            except Exception:
                return 500, {}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_post)

async def async_get_json(url: str, headers: dict = None, timeout: float = 4.0) -> tuple:
    """
    High-compatibility async HTTP GET function.
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = await resp.text()
                return resp.status, data
    except ImportError:
        def _sync_get():
            req = urllib.request.Request(url, headers=headers or {}, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    res_body = response.read().decode("utf-8")
                    try:
                        return response.status, json.loads(res_body)
                    except Exception:
                        return response.status, res_body
            except urllib.error.HTTPError as e:
                return e.code, {}
            except Exception:
                return 500, {}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_get)
