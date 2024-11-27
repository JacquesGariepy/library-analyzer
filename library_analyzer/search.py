import importlib
import inspect
import re
from typing import List, Set

def discover_api_endpoints(library_name: str) -> List[str]:
    """Discover API endpoints provided by the library."""
    try:
        module = importlib.import_module(library_name)
        endpoints = []
        for name, obj in inspect.getmembers(module):
            if hasattr(obj, 'route'):
                endpoints.append(obj.route)
        return endpoints
    except Exception as e:
        return []

def find_urls(library_name: str) -> List[str]:
    """Find URLs in the library."""
    urls = []
    try:
        module = importlib.import_module(library_name)
        visited = set()
        _find_urls_in_object(module, urls, visited)
    except Exception as e:
        pass
    return urls

def _find_urls_in_object(obj, urls: List[str], visited: Set[int]):
    """Recursively find URLs in an object."""
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(obj, str) and obj.startswith("http"):
        urls.append(obj)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            _find_urls_in_object(item, urls, visited)
    elif inspect.ismodule(obj) or inspect.isclass(obj):
        try:
            source = inspect.getsource(obj)
            urls.extend(extract_urls_from_source(source))
        except Exception:
            pass
        for name, member in inspect.getmembers(obj):
            if not name.startswith('_'):
                _find_urls_in_object(member, urls, visited)

def extract_urls_from_source(source: str) -> List[str]:
    """Extract URLs from the source code."""
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(source)
