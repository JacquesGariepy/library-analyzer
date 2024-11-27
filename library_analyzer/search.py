import importlib
import inspect
import re
import logging
from typing import List, Set, Dict
from .logging_config import setup_logging

# Configurer le logging
setup_logging()
logger = logging.getLogger(__name__)

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
        logger.error(f"Error discovering API endpoints: {str(e)}")
        return []

def find_urls(library_name: str) -> List[str]:
    """Find URLs in the library."""
    urls = []
    try:
        module = importlib.import_module(library_name)
        visited = set()
        _find_urls_in_object(module, urls, visited)
    except Exception as e:
        logger.error(f"Error finding URLs: {str(e)}")
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

def perform_search(analyzer, analysis: Dict, search_query: str) -> List[str]:
    """
    Perform a search on the analyzed data.

    Args:
        analyzer (LibraryAnalyzer): The analyzer instance.
        analysis (dict): The analysis results.
        search_query (str): The search query.

    Returns:
        list: The search results.
    """
    results = []
    for item in analysis['data']:
        if search_query.lower() in item['text'].lower():
            results.append(item['text'])
    return results
