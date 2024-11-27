import inspect
import json
import os
import yaml
import logging
from .logging_config import setup_logging

# Configurer le logging
setup_logging()
logger = logging.getLogger(__name__)

def explore_module(obj, path="", depth=0, max_depth=5, explored=None):
    """
    Recursively explore a module or class to find potential API endpoints.

    Args:
        obj (object): The module or class to explore.
        path (str): The current path of the object being explored.
        depth (int): The current depth of the exploration.
        max_depth (int): The maximum depth to explore.
        explored (set): A set of already explored object IDs to avoid infinite loops.

    Returns:
        list: A list of dictionaries containing information about the discovered endpoints.
    """
    if explored is None:
        explored = set()
    
    obj_id = id(obj)
    if obj_id in explored or depth > max_depth:
        return []
    
    explored.add(obj_id)
    endpoints = []
    
    try:
        for name in dir(obj):
            if name.startswith('_'):
                continue
                
            try:
                attr = getattr(obj, name)
            except:
                continue
                
            current_path = f"{path}.{name}" if path else name
            
            if inspect.ismethod(attr) or inspect.isfunction(attr):
                if any(keyword in name.lower() for keyword in ['create', 'list', 'get', 'delete', 'update', 'retrieve']):
                    sig = inspect.signature(attr)
                    doc = inspect.getdoc(attr)
                    parameters = []
                    for param in sig.parameters.values():
                        parameters.append({
                            'name': param.name,
                            'type': str(param.annotation),
                            'default': param.default if param.default is not param.empty else None
                        })
                    endpoints.append({
                        'path': current_path,
                        'type': 'method',
                        'parameters': parameters,
                        'return_type': str(sig.return_annotation),
                        'docstring': doc
                    })
            
            elif inspect.isclass(attr) or inspect.ismodule(attr):
                class_info = {
                    'path': current_path,
                    'type': 'class' if inspect.isclass(attr) else 'module',
                    'bases': [base.__name__ for base in attr.__bases__] if inspect.isclass(attr) else [],
                    'docstring': inspect.getdoc(attr),
                    'members': explore_module(attr, current_path, depth + 1, max_depth, explored)
                }
                endpoints.append(class_info)
                
    except Exception as e:
        pass
        
    return endpoints

def parse_json_file(file_path):
    """
    Parse a JSON file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {}
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_function_signatures(data):
    """
    Extract function signatures from the analyzed data.

    Args:
        data (dict): The analyzed data.

    Returns:
        dict: A dictionary containing function signatures.
    """
    signatures = {}

    def extract_from_members(members):
        for name, info in members.items():
            if 'parameters' in info:
                signatures[name] = info['parameters']
            if 'members' in info:
                extract_from_members(info['members'])

    if 'members' in data:
        extract_from_members(data['members'])

    return signatures

def load_config():
    """
    Load the configuration from a YAML file.

    Returns:
        dict: The configuration data.
    """
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info("Configuration loaded successfully.")
    return config
