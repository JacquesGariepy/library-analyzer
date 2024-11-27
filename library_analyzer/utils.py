import inspect
import json
import os
import yaml

def explore_module(obj, path="", depth=0, max_depth=5, explored=None):
    if explored is None:
        explored = set()
    
    # Avoid infinite loops
    obj_id = id(obj)
    if obj_id in explored or depth > max_depth:
        return []
    
    explored.add(obj_id)
    endpoints = []
    
    try:
        # Explore the object's attributes
        for name in dir(obj):
            if name.startswith('_'):
                continue
                
            try:
                attr = getattr(obj, name)
            except:
                continue
                
            current_path = f"{path}.{name}" if path else name
            
            # Check if it's a method that could be an endpoint
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
            
            # Recursively explore classes and modules
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
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_function_signatures(data):
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
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config
