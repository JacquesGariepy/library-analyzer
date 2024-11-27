from enum import Enum
import inspect
import asyncio
import contextlib
from typing import Dict
from dataclasses import is_dataclass, fields
from .element import ElementType

class ElementType(Enum):
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    PROPERTY = "property"
    MODULE = "module"
    VARIABLE = "variable"
    ENUM = "enum"
    CONSTANT = "constant"
    DATACLASS = "dataclass"
    COROUTINE = "coroutine"
    GENERATOR = "generator"
    DESCRIPTOR = "descriptor"
    EXCEPTION = "exception"
    PROTOCOL = "protocol"

def analyze_element(obj, name: str, module_name: str) -> Dict:
    """Analyze an individual library element."""
    try:
        if id(obj) in self.explored:
            return {}
            
        self.explored.add(id(obj))
        current_path = '.'.join(self.current_path + [name])
        
        element_type = self.get_element_type(obj)
        element_info = {
            'type': element_type.value,
            'name': name,
            'path': current_path,
            'module': module_name
        }

        if element_type in [ElementType.METHOD, ElementType.FUNCTION, ElementType.COROUTINE, ElementType.GENERATOR]:
            element_info.update(self.get_signature_info(obj))
        elif element_type == ElementType.CLASS:
            element_info.update(self.get_class_info(obj))
            
            self.current_path.append(name)
            for attr_name, attr_value in inspect.getmembers(obj):
                if not attr_name.startswith('_'):
                    with contextlib.suppress(Exception):
                        sub_info = self.analyze_element(attr_value, attr_name, module_name)
                        if sub_info:
                            if 'members' not in element_info:
                                element_info['members'] = {}
                            element_info['members'][attr_name] = sub_info
            self.current_path.pop()
            
        elif element_type == ElementType.MODULE:
            if obj.__name__.startswith(module_name):
                self.current_path.append(name)
                element_info['members'] = {}
                for attr_name, attr_value in inspect.getmembers(obj):
                    if not attr_name.startswith('_'):
                        with contextlib.suppress(Exception):
                            sub_info = self.analyze_element(attr_value, attr_name, module_name)
                            if sub_info:
                                element_info['members'][attr_name] = sub_info
                self.current_path.pop()
                
        elif element_type == ElementType.PROPERTY:
            element_info['docstring'] = inspect.getdoc(obj)
            for accessor in ['fget', 'fset', 'fdel']:
                if hasattr(obj, accessor):
                    accessor_obj = getattr(obj, accessor)
                    if accessor_obj:
                        element_info[accessor] = self.get_signature_info(accessor_obj)

        return element_info
    except Exception as e:
        self.errors.append(f"Error analyzing element {name}: {str(e)}")
        return {}

def get_element_type(obj) -> ElementType:
    """Determine the precise type of an element."""
    try:
        if inspect.ismodule(obj):
            return ElementType.MODULE
        elif inspect.isclass(obj):
            if issubclass(obj, Exception):
                return ElementType.EXCEPTION
            elif is_dataclass(obj):
                return ElementType.DATACLASS
            elif issubclass(obj, Enum):
                return ElementType.ENUM
            elif hasattr(obj, '__protocol__'):
                return ElementType.PROTOCOL
            return ElementType.CLASS
        elif inspect.ismethod(obj) or inspect.isfunction(obj):
            if asyncio.iscoroutinefunction(obj):
                return ElementType.COROUTINE
            elif inspect.isgeneratorfunction(obj):
                return ElementType.GENERATOR
            elif inspect.ismethod(obj):
                return ElementType.METHOD
            return ElementType.FUNCTION
        elif isinstance(obj, property):
            return ElementType.PROPERTY
        elif isinstance(obj, (int, float, str, bool)) and \
                (isinstance(obj, str) and obj.isupper() or \
                 not isinstance(obj, str)):
            return ElementType.CONSTANT
        elif hasattr(obj, '__get__') and hasattr(obj, '__set__'):
            return ElementType.DESCRIPTOR
        elif isinstance(obj, (int, float, str, bool, list, dict, tuple, set)):
            return ElementType.VARIABLE
        else:
            return ElementType.VARIABLE
    except Exception as e:
        self.errors.append(f"Error determining type for {obj}: {str(e)}")
        return ElementType.VARIABLE
