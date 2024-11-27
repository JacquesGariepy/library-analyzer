from enum import Enum
import inspect
import asyncio
import contextlib
from typing import Dict
from dataclasses import is_dataclass, fields

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
