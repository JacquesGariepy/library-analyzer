# import inspect
# import types
# from typing import Any, Dict, List, Set, Union, get_type_hints, ClassVar, TypeVar, Generic
# import importlib
# import sys
# from enum import Enum
# import json
# from dataclasses import is_dataclass, fields
# from functools import partial
# import asyncio
# import typing
# import warnings
# import contextlib
# import os
# from whoosh.index import create_in
# from whoosh.fields import Schema, TEXT, ID
# from whoosh.qparser import QueryParser
# from whoosh import index
# from sentence_transformers import SentenceTransformer
# import torch
# import faiss
# import numpy as np
# import yaml

# class ElementType(Enum):
#     CLASS = "class"
#     METHOD = "method"
#     FUNCTION = "function"
#     PROPERTY = "property"
#     MODULE = "module"
#     VARIABLE = "variable"
#     ENUM = "enum"
#     CONSTANT = "constant"
#     DATACLASS = "dataclass"
#     COROUTINE = "coroutine"
#     GENERATOR = "generator"
#     DESCRIPTOR = "descriptor"
#     EXCEPTION = "exception"
#     PROTOCOL = "protocol"

# class LibraryAnalyzer:
#     """
#     LibraryAnalyzer is a class designed to analyze Python libraries and extract detailed information about their elements, such as classes, functions, methods, and modules. It provides methods to evaluate types, extract type information, analyze signatures, and gather comprehensive details about classes and their members.
#     Attributes:
#         explored (set): A set to keep track of explored elements to avoid redundant analysis.
#         errors (list): A list to store error messages encountered during analysis.
#         current_path (list): A list to maintain the current path of elements being analyzed.
#         library_elements (dict): A dictionary to store information about analyzed library elements.
#         type_namespace (dict): A dictionary to store type information for safe evaluation.
#         ix (index.Index): An instance of the Whoosh index.
#         bert_model (SentenceTransformer): An instance of the BERT model.
#         faiss_index (faiss.Index): An instance of the FAISS index.
#     Methods:
#         __init__(): Initializes the LibraryAnalyzer instance and sets up the type environment.
#         _setup_type_environment(): Configures the environment for type management.
#         safe_eval(type_str: str): Safely evaluates a type string.
#         get_type_info(typ) -> str: Converts a type to a string representation safely.
#         get_signature_info(obj) -> Dict: Extracts signature information from a function/method.
#         get_class_info(obj) -> Dict: Extracts detailed information from a class.
#         analyze_element(obj, name: str, module_name: str) -> Dict: Analyzes an individual library element.
#         get_element_type(obj) -> ElementType: Determines the precise type of an element.
#         analyze_library(library_name: str) -> Dict: Performs a complete analysis of a library.
#         save_analysis(analysis: Dict, output_file: str): Saves the analysis to a JSON file.
#         extract_text_data(analysis: Dict) -> List[Dict]: Extracts relevant text data from the analysis results.
#         index_data(data: List[Dict]): Indexes the extracted data using Whoosh and BERT.
#         search(query: str, use_bert: bool = True, use_whoosh: bool = True, top_k: int = 3) -> List[Dict]: Searches the indexed data using Whoosh and BERT.
#     """
#     def __init__(self):
#         self.explored = set()
#         self.errors = []
#         self.current_path = []
#         self.library_elements = {}
#         self._setup_type_environment()
#         schema = Schema(path=ID(stored=True), text=TEXT(stored=True))
#         if not os.path.exists("indexdir"):
#             os.mkdir("indexdir")
#         self.ix = create_in("indexdir", schema)
#         self.bert_model = self.load_bert_model()
#         self.faiss_index = self.create_faiss_index()
#         self.documents = []

#     def _setup_type_environment(self):
#         """Configure the environment for type management."""
#         # Add common types to the global namespace
#         self.type_namespace = {
#             'ClassVar': ClassVar,
#             'TypeVar': TypeVar,
#             'Generic': Generic,
#             'Any': Any,
#             'Union': Union,
#             'List': List,
#             'Dict': Dict,
#             'Set': Set,
#             'Optional': typing.Optional,
#             # Add other types specific to OpenAI
#             # 'OpenAI': type('OpenAI', (), {}),
#             # 'AsyncOpenAI': type('AsyncOpenAI', (), {}),
#         }

#     def safe_eval(self, type_str: str):
#         """Safely evaluate a type string."""
#         try:
#             with contextlib.suppress(Exception):
#                 return eval(type_str, self.type_namespace, {})
#         except:
#             return Any

#     def get_type_info(self, typ) -> str:
#         """Convert a type to a string representation safely."""
#         try:
#             if typ == inspect.Signature.empty:
#                 return 'Any'
            
#             # Handle generic types
#             origin = typing.get_origin(typ)
#             if origin is not None:
#                 args = typing.get_args(typ)
#                 if args:
#                     args_str = ', '.join(self.get_type_info(arg) for arg in args)
#                     return f"{origin.__name__}[{args_str}]"
#                 return origin.__name__
            
#             # Handle simple types
#             if isinstance(typ, type):
#                 return typ.__name__
            
#             # Handle TypeVar
#             if isinstance(typ, TypeVar):
#                 return f"TypeVar('{typ.__name__}')"
            
#             return str(typ)
#         except Exception as e:
#             self.errors.append(f"Error getting type info for {typ}: {str(e)}")
#             return 'Any'

#     def get_signature_info(self, obj) -> Dict:
#         """Extract signature information from a function/method."""
#         try:
#             if inspect.isfunction(obj) or inspect.ismethod(obj):
#                 sig = inspect.signature(obj)
#                 return {
#                     'parameters': {
#                         name: {
#                             'kind': str(param.kind),
#                             'default': str(param.default) if param.default != inspect.Parameter.empty else None,
#                             'annotation': self.get_type_info(param.annotation)
#                         }
#                         for name, param in sig.parameters.items()
#                     },
#                     'return_annotation': self.get_type_info(sig.return_annotation),
#                     'docstring': inspect.getdoc(obj),
#                     'is_async': asyncio.iscoroutinefunction(obj),
#                     'is_generator': inspect.isgeneratorfunction(obj),
#                 }
#         except Exception as e:
#             self.errors.append(f"Error getting signature for {obj}: {str(e)}")
#         return {}

#     def get_class_info(self, obj) -> Dict:
#         """Extract detailed information from a class."""
#         try:
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
                
#                 info = {
#                     'bases': [self.get_type_info(base) for base in obj.__bases__],
#                     'docstring': inspect.getdoc(obj),
#                     'methods': {},
#                     'properties': {},
#                     'class_variables': {},
#                     'instance_variables': {}
#                 }
                
#                 # Retrieve type annotations safely
#                 if hasattr(obj, '__annotations__'):
#                     try:
#                         hints = get_type_hints(obj, globalns=self.type_namespace)
#                         info['type_hints'] = {
#                             name: self.get_type_info(hint) 
#                             for name, hint in hints.items()
#                         }
#                     except Exception as e:
#                         info['type_hints'] = {
#                             name: str(hint) 
#                             for name, hint in obj.__annotations__.items()
#                         }

#                 # For dataclasses
#                 if is_dataclass(obj):
#                     info['dataclass_fields'] = {}
#                     for field in fields(obj):
#                         field_info = {
#                             'type': self.get_type_info(field.type),
#                             'default': str(field.default),
#                             'metadata': field.metadata
#                         }
#                         info['dataclass_fields'][field.name] = field_info

#                 # For enums
#                 if issubclass(obj, Enum):
#                     info['enum_values'] = {
#                         name: str(value.value)
#                         for name, value in obj.__members__.items()
#                     }

#                 return info
#         except Exception as e:
#             self.errors.append(f"Error getting class info for {obj}: {str(e)}")
#             return {}

#     def analyze_element(self, obj, name: str, module_name: str) -> Dict:
#         """Analyze an individual library element."""
#         try:
#             if id(obj) in self.explored:
#                 return {}
                
#             self.explored.add(id(obj))
#             current_path = '.'.join(self.current_path + [name])
            
#             element_type = self.get_element_type(obj)
#             element_info = {
#                 'type': element_type.value,
#                 'name': name,
#                 'path': current_path,
#                 'module': module_name
#             }

#             if element_type in [ElementType.METHOD, ElementType.FUNCTION, ElementType.COROUTINE, ElementType.GENERATOR]:
#                 element_info.update(self.get_signature_info(obj))
#             elif element_type == ElementType.CLASS:
#                 element_info.update(self.get_class_info(obj))
                
#                 self.current_path.append(name)
#                 for attr_name, attr_value in inspect.getmembers(obj):
#                     if not attr_name.startswith('_'):
#                         with contextlib.suppress(Exception):
#                             sub_info = self.analyze_element(attr_value, attr_name, module_name)
#                             if sub_info:
#                                 if 'members' not in element_info:
#                                     element_info['members'] = {}
#                                 element_info['members'][attr_name] = sub_info
#                 self.current_path.pop()
                
#             elif element_type == ElementType.MODULE:
#                 if obj.__name__.startswith(module_name):
#                     self.current_path.append(name)
#                     element_info['members'] = {}
#                     for attr_name, attr_value in inspect.getmembers(obj):
#                         if not attr_name.startswith('_'):
#                             with contextlib.suppress(Exception):
#                                 sub_info = self.analyze_element(attr_value, attr_name, module_name)
#                                 if sub_info:
#                                     element_info['members'][attr_name] = sub_info
#                     self.current_path.pop()
                    
#             elif element_type == ElementType.PROPERTY:
#                 element_info['docstring'] = inspect.getdoc(obj)
#                 for accessor in ['fget', 'fset', 'fdel']:
#                     if hasattr(obj, accessor):
#                         accessor_obj = getattr(obj, accessor)
#                         if accessor_obj:
#                             element_info[accessor] = self.get_signature_info(accessor_obj)

#             return element_info
#         except Exception as e:
#             self.errors.append(f"Error analyzing element {name}: {str(e)}")
#             return {}

#     def get_element_type(self, obj) -> ElementType:
#         """Determine the precise type of an element."""
#         try:
#             if inspect.ismodule(obj):
#                 return ElementType.MODULE
#             elif inspect.isclass(obj):
#                 if issubclass(obj, Exception):
#                     return ElementType.EXCEPTION
#                 elif is_dataclass(obj):
#                     return ElementType.DATACLASS
#                 elif issubclass(obj, Enum):
#                     return ElementType.ENUM
#                 elif hasattr(obj, '__protocol__'):
#                     return ElementType.PROTOCOL
#                 return ElementType.CLASS
#             elif inspect.ismethod(obj) or inspect.isfunction(obj):
#                 if asyncio.iscoroutinefunction(obj):
#                     return ElementType.COROUTINE
#                 elif inspect.isgeneratorfunction(obj):
#                     return ElementType.GENERATOR
#                 elif inspect.ismethod(obj):
#                     return ElementType.METHOD
#                 return ElementType.FUNCTION
#             elif isinstance(obj, property):
#                 return ElementType.PROPERTY
#             elif isinstance(obj, (int, float, str, bool)) and \
#                     (isinstance(obj, str) and obj.isupper() or \
#                      not isinstance(obj, str)):
#                 return ElementType.CONSTANT
#             elif hasattr(obj, '__get__') and hasattr(obj, '__set__'):
#                 return ElementType.DESCRIPTOR
#             elif isinstance(obj, (int, float, str, bool, list, dict, tuple, set)):
#                 return ElementType.VARIABLE
#             else:
#                 return ElementType.VARIABLE
#         except Exception as e:
#             self.errors.append(f"Error determining type for {obj}: {str(e)}")
#             return ElementType.VARIABLE

#     def analyze_library(self, library_name: str) -> Dict:
#         """Perform a complete analysis of a library."""
#         try:
#             # Import the library
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 module = importlib.import_module(library_name)
            
#             # Reset state
#             self.explored = set()
#             self.errors = []
#             self.current_path = []
#             self.library_elements = {}
            
#             # Analyze the main module
#             library_info = self.analyze_element(module, library_name, library_name)
            
#             # Add metadata information
#             library_info['metadata'] = {
#                 'name': library_name,
#                 'version': getattr(module, '__version__', 'Unknown'),
#                 'file': getattr(module, '__file__', 'Unknown'),
#                 'doc': inspect.getdoc(module),
#                 'analysis_errors': self.errors
#             }
            
#             return library_info
            
#         except Exception as e:
#             return {
#                 'error': f"Failed to analyze library {library_name}: {str(e)}",
#                 'analysis_errors': self.errors
#             }

#     def save_analysis(self, analysis: Dict, output_file: str):
#         """Save the analysis to a JSON file."""
#         try:
#             base_name, ext = os.path.splitext(output_file)
#             version = analysis.get('metadata', {}).get('version', 'Unknown')
#             output_file = f"{base_name}_v{version}{ext}"
            
#             counter = 1
#             original_output_file = output_file
#             while os.path.exists(output_file):
#                 output_file = f"{base_name}_v{version}_{counter}{ext}"
#                 counter += 1

#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(analysis, f, indent=2, ensure_ascii=False)
#             print(f"Analysis saved to {output_file}")
#         except Exception as e:
#             print(f"Error saving analysis: {str(e)}")

#     def extract_text_data(self, analysis: Dict) -> List[Dict]:
#         """Extract relevant text data from the analysis results."""
#         text_data = []

#         def extract_from_element(element, path=""):
#             if isinstance(element, dict):
#                 if 'docstring' in element and element['docstring']:
#                     text_data.append({
#                         'path': path,
#                         'text': element['docstring']
#                     })
#                 if 'members' in element:
#                     for name, member in element['members'].items():
#                         extract_from_element(member, f"{path}.{name}" if path else name)

#         extract_from_element(analysis)
#         return text_data

#     def index_data(self, data: List[Dict]):
#         """Index the extracted data using Whoosh and BERT."""
#         # Index data using Whoosh
#         writer = self.ix.writer()
#         for item in data:
#             writer.add_document(path=item['path'], text=item['text'])
#         writer.commit()

#         # Index data using BERT and FAISS
#         self.documents = [item['text'] for item in data]  # Store documents in the instance variable
#         embeddings = self.bert_model.encode(self.documents)
#         self.faiss_index.add(embeddings)

#     def search(self, query: str, use_bert: bool = True, use_whoosh: bool = True, top_k: int = 3) -> List[Dict]:
#         """Search the indexed data using Whoosh and BERT."""
#         results = []
        
#         # Whoosh search
#         if use_whoosh:
#             with self.ix.searcher() as searcher:
#                 query_parser = QueryParser("text", self.ix.schema)
#                 whoosh_query = query_parser.parse(query)
#                 whoosh_results = searcher.search(whoosh_query, limit=top_k)
#                 results.extend([
#                     {"source": "Whoosh", "path": hit["path"], "text": hit["text"], "score": hit.score}
#                     for hit in whoosh_results
#                 ])
        
#         # BERT search
#         if use_bert:
#             query_embedding = self.bert_model.encode([query])
#             distances, indices = self.faiss_index.search(query_embedding, top_k)
#             for i, idx in enumerate(indices[0]):
#                 results.append({
#                     "source": "BERT",
#                     "path": "",  # BERT results do not have a path, so we leave it empty
#                     "text": self.documents[idx],  # Use the instance variable here
#                     "score": -distances[0][i]  # Use negative distance as score
#                 })
        
#         # Combine and sort results
#         results = sorted(results, key=lambda x: x["score"], reverse=True)
#         return results

#     def load_bert_model(self):
#         """Load the BERT model with GPU detection."""
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {device}")
#         return SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

#     def create_faiss_index(self):
#         """Create a FAISS index."""
#         dimension = 384  # Dimension of the MiniLM embeddings
#         return faiss.IndexFlatL2(dimension)

# def analyze_and_display(library_name: str, save_to_file: bool = True):
#     """Main function to analyze and display the results."""
#     analyzer = LibraryAnalyzer()
#     print(f"\nAnalyzing library: {library_name}")
    
#     analysis = analyzer.analyze_library(library_name)
    
#     if save_to_file:
#         output_dir = "metrics"
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         output_file = os.path.join(output_dir, f"{library_name}_analysis.json")
#         analyzer.save_analysis(analysis, output_file)
    
#     if 'error' in analysis:
#         print(f"\nError during analysis: {analysis['error']}")
#     else:
#         print("\nAnalysis Summary:")
#         print(f"Version: {analysis['metadata']['version']}")
#         print(f"Location: {analysis['metadata']['file']}")
#         print(f"Number of errors: {len(analysis['metadata']['analysis_errors'])}")
        
#         def count_elements(data, counts=None):
#             if counts is None:
#                 counts = {t.value: 0 for t in ElementType}
            
#             if isinstance(data, dict):
#                 if 'type' in data and isinstance(data['type'], str):
#                     counts[data['type']] = counts.get(data['type'], 0) + 1
#                 for value in data.values():
#                     if isinstance(value, (dict, list)):
#                         count_elements(value, counts)
#             elif isinstance(data, list):
#                 for item in data:
#                     count_elements(item, counts)
            
#             return counts
        
#         element_counts = count_elements(analysis)
#         print("\nElement counts:")
#         for element_type, count in element_counts.items():
#             if count > 0:
#                 print(f"- {element_type}: {count}")

#     # Extract text data for indexing
#     text_data = analyzer.extract_text_data(analysis)
#     # Index the extracted data
#     analyzer.index_data(text_data)
#     # Perform a sample search
#     search_query = "example search query"
#     search_results = analyzer.search(search_query)
#     print(f"\nSearch results for query '{search_query}':")
#     for result in search_results:
#         print(f"- Path: {result['path']}, Text: {result['text']}")

#     return analysis

# def explore_module(obj, path="", depth=0, max_depth=5, explored=None):
#     if explored is None:
#         explored = set()
    
#     # Avoid infinite loops
#     obj_id = id(obj)
#     if obj_id in explored or depth > max_depth:
#         return []
    
#     explored.add(obj_id)
#     endpoints = []
    
#     try:
#         # Explore the object's attributes
#         for name in dir(obj):
#             if name.startswith('_'):
#                 continue
                
#             try:
#                 attr = getattr(obj, name)
#             except:
#                 continue
                
#             current_path = f"{path}.{name}" if path else name
            
#             # Check if it's a method that could be an endpoint
#             if inspect.ismethod(attr) or inspect.isfunction(attr):
#                 if any(keyword in name.lower() for keyword in ['create', 'list', 'get', 'delete', 'update', 'retrieve']):
#                     sig = inspect.signature(attr)
#                     endpoints.append({
#                         'path': current_path,
#                         'type': 'method',
#                         'parameters': str(sig)
#                     })
            
#             # Recursively explore classes and modules
#             elif inspect.isclass(attr) or inspect.ismodule(attr):
#                 endpoints.extend(explore_module(attr, current_path, depth + 1, max_depth, explored))
                
#     except Exception as e:
#         pass
        
#     return endpoints

# def parse_json_file(file_path):
#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}")
#         return {}
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data

# def extract_function_signatures(data):
#     signatures = {}

#     def extract_from_members(members):
#         for name, info in members.items():
#             if 'parameters' in info:
#                 signatures[name] = info['parameters']
#             if 'members' in info:
#                 extract_from_members(info['members'])

#     if 'members' in data:
#         extract_from_members(data['members'])

#     return signatures
                
# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         library_name = sys.argv[1]
#         library_dict = analyze_and_display(library_name)
        
#         if len(sys.argv) > 2:
#             file_path = sys.argv[2]
#         else:
#             file_path = os.path.join("metrics", f"{library_name}_analysis.json")
        
#         data = parse_json_file(file_path)
#         if data:
#             print("Data loaded successfully.")
#             print(f"Data keys: {list(data.keys())}")
#             signatures = extract_function_signatures(data)
#             print(f"Extracted signatures: {signatures}")
#             for func_name, params in signatures.items():
#                 print(f"Function: {func_name}")
#                 for param_name, param_info in params.items():
#                     print(f"  Param: {param_name}")
#                     print(f"    Kind: {param_info['kind']}")
#                     print(f"    Default: {param_info['default']}")
#                     print(f"    Annotation: {param_info['annotation']}")
#         else:
#             print("No data found.")
#     else:
#         print("Please provide a library name as argument")
#         print("Example: python library_analyzer.py openai")
        
#     # python simulator\library_analyzer.py mistralai C:\metrics\mistralai_analysis_v1.2.3.json
#     # conda run python use_case.py
#     # docker run library-analyzer
