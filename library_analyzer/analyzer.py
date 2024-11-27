import os
import json
import warnings
import importlib
import inspect
import asyncio
import contextlib
from datetime import datetime
from typing import Any, Dict, List
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
from .element import ElementType, get_element_type

class LibraryAnalyzer:
    """
    LibraryAnalyzer is a class designed to analyze Python libraries and extract detailed information about their elements, such as classes, functions, methods, and modules. It provides methods to evaluate types, extract type information, analyze signatures, and gather comprehensive details about classes and their members.
    Attributes:
        explored (set): A set to keep track of explored elements to avoid redundant analysis.
        errors (list): A list to store error messages encountered during analysis.
        current_path (list): A list to maintain the current path of elements being analyzed.
        library_elements (dict): A dictionary to store information about analyzed library elements.
        type_namespace (dict): A dictionary to store type information for safe evaluation.
        ix (index.Index): An instance of the Whoosh index.
        bert_model (SentenceTransformer): An instance of the BERT model.
        faiss_index (faiss.Index): An instance of the FAISS index.
    Methods:
        __init__(): Initializes the LibraryAnalyzer instance and sets up the type environment.
        _setup_type_environment(): Configures the environment for type management.
        safe_eval(type_str: str): Safely evaluates a type string.
        get_type_info(typ) -> str: Converts a type to a string representation safely.
        get_signature_info(obj) -> Dict: Extracts signature information from a function/method.
        get_class_info(obj) -> Dict: Extracts detailed information from a class.
        analyze_element(obj, name: str, module_name: str) -> Dict: Analyzes an individual library element.
        analyze_library(library_name: str) -> Dict: Performs a complete analysis of a library.
        save_analysis(analysis: Dict, output_file: str): Saves the analysis to a JSON file.
        extract_text_data(analysis: Dict) -> List[Dict]: Extracts relevant text data from the analysis results.
        index_data(data: List[Dict]): Indexes the extracted data using Whoosh and BERT.
        search(query: str, use_bert: bool = True, use_whoosh: bool = True, top_k: int = 3) -> List[Dict]: Searches the indexed data using Whoosh and BERT.
    """
    def __init__(self):
        self.explored = set()
        self.errors = []
        self.current_path = []
        self.library_elements = {}
        self._setup_type_environment()
        schema = Schema(path=ID(stored=True), text=TEXT(stored=True))
        if not os.path.exists("indexdir"):
            os.mkdir("indexdir")
        self.ix = create_in("indexdir", schema)
        self.bert_model = self.load_bert_model()
        self.faiss_index = self.create_faiss_index()
        self.documents = []
        self.load_indexed_data()

    def _setup_type_environment(self):
        """Configure the environment for type management."""
        # Add common types to the global namespace
        self.type_namespace = {
            'ClassVar': ClassVar,
            'TypeVar': TypeVar,
            'Generic': Generic,
            'Any': Any,
            'Union': Union,
            'List': List,
            'Dict': Dict,
            'Set': Set,
            'Optional': typing.Optional
        }

    def safe_eval(self, type_str: str):
        """Safely evaluate a type string."""
        try:
            with contextlib.suppress(Exception):
                return eval(type_str, self.type_namespace, {})
        except:
            return Any

    def get_type_info(self, typ) -> str:
        """Convert a type to a string representation safely."""
        try:
            if typ == inspect.Signature.empty:
                return 'Any'
            
            # Handle special forms like Union, Optional, and Any
            if isinstance(typ, typing._SpecialForm):
                if typ is typing.Any:
                    return 'Any'
                elif typ is typing.Optional:
                    args = typing.get_args(typ)
                    args_str = ', '.join(self.get_type_info(arg) for arg in args)
                    return f"Optional[{args_str}]"
                elif typ is typing.Union:
                    args = typing.get_args(typ)
                    args_str = ', '.join(self.get_type_info(arg) for arg in args)
                    return f"Union[{args_str}]"
                else:
                    return str(typ)
            
            # Handle generic types
            origin = typing.get_origin(typ)
            if origin is not None:
                args = typing.get_args(typ)
                args_str = ', '.join(self.get_type_info(arg) for arg in args)
                origin_name = getattr(origin, '__name__', str(origin))
                return f"{origin_name}[{args_str}]"
            
            # Handle simple types
            if isinstance(typ, type):
                return typ.__name__
            
            # Handle TypeVar
            if isinstance(typ, TypeVar):
                return f"TypeVar('{typ.__name__}')"
            
            # For all other types, fallback to string representation
            return str(typ)
        except Exception as e:
            self.errors.append(f"Error getting type info for {typ}: {str(e)}")
            return 'Any'

    def get_signature_info(self, obj) -> Dict:
        """Extract signature information from a function/method."""
        try:
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                # Skip built-in methods
                if obj.__name__ in dir(dict):
                    return {}
                sig = inspect.signature(obj)
                return {
                    'parameters': {
                        name: {
                            'kind': str(param.kind),
                            'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                            'annotation': self.get_type_info(param.annotation)
                        }
                        for name, param in sig.parameters.items()
                    },
                    'return_annotation': self.get_type_info(sig.return_annotation),
                    'docstring': inspect.getdoc(obj),
                    'is_async': asyncio.iscoroutinefunction(obj),
                    'is_generator': inspect.isgeneratorfunction(obj),
                }
        except Exception as e:
            self.errors.append(f"Error getting signature for {obj}: {str(e)}")
        return {}

    def get_class_info(self, obj) -> Dict:
        """Extract detailed information from a class."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                info = {
                    'bases': [self.get_type_info(base) for base in obj.__bases__],
                    'docstring': inspect.getdoc(obj),
                    'methods': {},
                    'properties': {},
                    'class_variables': {},
                    'instance_variables': {}
                }
                
                # Retrieve type annotations safely
                if hasattr(obj, '__annotations__'):
                    try:
                        hints = get_type_hints(obj, globalns=self.type_namespace)
                        info['type_hints'] = {
                            name: self.get_type_info(hint) 
                            for name, hint in hints.items()
                        }
                    except Exception as e:
                        info['type_hints'] = {
                            name: str(hint) 
                            for name, hint in obj.__annotations__.items()
                        }

                # For dataclasses
                if is_dataclass(obj):
                    info['dataclass_fields'] = {}
                    for field in fields(obj):
                        field_info = {
                            'type': self.get_type_info(field.type),
                            'default': str(field.default),
                            'metadata': field.metadata
                        }
                        info['dataclass_fields'][field.name] = field_info

                # For enums
                if issubclass(obj, Enum):
                    info['enum_values'] = {
                        name: str(value.value)
                        for name, value in obj.__members__.items()
                    }

                return info
        except Exception as e:
            self.errors.append(f"Error getting class info for {obj}: {str(e)}")
            return {}

    def analyze_element(self, obj, name: str, module_name: str) -> Dict:
        """Analyze an individual library element."""
        try:
            if id(obj) in self.explored:
                return {}
                
            self.explored.add(id(obj))
            current_path = '.'.join(self.current_path + [name])
            
            element_type = get_element_type(obj)
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

    def analyze_library(self, library_name: str) -> Dict:
        """Perform a complete analysis of a library."""
        try:
            # Import the library
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = importlib.import_module(library_name)

            # Get the current version of the library
            current_version = getattr(module, '__version__', 'Unknown')
            
            # Check if a previous analysis exists
            output_file = f"metrics/{library_name}_analysis.json"
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    previous_analysis = json.load(f)
                    previous_version = previous_analysis.get('metadata', {}).get('version', 'Unknown')
                    if previous_version == current_version:
                        renew = input(f"The library '{library_name}' version '{current_version}' has already been analyzed. Do you want to renew the analysis? (yes/no): ")
                        if renew.lower() != 'yes':
                            return previous_analysis
            
            # Reset state
            self.explored = set()
            self.errors = []
            self.current_path = []
            self.library_elements = {}
            
            # Analyze the main module
            library_info = self.analyze_element(module, library_name, library_name)
            
            # Add metadata information
            library_info['metadata'] = {
                'name': library_name,
                'version': current_version,
                'file': getattr(module, '__file__', 'Unknown'),
                'doc': inspect.getdoc(module),
                'analysis_errors': self.errors
            }
            
            return library_info
            
        except Exception as e:
            return {
                'error': f"Failed to analyze library {library_name}: {str(e)}",
                'analysis_errors': self.errors
            }

    def save_analysis(self, analysis: Dict, output_file: str):
        """Save the analysis to a JSON file."""
        def default_serializer(obj):
            """Handle non-serializable objects."""
            if isinstance(obj, (datetime,)):
                return obj.isoformat()
            return str(obj)
        
        try:
            base_name, ext = os.path.splitext(output_file)
            version = analysis.get('metadata', {}).get('version', 'Unknown')
            output_file = f"{base_name}_v{version}{ext}"
            
            counter = 1
            original_output_file = output_file
            while os.path.exists(output_file):
                output_file = f"{base_name}_v{version}_{counter}{ext}"
                counter += 1

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=default_serializer)
            print(f"Analysis saved to {output_file}")
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")

    def extract_text_data(self, analysis: Dict) -> List[Dict]:
        """Extract relevant text data from the analysis results."""
        text_data = []

        def extract_from_element(element, path=""):
            if isinstance(element, dict):
                if 'docstring' in element and element['docstring']:
                    text_data.append({
                        'path': path,
                        'text': element['docstring']
                    })
                if 'members' in element:
                    for name, member in element['members'].items():
                        extract_from_element(member, f"{path}.{name}" if path else name)

        extract_from_element(analysis)
        return text_data

    def index_data(self, data: List[Dict]):
        """Index the extracted data using Whoosh and BERT."""
        # Index data using Whoosh
        writer = self.ix.writer()
        for item in data:
            writer.add_document(path=item['path'], text=item['text'])
        writer.commit()

        # Index data using BERT and FAISS
        new_documents = [item['text'] for item in data]
        new_embeddings = self.bert_model.encode(new_documents)
        self.documents.extend(new_documents)
        self.faiss_index.add(new_embeddings)
        self.save_indexed_data()

    def search(self, query: str, use_bert: bool = True, use_whoosh: bool = True, top_k: int = 3) -> List[Dict]:
        """Search the indexed data using Whoosh and BERT."""
        results = []
        
        # Whoosh search
        if use_whoosh:
            with self.ix.searcher() as searcher:
                query_parser = QueryParser("text", self.ix.schema)
                whoosh_query = query_parser.parse(query)
                whoosh_results = searcher.search(whoosh_query, limit=top_k)
                results.extend([
                    {"source": "Whoosh", "path": hit["path"], "text": hit["text"], "score": hit.score}
                    for hit in whoosh_results
                ])
        
        # BERT search
        if use_bert:
            query_embedding = self.bert_model.encode([query])
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            for i, idx in enumerate(indices[0]):
                results.append({
                    "source": "BERT",
                    "path": "",  # BERT results do not have a path, so we leave it empty
                    "text": self.documents[idx],  # Use the instance variable here
                    "score": -distances[0][i]  # Use negative distance as score
                })
        
        # Combine and sort results
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return results

    def load_bert_model(self):
        """Load the BERT model with GPU detection."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        return SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    def create_faiss_index(self):
        """Create a FAISS index."""
        dimension = 384  # Dimension of the MiniLM embeddings
        return faiss.IndexFlatL2(dimension)

    def load_indexed_data(self):
        """Load indexed data from a file."""
        try:
            if os.path.exists("indexed_data.json"):
                with open("indexed_data.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data["documents"]
                    embeddings = np.array(data["embeddings"])
                    self.faiss_index.add(embeddings)
                    print("Indexed data loaded successfully.")
        except Exception as e:
            print(f"Error loading indexed data: {str(e)}")

    def save_indexed_data(self):
        """Save indexed data to a file."""
        try:
            embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
            data = {
                "documents": self.documents,
                "embeddings": embeddings.tolist()
            }
            with open("indexed_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("Indexed data saved successfully.")
        except Exception as e:
            print(f"Error saving indexed data: {str(e)}")
