## Overview

The script `library_analyzer.py` in the [JacquesGariepy/library-analyzer](https://github.com/JacquesGariepy/library-analyzer) repository is designed to analyze Python libraries and extract detailed information about their elements, such as classes, methods, functions, properties, and more. The analysis results can be saved to a JSON file for further inspection.

### Capabilities of the Code:
- **Analyze Python Libraries**: The script can analyze Python libraries and extract detailed information about various elements within the library.
- **Element Types Identified**: It identifies and categorizes elements such as classes, methods, functions, properties, modules, variables, enums, constants, dataclasses, coroutines, generators, descriptors, exceptions, and protocols.
- **Extract Type Information**: The script can safely evaluate and extract type information for various elements.
- **Extract Signatures**: It can extract function/method signatures and other relevant details such as docstrings, parameter types, and return types.
- **Class Analysis**: The script provides detailed information about classes, including base classes, methods, properties, and type hints.
- **Dataclass and Enum Analysis**: It can analyze dataclasses and enums, extracting field types and enum values.
- **Save Analysis Results**: The analysis results can be saved to a JSON file for further inspection and documentation.
- **Error Handling**: The script includes error handling to capture and log errors encountered during the analysis process.
- **Semantic Search**: The script includes semantic search functionality using Whoosh and BERT to index and search extracted text data from the analysis results.

### Tools and Technologies Used:
- **Python**: The primary programming language used for the script.
- **Whoosh**: A fast, featureful full-text indexing and searching library implemented in pure Python.
- **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer-based machine learning technique for natural language processing pre-training.
- paraphrase-MiniLM-L6-v2: A pre-trained model for paraphrase identification.
- **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors.
- **Sentence Transformers**: A Python framework for state-of-the-art sentence, text, and image embeddings.
- **PyYAML**: A YAML parser and emitter for Python.
- **Unittest**: Python's built-in module for testing code.
- **Docker**: A platform for developing, shipping, and running applications in containers.

This script is part of a larger ongoing project aimed at creating a comprehensive tool for analyzing and documenting Python libraries. The project aims to provide insights into the structure and content of libraries, aiding developers in understanding and utilizing various libraries efficiently.

## Usage

To use the script, run it from the command line with the name of the library to analyze:

```sh
python -m library_analyzer <library_name> [search_query]
```

## Using Docker

You can also use Docker to run the script. First, build the Docker image:

```sh
docker build -t library-analyzer .
```

Then, run the container with the necessary arguments:

```sh
docker run --rm -v $(pwd):/app library-analyzer <library_name> [search_query]
```

## Testing

For run the tests, you can use the following command:

```sh
python -m unittest discover -s tests
```

## Output
The analysis results include metadata about the library, such as its name, version, file location, and documentation, as well as detailed information about each element in the library. The results can be saved to a JSON file for further inspection. The filename includes the library version and increments if the file already exists. Look for files like `openai_analysis_v1.0.json` or `openai_analysis_v1.0_1.json` for the analysis results of the openai library.

(json output)
![1732065708571](https://github.com/user-attachments/assets/f384f7e2-be33-4353-a813-191d162a9036)

## Semantic Search

The script now includes semantic search functionality using Whoosh and BERT. This allows you to perform searches on the extracted text data from the analysis results, such as docstrings, function signatures, and class descriptions.

### How to Use Semantic Search

1. **Extract Text Data for Indexing**: The script extracts relevant text data from the analysis results, including docstrings, function signatures, and class descriptions.
2. **Index the Extracted Data**: The extracted text data is indexed using Whoosh and BERT.
3. **Perform Searches**: You can perform searches on the indexed data using the search function in the `LibraryAnalyzer` class.

### Example

To perform a search, you can use the following code snippet:

```python
analyzer = LibraryAnalyzer()
analysis = analyzer.analyze_library("your_library_name")
text_data = analyzer.extract_text_data(analysis)
analyzer.index_data(text_data)
search_results = analyzer.search("your_search_query")
for result in search_results:
    print(f"Path: {result['path']}, Text: {result['text']}")
```

This will output the search results, showing the paths and text snippets that match the search query.

## Configuration

The script uses a configuration file `config.yaml` to define preferences for the semantic search functionality. The configuration options include enabling or disabling BERT and Whoosh, and setting the top_k value for the number of search results to return.

### Example Configuration

Here is an example `config.yaml` file:

```yaml
use_bert: true
use_whoosh: true
top_k: 5
```

## Conclusion
The library_analyzer.py script is a powerful tool for analyzing Python libraries and extracting detailed information about their elements. It can be used to gain insights into the structure and contents of a library, making it easier to understand and work with.

[![Reddit Badge](https://img.shields.io/badge/Discussion-reddit-red)](https://www.reddit.com/r/Python/comments/1gx9j3t/library_analyzer_python_libraries_and_extract/)

Let’s stay in touch here or on LinkedIn.
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/jacquesgariepy)
