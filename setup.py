
from setuptools import setup, find_packages

setup(
    name='library_analyzer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'whoosh',
        'sentence-transformers',
        'torch',
        'faiss-cpu',
        'numpy',
        'pyyaml'
    ],
    entry_points={
        'console_scripts': [
            'library-analyzer=library_analyzer.__main__:main',
        ],
    },
)