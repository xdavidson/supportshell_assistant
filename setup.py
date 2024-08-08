from setuptools import setup, find_packages

setup(
    name='supportshell_assistant',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pymilvus',
        'sentence-transformers',
        'langchain',
        'transformers'
    ],
    entry_points={
        'console_scripts': [
            'supportshell_assistant=supportshell_assistant.main:main',
        ],
    },
)
