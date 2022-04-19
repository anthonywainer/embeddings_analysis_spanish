import os

from setuptools import setup

readme = open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding='latin-1')
setup(
    name="embeddings_analysis_spanish",
    version="1.0",
    author="Anthony Cachay",
    author_email='anthony.cachay@pucp.edu.pe',
    description="Embeddings Analysis Spanish",
    long_description=readme.read(),
    long_description_content_type="text/markdown",
    package_dir={
        '': 'src/main'
    },
    packages=['embeddings_analysis_spanish',
              'embeddings_analysis_spanish.cleaning',
              'embeddings_analysis_spanish.embedding',
              'embeddings_analysis_spanish.utils',
              ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Open Source"
    ],
    python_requires='>=3.7.x',
    install_requires=[
        "scikit-learn==0.22.2", "transformers==4.18.0"
    ]
)

readme.close()
