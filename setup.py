import os

from setuptools import setup

readme = open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding='latin-1')
requirements = open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r", encoding='latin-1')
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
              'embeddings_analysis_spanish.abstracts',
              'embeddings_analysis_spanish.cleaning',
              'embeddings_analysis_spanish.embeddings',
              'embeddings_analysis_spanish.evaluation',
              'embeddings_analysis_spanish.modeling',
              'embeddings_analysis_spanish.models',
              'embeddings_analysis_spanish.utils'
              ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Open Source with education focus"
    ],
    python_requires='>=3.7.x',
    install_requires=requirements.readlines()
)

requirements.close()
readme.close()
