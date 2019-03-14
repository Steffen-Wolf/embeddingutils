"""
embeddingutils - utilities for learning instance embeddings
"""

import setuptools


setuptools.setup(
    name="embeddingutils",
    author="Steffen Wolf, Roman Remme",
    author_email="steffen.wolf@iwr.uni-heidelberg.de, roman.remme@iwr.uni-heidelberg.de",
    description="utilities for learning instance embeddings",
    version="0.1",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "torch",
    ],
    #license="Apache Software License 2.0",  # TODO: add a license
    packages=setuptools.find_packages(),
)