import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mue",
    version="0.0.1",
    author="Eli Weinstein",
    author_email="eweinstein@g.harvard.edu",
    description="A package for building MuE models in Edward2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debbiemarkslab/MuE",
    packages=setuptools.find_packages(),
    install_requires=['tensorflow>=2.2.0',
                      'tensorflow-probability>=0.10.0',
                      'numpy>=1.18.1',
                      'scipy>=1.4.1',
                      'biopython>=1.77'],
    extras_require={
        'extras': ['logomaker>=0.8',
                   'dill>=0.3.1.1',
                   'matplotlib>=3.1.3',
                   'pandas>=1.1.0',
                   'scikit-learn>=0.23.1',
                   'pytest>=5.4.3'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        'Topic :: Scientific/Engineering :: Computational Biology',
    ],
    python_requires='>=3.7',
    keywords=('biological sequences proteins probabilistic programming ' +
              'tensorflow machine learning'),
)
