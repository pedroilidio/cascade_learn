from setuptools import setup, find_packages

setup(
    name='cascade_learn',
    version='0.0.1.dev0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'joblib',
        'scikit-learn',
        'imbalanced-learn',
    ],
)
