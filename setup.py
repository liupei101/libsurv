from setuptools import setup, find_packages

setup(name='libsurv',
    version='1.0.0',
    description='A library of efficient survival analysis models, including DeepCox, HitBoost and EfnBoost methods.',
    keywords = "survival analysis, deep learning, cox regression, XGBoost",
    url='https://github.com/liupei101/libsurv',
    author='Pei Liu',
    author_email='yuukilp@163.com',
    license='MIT',
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python 3",
        "Topic :: Scientific/Engineering",
    ],
    packages = find_packages(),
    install_requires=[
        'tensorflow>=1.10.0',
        'pandas>=0.24.2',
        'numpy>=1.14.5',
        'matplotlib>=3.0.3',
        'lifelines>=0.14.6',
        'xgboost>=0.82',
    ],
)