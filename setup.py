# from setuptools import find_packages
from setuptools import setup


setup(
    name='ml_examples',
    version='0.0.0',
    author='AndreyNikitin',
    author_email='nikitinandrews@gmail.com',
    # packages=find_packages(exclude=['developer_instruments', 'examples', '*dnn_quant_tests*', 'legacy_developer_docs']),
    package_data={
        'ml_examples': [],
    },
    install_requires=[
        'numpy==1.22.3',
        'keras==2.7',
        'tensorflow==2.7.1',  # to compute quant-graph reference (also for docs)

        'pytest',

        # Code Quality
        'flake8-colors',  # ANSI colors highlight for flake8
        'flake8-polyfill',  # flake8 compatibility helper
        'flake8>=3.9',  # PEP8 compliance
        'mypy==0.950',  # static analysis
        'pylint',  # provisional error checker
        'radon',  # computes complexity code metrics
        'xenon',  # fails if any of complexity requirements is not met
    ],
)
