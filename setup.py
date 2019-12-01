import soydata
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="soydata",
    version=soydata.__version__,
    author=soydata.__author__,
    author_email='soy.lovit@gmail.com',
    url='https://github.com/lovit/synthetic_dataset',
    description="Synthetic data generator for machine learning",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=["numpy>=1.17.0", "bokeh>=1.4.0", "plotly>=4.3.0", "scikit-learn>=0.21.3"],
    keywords = ['Data generator'],
    packages=find_packages()
)
