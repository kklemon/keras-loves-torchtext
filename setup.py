from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf8') as fp:
    long_description = fp.read()

setup(
    name='keras-loves-torchtext',
    version='0.0.1',
    author='Kristian Klemon',
    author_email='kristian.klemon@gmail.com',
    description='Make torchtext work with Keras',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kklemon/keras-loves-torchtext",
    license='BSD',
    packages=find_packages(exclude=('test', 'examples')),
    zip_safe=True
)
