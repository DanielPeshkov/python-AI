from setuptools import find_packages, setup

with open('README.md', 'r') as f:
	long_description = f.read()

setup(
	name='pythonai', 
	version='0.0.1', 
	description='A Python deep learning framework', 
	package_dir={'': 'src'}, 
	packages=find_packages(where='src'), 
	long_description=long_description, 
	long_description_content_type='text/markdown', 
	url='https://github.com/DanielPeshkov/python-AI', 
	author='DanielPeshkov', 
	license='MIT', 
	classifiers=[
		'License :: OSI Approved :: MIT License', 
		'Programming Language :: Python :: 3.10', 
		'Operating System :: OS Independent', 
	], 
	install_requires=['numpy>=1.24.1'], 
	extras_require={'dev': ['pytest>=7.0', 'pytest-cov>=4.1.0', 'twine>=4.0.2'],}, 
	python_requires='>=3.10', 
	)