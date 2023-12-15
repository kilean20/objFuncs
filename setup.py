from setuptools import setup, find_packages

    
setup(name='objFuncs',
    version=0.0,
    description='object functions for FRIB beam tuning',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.6',
    'Topic :: FRIB beam tuning'
    ],
    keywords = ['machine-learning', 'optimization', 'FRIB beam tuning'],
    author='Kilean Hwang',
    author_email='hwang@frib.msu.edu',
#     license='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
    ],
    zip_safe=False)
