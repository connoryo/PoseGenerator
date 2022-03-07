"""
Program to overlay skeleton over predetermined locations in video.

"""

from setuptools import setup

setup(
    name='posegenerator',
    version='0.1.0',
    packages=['posegenerator'],
    include_package_data=True,
    install_requires=[
        'click',
        'numpy',
        'opencv-python'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'posegenerator = posegenerator.__main__:main'
        ]
    },
)
