"""
Setup script for sentiment analysis package.
Allows installation as a package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentiment-analysis-engine",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fear & Greed Sentiment Analysis Engine for Crypto Markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sentiment-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sentiment-engine=main:main",
        ],
    },
)
