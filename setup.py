from setuptools import setup, find_packages

setup(
    name="game-content-generator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for generating game content using fine-tuned language models.",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "pandas",
        "numpy",
        "scikit-learn",
        "pytest"
    ],
    entry_points={
        "console_scripts": [
            "run-api=api.main:app"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)