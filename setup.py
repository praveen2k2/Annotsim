from setuptools import setup, find_packages

setup(
    name="Annotsim",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A repository for diffusion-based annotation and simulation models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MAXNORM8650/Annotsim",  
    packages=find_packages(where="src"),  # Finds all packages in src/
    package_dir={"": "src"},  # Root of the package is the src/ directory
    include_package_data=True,  # Include non-Python files (e.g., assets)
    install_requires=[
        "torch>=1.11.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.3",
        "scikit-learn>=0.24.0",
    ],  # Add dependencies here
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
