from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="herald",
    version="0.1.0",
    author="Athiyo Chakma",
    author_email="athiyo22118@iiitd.ac.in",
    description="API-free phishing domain detection for critical infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/athiyo/herald",
    packages=find_packages(exclude=["tests", "tests.*", "ml", "scripts", "dashboard"]),
    python_requires=">=3.12",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Topic :: Internet :: WWW/HTTP",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Developers",
    ],
    entry_points={
        "console_scripts": [
            "herald=herald.main_pipeline:main",
        ],
    },
)
