from setuptools import setup, find_packages

setup(
    packages=["gui", "gui.api", "gui.widgets", "gui.workers"],
    package_dir={"gui": "."},
)
