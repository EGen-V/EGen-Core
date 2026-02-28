import sys
import setuptools
from setuptools.command.install import install
import subprocess

# upgrade transformers to latest version
class PostInstallCommand(install):
    def run(self):
        install.run(self)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
        except subprocess.CalledProcessError:
            print("Warning: Unable to upgrade transformers package. Please upgrade manually.")

# Windows uses a different default encoding (use a consistent encoding)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EGen-Core",
    version="1.0.3",
    author="ErebusTN",
    author_email="mouhebza0@gmail.com",
    description="EGen-Core allows single GPUs to run large language models with memory-efficient layer-wise inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ErebusTN/EGen-Core",
    packages=setuptools.find_packages(),
    install_requires=[
        'tqdm',
        'torch',
        'transformers',
        'accelerate',
        'safetensors',
        'optimum',
        'huggingface-hub',
        'scipy',
        # 'bitsandbytes' optional
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
