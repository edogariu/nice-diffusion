from setuptools import setup

setup(
    name='nicediffuson',
    version='0.1.0',
    author='Evan Dogariu',
    author_email='eadogariu@aol.com / edogariu@princeton.edu',
    packages=['nicediffusion'],
    scripts=['scripts/sample.py', 'scripts/train.py'],
    url='http://pypi.python.org/pypi/nicediffusion/',
    license='LICENSE.txt',
    description='Package for a nice diffusion model',
    long_description=open('README.md').read(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'tqdm',
        'matplotlib', # for visualization only
        'basicsr', # for upsampling only --- comment out to avoid LOTS of requirements
    ],
)
