from setuptools import setup, find_packages

setup(name='matplotlib_surface_plotting',
      version='0.1',
      packages=find_packages(),
      install_requires=['nibabel',
                        'matplotlib>=3.3.2'],
      package_dir={'matplotlib_surface_plotting':'matplotlib_surface_plotting'},
      url="https://github.com/kwagstyl/matplotlib_surface_plotting"
     )