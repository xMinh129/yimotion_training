from setuptools import setup

setup(name='DetectionTest',
      version='0.0.1',
      description='Detection',
      author='Minh Vu',
      author_email='xuanminh12995@gmail.com',
      python_requires='>=2.7, <4',
      packages=['sql'],
      tests_require=['pytest==3.0.3', 'cachetools'],
      install_requires=['numpy',
                        'tensorflow',
                        'keras',
                        'opencv-python', 'h5py']
      )
