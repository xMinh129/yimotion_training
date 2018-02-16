from setuptools import setup

setup(name='Cascade Test',
      version='0.0.1',
      description='The Haar Cascade Face And Eye Detection',
      url='https://github.com/yimotion/face-detection',
      author='Minh Vu',
      author_email='xuanminh12995@gmail.com',
      python_requires='>=2.7, <4',
      packages=['sql'],
      tests_require=['pytest==3.0.3', 'cachetools'],
      install_requires=['opencv-python', 'numpy',
                        'matplotlib', 'requests', 'Pillow'
                        ]
      )