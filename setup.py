from setuptools import setup

setup(name='YimotionDetection',
      version='0.0.1',
      description='Detection',
      author='Minh Vu',
      author_email='minh@healthy.io',
      python_requires='>=2.7, <4',
      packages=['sql'],
      tests_require=['pytest==3.0.3', 'cachetools'],
      install_requires=['flask',
                        'Werkzeug',
                        'six',
                        'pymongo',
                        'pybase64', 'pillow',
                        'numpy',
                        'tensorflow', 'matplotlib',
                        'opencv-python',
                        'PyDrive',
                        'google-auth-oauthlib',
                        'keras',
                         'h5py'
                        ]
      )