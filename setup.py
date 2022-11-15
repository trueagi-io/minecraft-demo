from setuptools import setup
import sys

if sys.version_info < (3,10):
    sys.exit('Sorry, Python < 3.10 is not supported')

setup(name='tagilmo',
      version='0.1.0',
      packages=['tagilmo',
                'tagilmo.VereyaPython',
                'tagilmo.utils'],
      install_requires=['numpy']
)
