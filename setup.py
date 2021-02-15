from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A Python application that runs Coral object detection behind an HTTP server. This is meant to allow interaction with Coral image processing in the same manner as with Deepstack, using POST requests.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="coral-detection-service",
    version=VERSION,
    author="Marcin Morys",
    author_email="<marcin.m.morys@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    entry_points={'console_scripts': ['coral-serve = coral_detection_service.coral_detection_launcher:coral_serve']},
    python_requires='>=3.5',
    install_requires=['numpy',
                      'pillow'],
    extras_require={'coral': ['pycoral']},
    package_data={'coral_detection_service': ['models/*.txt', 'models/*.tflite']},
    include_package_data=True,
    keywords=['python', 'coral']
)