from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autorace_core_ROSticsv8'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vadik',
    maintainer_email='vadimperminov08@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'ros_competition_node = autorace_core_ROSticsv8.ros_competition_node:main',
        'task_1_node = autorace_core_ROSticsv8.task_1_node:main',
        'task_2_node = autorace_core_ROSticsv8.task_2_node:main',
        'task_3_node = autorace_core_ROSticsv8.task_3_node:main',
        'yolo_node = autorace_core_ROSticsv8.yolo_node:main',
        'drive_node = autorace_core_ROSticsv8.drive_node:main',
        ],
    },
)
