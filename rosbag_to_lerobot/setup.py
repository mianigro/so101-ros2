from setuptools import find_packages, setup

package_name = "rosbag_to_lerobot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            [
                "config/so101_30hz.yaml",
                "config/so101_50hz.yaml",
            ],
        ),
    ],
    install_requires=[
        "setuptools",
        "numpy>=2.0,<2.3",
        "pyyaml>=6.0.3,<7",
        "imageio>=2.34,<3",  # for CompressedImage decoding
        "lerobot[dataset]==0.6.1",
    ],
    zip_safe=True,
    maintainer="Dmitri Manajev",
    maintainer_email="dmitri@manajev.com",
    description="Convert ROS 2 bag files (MCAP) to LeRobot datasets",
    license="Apache-2.0",
    tests_require=["pytest"],
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "convert = rosbag_to_lerobot.cli:main",
        ],
    },
)
