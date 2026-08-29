from glob import glob

from setuptools import setup

package_name = "so101_icl_bridge"

setup(
    name=package_name,
    version="0.1.0",
    python_requires=">=3.12",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dmitri Manajev",
    maintainer_email="dmitri@manajev.com",
    description="ROS 2 shell for the ICL bridge node (dispatch packages to demo packs)",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "bridge_icl_node = so101_icl_bridge.bridge:main",
        ],
    },
)
