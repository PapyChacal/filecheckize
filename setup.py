import re
from pathlib import Path
from typing import cast

from setuptools import find_packages, setup

import versioneer

# Add README.md as long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

git_regex = r"git\+(?P<url>https:\/\/github.com\/[\w]+\/[\w]+\.git)(@(?P<version>[\w]+))?(#egg=(?P<name>[\w]+))?"

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-optional.txt") as f:
    optionals = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == "git":
        name = ir.split("/")[-1]
        reqs += [f"{name} @ {ir}@main"]
    else:
        reqs += [ir]

extras_require = {}
for mreqs, mode in zip(
    [
        optionals,
    ],
    [
        "extras",
    ],
):
    opt_reqs = []
    for ir in mreqs:
        # For conditionals like pytest=2.1; python == 3.6
        if ";" in ir:
            entries = ir.split(";")
            extras_require[entries[1]] = entries[0]
        elif ir[0:3] == "git":
            m = re.match(git_regex, ir)
            assert m is not None
            items = m.groupdict()
            name = items["name"]
            url = items["url"]
            version = items.get("version")
            if version is None:
                version = "main"

            opt_reqs += [f"{name} @ git+{url}@{version}"]
        else:
            opt_reqs += [ir]
    extras_require[mode] = opt_reqs

setup(
    name="filecheckize",
    version=cast(str, versioneer.get_version()),
    cmdclass=versioneer.get_cmdclass(),
    description="FileCheck generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "filecheckize = filecheckize.main:main",
        ]
    },
    project_urls={
        "Source Code": "https://github.com/papychacal/filecheckize",
        "Issue Tracker": "https://github.com/papychacal/filecheckize/issues",
    },
    platforms=["Linux", "Mac OS-X", "Unix", "Windows"],
    test_suite="FileCheck",
    author="Emilien Bauer",
    author_email="emilien.bauer@ed.ac.uk",
    license="MIT",
    packages=find_packages(),
    package_data={"filcheckize": ["py.typed"]},
    install_requires=reqs,
    extras_require=extras_require,
    zip_safe=False,
)
