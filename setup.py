import codecs

from setuptools import find_packages
from setuptools import setup


requirements = [
    "numpy >= 1.21",
    "pandas >= 2.0.3",
    "tensorflow",
    "tensorflow-probability",
    "scikit-learn >= 1.3.0",
    "h5py >= 3.9.0",
    "rdkit",
    "deepchem >= 2.7.1",
]

with open("./test-requirements.txt") as test_reqs_txt:
    test_requirements = [line for line in test_reqs_txt]


long_description = ""
with codecs.open("./README.md", encoding="utf-8") as readme_md:
    long_description = readme_md.read()

setup(
    name="cdrpy",
    use_scm_version={"write_to": "cdrpy/_version.py"},
    description="Cancer drug response prediction toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csederman/cdrpy",
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    package_data={
        "cdrpy.data.resources.genelists": ["*.pkl"],
    },
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    tests_require=test_requirements,
    python_requires=">=3.8",
    zip_safe=False,
    test_suite="tests",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    maintainer="Casey Sederman",
    maintainer_email="casey.sederman@hsc.utah.edu",
)
