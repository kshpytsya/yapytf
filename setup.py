from setuptools import setup, find_packages

setup(
    name="yapytf",
    description="Yet Another Python Terraform Wrapper",
    long_description=open("README.md").read(),  # no "with..." will do for setup.py
    long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
    license="MIT",
    author="Kyrylo Shpytsya",
    author_email="kshpitsa@gmail.com",
    url="https://github.com/kshpytsya/yapytf",
    install_requires=[
        "click>=7.0,<8",
        "click-log>=0.3.2,<1",
        "implements>=0.1.4,<1",
        "jinja2>=2.10.1,<3",
        "jsonschema>=3.0.1,<4",
        "requests>=2.21.0,<3",
        "tqdm>=4.31.1,<5",
        "xdgappdirs>=1.4.5,<2",
    ],
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    python_requires=">=3.6, <3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": ["yapytf = yapytf._cli:main"]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: BSD :: FreeBSD",
        "Operating System :: POSIX :: BSD :: OpenBSD",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: System :: Systems Administration",
    ],
)
