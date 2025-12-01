#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2019-2025 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# Based on code in https://github.com/pybind/python_example
#

import os
import sys
from setuptools import setup
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.0.0"
comp_libcommute_versions = ">=1.0.0"

ext_modules = [
    Pybind11Extension("pycommute/expression", ["src/pycommute/expression.cpp"]),
    Pybind11Extension("pycommute/loperator", ["src/pycommute/loperator.cpp"])
]


class pycommute_build_ext(build_ext):
    """A custom build extension for adding libcommute-specific options."""

    def find_libcommute(self):
        from os import environ, path

        # 1. Try to find libcommute sources at $LIBCOMMUTE_INCLUDEDIR
        if 'LIBCOMMUTE_INCLUDEDIR' in environ:
            inc = environ['LIBCOMMUTE_INCLUDEDIR']
            version = self.detect_libcommute_version(inc)
            if version is not None:
                self.libcommute_includedir = inc
                return version

        # 2. Try to find the bundled sources
        inc = path.dirname(path.realpath(__file__))
        inc = path.join(inc, "src", "libcommute")
        if path.exists(path.join(inc, "libcommute", "version.hpp")):
            version = self.detect_libcommute_version(inc)
            if version is not None:
                self.libcommute_includedir = inc
                return version

        # 3. Let the C++ compiler find the headers
        version = self.detect_libcommute_version()
        self.libcommute_includedir = None
        return version

    def detect_libcommute_version(self, include_dir=None):
        import os
        import distutils.errors
        from tempfile import TemporaryDirectory
        from subprocess import run, PIPE

        test_prog = """
        #include <iostream>
        #include <libcommute/version.hpp>
        int main () { std::cout << LIBCOMMUTE_VERSION; return 0; }
        """

        with TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            include_dirs = [] if (include_dir is None) else [include_dir]
            open("libcommute_version.cpp", 'w').write(test_prog)
            try:
                objs = self.compiler.compile(
                    ["libcommute_version.cpp"], include_dirs=include_dirs
                )
                self.compiler.link_executable(
                    objs, "libcommute_version", target_lang='c++'
                )
                version = run(["./libcommute_version"], stdout=PIPE).stdout
                version = str(version, 'utf-8', 'ignore')
            except distutils.errors.CompileError:
                return None
            finally:
                os.chdir(cwd)
        return version

    def build_extensions(self):
        libcommute_version = self.find_libcommute()
        libcommute_loc = self.libcommute_includedir if \
            (self.libcommute_includedir is not None) else "a system location"

        if libcommute_version:
            print(f"Found libcommute version {libcommute_version} "
                  f"at {libcommute_loc}")
        else:
            raise RuntimeError(
                "Could not find libcommute headers. pycommute inspects the "
                "following locations while looking for the headers "
                "(in order):\n"
                "- Value of the environment variable $LIBCOMMUTE_INCLUDEDIR.\n"
                "- Subdirectory `src/libcommute` in pycommute's sources.\n"
                "- Standard system locations used by the C++ compiler."
            )

        if Version(libcommute_version) not in \
           SpecifierSet(comp_libcommute_versions):
            raise RuntimeError(
                f"Found an incompatible libcommute version {libcommute_version}"
                f" (required {comp_libcommute_versions})"
                f" at {libcommute_loc}."
            )

        for ext in self.extensions:
            if self.libcommute_includedir:
                ext.include_dirs.append(self.libcommute_includedir)
            ext.cxx_std = 17
            ext.extra_compile_args.append("-O3")
            if not sys.platform.startswith("darwin"):
                ext.extra_compile_args.append("-march=native")

        build_ext.build_extensions(self)


cmdclass = {'build_ext': pycommute_build_ext}


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pycommute",
    version=__version__,
    author="Igor Krivenko",
    author_email="iskrivenko@proton.me",
    description="Python bindings for the libcommute C++ library",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://krivenko.github.io/pycommute",
    keywords="libcommute dsl algebra quantum qubit",
    license="MPL-2.0",
    project_urls={
        "Bug Tracker": "https://github.com/krivenko/pycommute/issues",
        "Documentation": "https://krivenko.github.io/pycommute",
        "Source Code": "https://github.com/krivenko/pycommute"
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    python_requires=">=3.8",
    install_requires=["numpy>=1.12.0"],
    packages=['pycommute'],
    package_dir={'pycommute': 'src/pycommute'},
    ext_modules=ext_modules,
    include_package_data=True,
    cmdclass=cmdclass,
    zip_safe=False
)
