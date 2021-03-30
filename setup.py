#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2019-2021 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# Based on code in https://github.com/pybind/python_example
#

import os
from setuptools import setup
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.6.1"
comp_libcommute_versions = ">=0.6.1"

ext_modules = [
    Pybind11Extension("pycommute/expression", ["pycommute/expression.cpp"]),
    Pybind11Extension("pycommute/loperator", ["pycommute/loperator.cpp"])
]


class pycommute_build_ext(build_ext):
    """A custom build extension for adding libcommute-specific options."""

    def find_libcommute(self):
        import os
        import tempfile
        import subprocess
        import distutils.errors
        from os import environ

        self.libcommute_includedir = environ.get('LIBCOMMUTE_INCLUDEDIR', None)
        inc = [self.libcommute_includedir] if self.libcommute_includedir else []

        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            open("libcommute_version.cpp", 'w').write("""
                  #include <iostream>
                  #include <libcommute/version.hpp>
                  int main () { std::cout << LIBCOMMUTE_VERSION; return 0; }
            """)
            try:
                objs = self.compiler.compile(["libcommute_version.cpp"],
                                             include_dirs=inc)
                self.compiler.link_executable(objs,
                                              "libcommute_version",
                                              target_lang='c++')
                v = subprocess.run(["./libcommute_version"],
                                   stdout=subprocess.PIPE).stdout
                v = str(v, 'utf-8', 'ignore')
            except distutils.errors.CompileError:
                return False
            finally:
                os.chdir(cwd)

        return v

    def build_extensions(self):
        libcommute_version = self.find_libcommute()
        if libcommute_version:
            print("Found libcommute version " + libcommute_version)
        else:
            raise RuntimeError(
                "Could not find libcommute headers. "
                "Use the LIBCOMMUTE_INCLUDEDIR environment variable "
                "to specify location of libcommute include directory."
            )

        if Version(libcommute_version) not in \
           SpecifierSet(comp_libcommute_versions):
            raise RuntimeError(
                "Incompatible libcommute version %s (required %s)." % (
                    libcommute_version, comp_libcommute_versions
                )
            )

        for ext in self.extensions:
            ext.include_dirs.append(self.libcommute_includedir)
            ext.cxx_std = 17

        build_ext.build_extensions(self)


cmdclass = {'build_ext': pycommute_build_ext}

# Detect Sphinx/sphinx_rtd_theme and enable documentation build
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    BuildDoc = None
if BuildDoc:
    class run_build_before_doc(BuildDoc):
        def run(self):
            self.run_command("build")
            super(run_build_before_doc, self).run()

    cmdclass['build_sphinx'] = run_build_before_doc
else:
    print("Note: "
          "Install Sphinx>=2.0.0 and sphinx_rtd_theme to build documentation")


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pycommute",
    version=__version__,
    author="Igor Krivenko",
    author_email="igor.s.krivenko@gmail.com",
    description="Python bindings for the libcommute C++ library",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://krivenko.github.io/pycommute",
    keywords="libcommute dsl computer algebra quantum",
    license="MPL-2",
    project_urls={
        "Bug Tracker": "https://github.com/krivenko/pycommute/issues",
        "Documentation": "https://krivenko.github.io/pycommute",
        "Source Code": "https://github.com/krivenko/pycommute"
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    python_requires=">=3.6",
    install_requires=["numpy>=1.12.0"],
    packages=['pycommute'],
    ext_modules=ext_modules,
    include_package_data=True,
    cmdclass=cmdclass,
    zip_safe=False
)
