#
# This file is part of pycommute, Python bindings for the libcommute C++ library
#
# Copyright (C) 2016-2019 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# Based on code in https://github.com/pybind/python_example
#

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import setuptools
from os import path, environ
import os
import sys
import subprocess

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules_src = [
    ["expression", "_expression"],
    ["expression", "factories", "_factories"],
    ["expression", "factories", "real"],
    ["expression", "factories", "cmplx"],
    ["qoperator"]
]

ext_modules = [
    Extension(
        path.join("pycommute", *s),
        [path.join("pycommute", *s) + ".cpp"],
        include_dirs = [
            get_pybind_include(),
            get_pybind_include(user = True)
        ],
        language = 'c++'
    )
    for s in ext_modules_src
]

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        open("has_flag.cpp", 'w').write(
            "int main (int argc, char **argv) { return 0; }"
        )
        try:
            os.chdir(d)
            compiler.compile(["has_flag.cpp"], extra_postargs = [flagname])
        except setuptools.distutils.errors.CompileError:
            return False
        finally:
            os.chdir(cwd)

    return True

def cpp17_flag(compiler):
    f = {'unix' : '-std=c++17', 'msvc' : '/std:c++17'}[compiler.compiler_type]
    if has_flag(compiler, f):
        return f
    else:
        raise RuntimeError(
          'Unsupported compiler -- at least C++17 support is needed!'
        )

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    user_options = [
        ('libcommute-dir=', None, 'path to libcommute include directory')
    ]

    c_opts = {'msvc': ['/EHsc /O2'], 'unix': ["-O3"]}
    l_opts = {'msvc': [], 'unix': []}

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.libcommute_dir = ''

    def finalize_options(self):
        build_ext.finalize_options(self)

    def find_libcommute(self, compiler_opts):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            self.compiler.announce("xxx")
            open("libcommute_version.cpp", 'w').write("""
                  #include <iostream>
                  #include <libcommute/version.hpp>
                  int main (int argc, char **argv) {
                    std::cout << LIBCOMMUTE_VERSION;
                   return 0;
                  }
            """)
            try:
                inc = [self.libcommute_dir] if self.libcommute_dir else []
                objs = self.compiler.compile(["libcommute_version.cpp"],
                                             include_dirs = inc,
                                             extra_postargs = compiler_opts)
                self.compiler.link_executable(objs,
                                              "libcommute_version",
                                              target_lang = 'c++')
                v = str(subprocess.run(["./libcommute_version"],
                                       stdout = subprocess.PIPE).stdout)
            except setuptools.distutils.errors.CompileError:
                return False
            finally:
                os.chdir(cwd)

        return v

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.append(cpp17_flag(self.compiler))

        link_opts = self.l_opts.get(ct, [])

        libcommute_version = self.find_libcommute(opts)
        if libcommute_version:
            self.compiler.announce("Found libcommute version" +
                                   libcommute_version)
        else:
            raise RuntimeError(
                "Could not find libcommute headers. "
                "Use the --libcommute-dir command line option "
                "to specify location of libcommute include directory."
            )

        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.include_dirs.append(self.libcommute_dir)
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name = "pycommute",
    version = "0.1",
    author = "Igor Krivenko",
    author_email = "igor.s.krivenko@gmail.com",
    description = "Python bindings for the libcommute C++ library",
    keywords = "libcommute dsl computer algebra quantum",
    license = "MPL-2",
    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    python_requires = ">=3.5",
    install_requires = ['pybind11>=2.3'],
    setup_requires = ['pybind11>=2.3'],
    packages = find_packages(exclude = ["*.cpp"]),
    include_package_data = True,
    ext_modules = ext_modules,
    cmdclass = {'build_ext' : BuildExt}
)
