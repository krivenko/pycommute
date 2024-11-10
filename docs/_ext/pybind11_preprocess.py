#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2024 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
A simple Sphinx extension that applies extra preprocessing to pybind11-generated
docstrings.
"""

from sphinx.util.docutils import SphinxDirective


class PyBind11Class(SphinxDirective):

    has_content = True
    registered_classes = []

    def run(self):
        self.registered_classes.append(self.content[0])
        return []


def autodoc_process_docstring(app, what, name, obj, options, lines):
    # Escape the asterisks in "*args" and "**kwrags"
    if what in ("function", "method"):
        for n, s in enumerate(lines):
            s = s.replace("*args", r"\*args").replace("**kwargs", r"\*\*kwargs")
            lines[n] = s

    # Turn pybind11 class names into :py:class: roles.
    for n, s in enumerate(lines):
        for c in PyBind11Class.registered_classes:
            s = s.replace(" " + c, " :py:class:`" + c + "`")
            s = s.replace("[" + c, "[:py:class:`" + c + "`")
        lines[n] = s


def setup(app):
    app.add_directive("pybind11class", PyBind11Class)

    app.connect('autodoc-process-docstring', autodoc_process_docstring)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True
    }
