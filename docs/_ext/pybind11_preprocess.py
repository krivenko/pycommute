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
            s = s.replace("*args", "\*args").replace("**kwargs", "\*\*kwargs")
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
