************************
Standard Library Support
************************

CrossHair reasons symbolically about the built-in types and much of the standard
library. The treemap below shows, for each operation, how well CrossHair can reason
*backward* through it -- given a desired output, how readily it can find inputs that
produce it (the core thing CrossHair does when it searches for a counterexample) --
**sized by how widely that operation is actually used** across the top 1000 PyPI
packages, so the more heavily-used operations get the bigger boxes.

.. raw:: html
   :file: support_treemap.svg

.. only:: not html

   .. image:: support_treemap.svg
      :width: 100%
      :alt: A treemap of operations sized by real-world usage and colored by how
            well CrossHair can reason about each.

Each box is one operation, nested under its builtin type / ``builtins`` functions /
standard-library module.  Box **area** grows with the number of top-PyPI packages
that use the operation (compressed so heavily-used operations don't crowd out the
rest; operations used by under a package are dropped entirely).  Box **color** is
**measured automatically**, not hand-maintained: CrossHair fuzzes a real input, runs
the operation forward to a concrete output, then asks itself to find inputs that
produce that output:

* **green** -- CrossHair handles it well: it readily finds inputs for a given
  output.
* **yellow** -- it works for small inputs but gets slow as inputs grow.
* **red** -- CrossHair struggles to reason backward through this one. (Common for
  transforms like encoding/compression, transcendental float math, and
  character-table lookups.)
* **black** -- CrossHair gives a *wrong* answer here: it confidently reports that no
  input produces the output, even though we just ran a real input that does. These
  are soundness bugs worth fixing.
* **grey** -- not measured: there is no drivable call (e.g. the operation needs a
  callback, or returns something that can't be compared by value).

Whether a cell is green or red reflects a *combination* of what the underlying SMT
solver can do and how CrossHair chooses to model the operation -- so a red cell may
turn green as CrossHair's modeling improves, not only as solvers do.

A red or grey cell does not mean CrossHair fails on your code -- it means CrossHair
is unlikely to *work backward* through that specific operation while searching for a
counterexample. Running your code forward still works.

Hover any box for a plain description, or click it to run that operation live in
your browser.

Regenerating the map
====================

The **support colors** are re-measured each release (slow, so parallelized with
``--jobs``), then the treemap is re-rendered against the checked-in usage prior::

    python -m crosshair.tools.measure_support surface --jobs 8 --json surface.json
    python -m crosshair.tools.measure_support funcs   --jobs 8 --json funcs.json
    python -m crosshair.tools.generate_treemap \
        --measured surface.json,funcs.json --weights doc/source/usage_prior.json \
        --scale sqrt --out doc/source/support_treemap.svg

The **usage prior** (``doc/source/usage_prior.json``) changes slowly, so it is
checked in and only refreshed occasionally.  Refreshing it only **downloads and
statically parses** package source -- wheels are unzipped and analyzed with
``ast``; no downloaded code is ever imported or run::

    python -m crosshair.tools.fetch_corpus --n 1000 --out corpus
    python -m crosshair.tools.mine_usage \
        --measured surface.json,funcs.json --corpus corpus \
        --out doc/source/usage_prior.json

You can also try any operation live in your browser at `crosshair-web.org`_.

.. _crosshair-web.org: https://crosshair-web.org
