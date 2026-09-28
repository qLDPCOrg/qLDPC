# qLDPC

`qLDPC` is a Python library for constructing and analyzing finite-size [quantum low-density parity-check codes](https://errorcorrectionzoo.org/c/qldpc), as well as stabilizer and subsystem codes.
You can use it to build codes, find logical operators and distances, decode errors, and create [`stim`](https://github.com/quantumlib/Stim)-based circuits.

Most code-level tools work over arbitrary finite fields.
Circuit tools currently support qubit codes only.

## Installation

qLDPC requires Python 3.10 or later:

```bash
python -m pip install qldpc
```

Install the optional Relay-BP decoder with:

```bash
python -m pip install 'qldpc[relay-bp]'
```

For development:

```bash
git clone https://github.com/qLDPCOrg/qLDPC.git
cd qLDPC
python -m pip install -e '.[dev]'
```

Some algebra tools, code lookups, and distance estimates require [GAP](https://www.gap-system.org).
If you use Conda on Linux or macOS, install it with `conda install -c conda-forge gap`; other installations work when `gap` is available on `PATH`.
GAP integration on Windows remains limited; see [issue #294](https://github.com/qLDPCOrg/qLDPC/issues/294).

If installing `cvxpy` fails on macOS, follow its [platform-specific installation guidance](https://www.cvxpy.org/install) before installing qLDPC.

## Quickstart

```python
from sympy.abc import x, y

from qldpc import codes

# The [[144, 12, 12]] bivariate bicycle "gross code".
code = codes.BBCode({x: 12, y: 6}, x**3 + y + y**2, y**3 + x + x**2)

print(code)
print("physical qubits:", len(code))
print("logical qubits:", code.dimension)
```

Continue with the [library map](https://qldpc.readthedocs.io/en/latest/library_map.html) for the data model and package structure, or open the [example notebooks](https://qldpc.readthedocs.io/en/latest/examples/index.html) for complete workflows.

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Understand code representations and how packages fit together | [Library map](https://qldpc.readthedocs.io/en/latest/library_map.html) |
| Learn by running complete workflows | [Examples](https://qldpc.readthedocs.io/en/latest/examples/index.html) |
| Browse code families, functions, and exact signatures | [API reference](https://qldpc.readthedocs.io/en/latest/autoapi/index.html) |
| Construct and analyze a first code | [qLDPC basics](https://qldpc.readthedocs.io/en/latest/examples/basics.html) |
| Estimate logical error rates | [Logical-error-rate examples](https://qldpc.readthedocs.io/en/latest/examples/index.html#logical-error-rates) |
| Build memory circuits or noise models | [Circuit examples](https://qldpc.readthedocs.io/en/latest/examples/index.html) |
| Change or extend qLDPC safely | [Agent and contributor guide](AGENTS.md) |

For the complete list of classes and functions, including construction-specific literature, use the API reference and source docstrings.
This README focuses on getting started and finding the right documentation.

## Limitations and caveats

- Circuit and tableau tools only work with qubit codes, even though other parts of qLDPC support prime-power-dimensional qudits.
- Circuits returned by `get_encoding_circuit` are not fault-tolerant.
  Fault-tolerant encoding is tracked in [issue #327](https://github.com/qLDPCOrg/qLDPC/issues/327).
- Exact distance calculations and transversal-gate searches can be exponential.
  When a method returns a bound rather than an exact distance, its docstring explains what kind of bound it is.
- Features that use GAP may start subprocesses, use local disk caches, access online resources, or ask for manual input or permission to install a package.
- Everything under `qldpc.experimental` has an unstable public API and may change without deprecation.
  In particular, lattice-surgery support has not yet received independent expert review; validate its results independently and pin the qLDPC version if you depend on it.

## Documentation and support

- [Documentation](https://qldpc.readthedocs.io/en/latest)
- [Examples](https://github.com/qLDPCOrg/qLDPC/tree/main/examples)
- [API reference](https://qldpc.readthedocs.io/en/latest/autoapi/index.html)
- [Agent and contributor guide](AGENTS.md)
- [Issue tracker](https://github.com/qLDPCOrg/qLDPC/issues)

Questions, feedback, and ideas are welcome through [GitHub issues](https://github.com/qLDPCOrg/qLDPC/issues/new) or by email at [mika.perlin@gmail.com](mailto:mika.perlin@gmail.com).

## Attribution

If you use this software in your work, please cite with:

```bibtex
@misc{perlin2023qldpc,
  author = {Perlin, Michael A.},
  title = {{qLDPC}},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/qLDPCOrg/qLDPC}},
}
```

This may require adding `\usepackage{url}` to your LaTeX file header.
Alternatively:

```text
Michael A. Perlin. qLDPC. https://github.com/qLDPCOrg/qLDPC, 2023.
```

qLDPC is distributed under the [Apache License 2.0](LICENSE).
