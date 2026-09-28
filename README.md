# qLDPC

`qLDPC` is a Python toolkit for constructing and analyzing finite-size [quantum low-density parity-check codes](https://errorcorrectionzoo.org/c/qldpc), as well as stabilizer and subsystem codes more broadly.
The goal is simple: make the quantum error-correction literature easier to explore, reproduce, and build on.
Codes can start from parity-check matrices or from higher-level ingredients such as polynomials, group rings, Tanner graphs, and chain complexes.
Once a code is built, you can inspect its logical operators, compute or bound its distance, plug in a decoder, estimate logical error rates, and—for qubit codes—build [`stim`](https://github.com/quantumlib/Stim) circuits.

Most code-level tools work over arbitrary finite fields.
Circuit tools currently only support qubit codes.

## ✨ A taste of qLDPC

- **From textbook codes to the research frontier.**
  Build familiar surface, toric, Hamming, and Reed–Muller codes alongside hypergraph-product, lifted-product, bivariate-bicycle, quantum Tanner, SHYPS, GALA, and other modern families.
- **Go beyond binary.**
  Many constructions and core algorithms work over arbitrary prime-power finite fields, not just qubits.
- **Ask more of a code than `[[n, k, d]]`.**
  Work with check matrices, Tanner graphs, canonical logical operators, subsystem gauges, exact distances, upper bounds, and code-capacity error estimates.
- **Move from codes to circuits.**
  Build memory experiments, encoding circuits, Pauli noise models, state-preparation diagnostics, Sinter decoders, sliding-window decoders, and transversal Clifford searches.
- **Use the algebra that modern constructions need.**
  qLDPC includes finite groups, group rings, lifted matrices, Cayley complexes, chain complexes, and optional GAP integration.
- **Steal from the examples.**
  The notebooks progress from the basics to circuit-level logical error rates, custom noise, transversal gates, and experimental lattice surgery.

This is still a research library, so the unusual corners are not hidden: expensive searches, heuristic bounds, experimental APIs, and external-tool side effects are called out where they matter.

## 📦 Installation

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

## 🚀 Quickstart

```python
from sympy.abc import x, y

from qldpc import codes

# The [[144, 12, 12]] bivariate bicycle "gross code".
code = codes.BBCode({x: 12, y: 6}, x**3 + y + y**2, y**3 + x + x**2)

print(code)
print("physical qubits:", len(code))
print("logical qubits:", code.dimension)
```

That object is ready for more than a parameter check: inspect its parity checks and logical operators, choose a decoder, estimate a logical error rate, or use it in a memory experiment.

### Go deeper: a lifted-product code

For a more technical taste, the next example builds a parity-check matrix over the group algebra `GF(2)[C₅]`, lifts each group-ring entry to a 5×5 binary block, and hands the result to `LPCode`.
It follows the [lift-connected surface-code construction](https://arxiv.org/abs/2401.02911) and produces a 125-qubit CSS code encoding 5 logical qubits.

```python
from qldpc import abstract, codes

group = abstract.CyclicGroup(5)
ring = abstract.GroupRing(group)
x = ring.generators[0]

repetition = abstract.RingArray.build(
    [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
    ring,
)
connections = abstract.RingArray.build(
    [[0, x, 0, 0], [0, 0, x, 0], [0, 0, 0, x]],
    ring,
)
code = codes.LPCode(repetition + connections)

print("physical qubits:", len(code))      # 125
print("logical qubits:", code.dimension)  # 5
```

Continue with the [library map](https://qldpc.readthedocs.io/library_map.html) for the data model and package structure, or open the [example notebooks](https://qldpc.readthedocs.io/examples/index.html) for complete workflows.

## 🧭 Find your way around

| Goal | Start here |
| --- | --- |
| Understand code representations and how packages fit together | [Library map](https://qldpc.readthedocs.io/library_map.html) |
| Learn by running complete workflows | [Examples](https://qldpc.readthedocs.io/examples/index.html) |
| Browse code families, functions, and exact signatures | [API reference](https://qldpc.readthedocs.io/autoapi/index.html) |
| Construct and analyze a first code | [qLDPC basics](https://qldpc.readthedocs.io/examples/basics.html) |
| Estimate logical error rates | [Logical-error-rate examples](https://qldpc.readthedocs.io/examples/index.html#logical-error-rates) |
| Build memory circuits or noise models | [Circuit examples](https://qldpc.readthedocs.io/examples/index.html) |
| Change or extend qLDPC safely | [Agent and contributor guide](AGENTS.md) |

For the complete list of classes and functions, including construction-specific literature, use the API reference and source docstrings.
This README focuses on getting started and finding the right documentation.

## ⚠️ Limitations and caveats

- Circuit and tableau tools only work with qubit codes, even though other parts of qLDPC support prime-power-dimensional qudits.
- Circuits returned by `get_encoding_circuit` are not fault-tolerant.
  Fault-tolerant encoding is tracked in [issue #327](https://github.com/qLDPCOrg/qLDPC/issues/327).
- Exact distance calculations and transversal-gate searches can be exponential.
  When a method returns a bound rather than an exact distance, its docstring explains what kind of bound it is.
- Features that use GAP may start subprocesses, use local disk caches, access online resources, or ask for manual input or permission to install a package.
- Everything under `qldpc.experimental` has an unstable public API and may change without deprecation.
  In particular, lattice-surgery support has not yet received independent expert review; validate its results independently and pin the qLDPC version if you depend on it.

## 📚 Documentation and support

- [Documentation](https://qldpc.readthedocs.io/)
- [Examples](https://github.com/qLDPCOrg/qLDPC/tree/main/examples)
- [API reference](https://qldpc.readthedocs.io/autoapi/index.html)
- [Agent and contributor guide](AGENTS.md)
- [Issue tracker](https://github.com/qLDPCOrg/qLDPC/issues)

Questions, feedback, and ideas are welcome through [GitHub issues](https://github.com/qLDPCOrg/qLDPC/issues/new) or by email at [mika.perlin@gmail.com](mailto:mika.perlin@gmail.com).

## ⚓ Attribution

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
