# qLDPC

`qLDPC` is a Python toolkit for constructing and analyzing finite-size [quantum low-density parity-check codes](https://errorcorrectionzoo.org/c/qldpc), as well as stabilizer and subsystem codes more broadly.
It aims to make constructions and algorithms from the quantum error-correction literature easier to explore, reproduce, and build on.
You can define codes using parity-check matrices or higher-level ingredients such as polynomials, group rings, Tanner graphs, and chain complexes.
From there, you can inspect logical operators, compute or bound code distance, choose a decoder, estimate logical error rates, and—for qubit codes—build [`stim`](https://github.com/quantumlib/Stim) circuits.

Most code-level tools work over arbitrary finite fields.
Circuit tools currently only support qubit codes.

## ✨ A few things to try

- **Code families.**
  Construct surface, toric, Hamming, and Reed–Muller codes, along with hypergraph-product, lifted-product, balanced-product, bivariate-bicycle, quantum Tanner, SHYPS, GALA, and other families.
- **Distance calculations.**
  Compute exact distances of binary codes with an [optimized Brouwer–Zimmermann search](src/qldpc/codes/distance/brouwer_zimmermann.md), or bound code distance using [QDistRnd](https://arxiv.org/abs/2308.15140) or [`sqetch`](https://github.com/a7b/yarn/tree/main/sqetch).
- **Code structure and analysis.**
  Work with check matrices, Tanner graphs, canonical logical operators, subsystem gauge operators, and code-capacity error estimates.
- **Circuits and decoding workflows.**
  Build memory experiments, encoding circuits, Pauli noise models, and state-preparation diagnostics; use Sinter and sliding-window decoders, or search for transversal Clifford gates.
- **Algebraic tools.**
  Work with finite groups, group rings, lifted matrices, Cayley complexes, chain complexes, and optional GAP integration.

The [example notebooks](https://qldpc.readthedocs.io/examples/index.html) cover (among other things) code construction, custom noise models, and circuit-level logical error rates.

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

The gross code is also available as the named construction, `codes.GrossCode()`.

### A subsystem lifted-product code

This example builds a parity-check matrix over the group algebra `GF(2)[C₅]`, lifts each group-ring entry to a 5×5 binary block, and passes the result to `SLPCode`.
The [subsystem lifted-product construction](https://arxiv.org/abs/2404.18302) produces an 80-qubit CSS subsystem code with 5 logical qubits and 45 gauge qubits.

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
code = codes.SLPCode(repetition + connections)

print("physical qubits:", len(code))            # 80
print("logical qubits:", code.dimension)        # 5
print("gauge qubits:", code.gauge_dimension)    # 45
```

The [library map](https://qldpc.readthedocs.io/library_map.html) explains the data model and package structure.

## 📦 Installation

### Standard installation

qLDPC requires Python 3.11 or later:

```bash
python -m pip install qldpc
```

### Optional extras

#### Decoders

qLDPC installs `ldpc` and `pymatching` by default, which provide several decoding algorithms.
Install the optional decoder integrations published as qLDPC extras for your platform with:

```bash
python -m pip install 'qldpc[decoders]'
```

You can also use individual extras to install a specific optional decoder:

```bash
python -m pip install 'qldpc[ilp]'
python -m pip install 'qldpc[relay-bp]'
python -m pip install 'qldpc[tesseract]'
```

The [Tesseract](https://github.com/quantumlib/tesseract-decoder) decoder requires CPython 3.12–3.14 on macOS arm64 or Linux x86-64.
[Frontier](https://github.com/aleverrier/frontier) is not published on PyPI and is not included in an extra; see the [decoder guide](https://qldpc.readthedocs.io/decoders.html) for availability and build-time installation guidance.

#### GAP integration

Some algebra tools, code lookups, and distance estimates require [GAP](https://www.gap-system.org).
For the recommended in-process integration, install qLDPC with its maintained libgap extra:

```bash
python -m pip install 'qldpc[gap]'
```

qLDPC uses `passagemath-gap` when available, so supported GAP operations do not launch a separate process.
The in-process binding requires Python 3.13 on a non-Windows platform.
If `passagemath-gap` is unavailable, qLDPC relies on a GAP executable in your `PATH` (possibly obtainable with `conda install -c conda-forge gap`), and a manual copy/paste fallback.
GAP integration on Windows remains limited; see [issue #294](https://github.com/qLDPCOrg/qLDPC/issues/294).

#### `sqetch` distance estimation

You can install the optional `sqetch` distance-estimation backend with:

```bash
python -m pip install 'sqetch[gpu] @ git+https://github.com/a7b/yarn.git@e9ce9d0fcecc973988558bfee27fab6d6b8d7f97#subdirectory=sqetch'
```

This command installs the upstream version of `sqetch` that qLDPC is tested against, with GPU support.
`sqetch` is not published on PyPI, so it is not available as a qLDPC extra.
For binary CSS codes, you can select this backend with `code.get_distance_bound(backend="sqetch")`.

### Troubleshooting

On macOS, you may need to install `cvxpy`, which the ILP decoder requires, manually by following its [installation instructions](https://www.cvxpy.org/install) before installing `qldpc[ilp]` or `qldpc[decoders]`.
If you use Conda to manage your Python environment, install `cvxpy` with:

```bash
conda install -c conda-forge cvxpy
```

### Development installation

For a development installation:

```bash
git clone https://github.com/qLDPCOrg/qLDPC.git
cd qLDPC
python -m pip install -e '.[dev]'
```

## 🧭 Find your way around

| Goal | Start here |
| --- | --- |
| Understand code representations and how packages fit together | [Library map](https://qldpc.readthedocs.io/library_map.html) |
| Learn by running complete workflows | [Examples](https://qldpc.readthedocs.io/examples/index.html) |
| Browse code families, functions, and exact signatures | [API reference](https://qldpc.readthedocs.io/autoapi/index.html) |
| Construct and analyze a code | [qLDPC basics](https://qldpc.readthedocs.io/examples/basics.html) |
| Estimate logical error rates | [Logical-error-rate examples](https://qldpc.readthedocs.io/examples/index.html#logical-error-rates) |
| Build memory circuits or noise models | [Circuit examples](https://qldpc.readthedocs.io/examples/index.html) |
| Change or extend qLDPC safely | [Agent and contributor guide](https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md) |

## ⚠️ Limitations and caveats

qLDPC is research software.
Some searches are expensive, some bounds are heuristic, experimental APIs can change, and optional tools may have side effects.

- Circuit and tableau tools only work with qubit codes, even though other parts of qLDPC support prime-power-dimensional qudits.
- Circuits returned by `get_encoding_circuit` are not fault-tolerant.
  Fault-tolerant encoding is tracked in [issue #327](https://github.com/qLDPCOrg/qLDPC/issues/327).
- Exact distance calculations and transversal-gate searches can be exponential.
  When a method returns a bound rather than an exact distance, its docstring explains what kind of bound it is.
- Features that use GAP may start subprocesses, access online resources, or ask for manual input or permission to install a package.
- Code lookups such as `QuditCode.from_qecdb_id` access online resources and cache results on disk; pass `refresh=True` to fetch the latest data.
- Everything under `qldpc.experimental` has an unstable public API and may change without deprecation.
- Lattice-surgery support in `qldpc.experimental.surgery` is experimental and unstable, and its implementation is still pending careful expert review.
  Validate its results independently, and pin the qLDPC version if you depend on it.

## 📚 Documentation and support

- [Documentation](https://qldpc.readthedocs.io/)
- [Examples](https://github.com/qLDPCOrg/qLDPC/tree/main/examples)
- [API reference](https://qldpc.readthedocs.io/autoapi/index.html)
- [Agent and contributor guide](https://github.com/qLDPCOrg/qLDPC/blob/main/AGENTS.md)
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

qLDPC is distributed under the [Apache License 2.0](https://github.com/qLDPCOrg/qLDPC/blob/main/LICENSE).
