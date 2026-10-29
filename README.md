# qLDPC

`qLDPC` is a Python toolkit for constructing and analyzing finite-size [quantum low-density parity-check codes](https://errorcorrectionzoo.org/c/qldpc), as well as stabilizer and subsystem codes more broadly.
It provides shared implementations and scientific conventions that researchers and their AI assistants can inspect, combine, and extend.

Most code-level tools work over arbitrary finite fields.
Circuit tools currently only support qubit codes.

## 🎯 Goals

- **Enable exploration.**
  Favor small, composable tools and extension points that let users quickly investigate and prototype new ideas.
- **Make scientific meaning explicit.**
  Expose conventions, assumptions, and limitations, and distinguish exact results, bounds, estimates, and trusted inputs.
- **Support checking and reproduction.**
  Provide independent evidence for maintained behavior and make the configuration needed to interpret and reproduce experiments clear.
- **Support learning.**
  Use small examples and explanations to connect computations to ideas, literature, and implementations.

## ✨ A few things to try

- **Code families.**
  Construct surface, toric, Hamming, and Reed–Muller codes, along with hypergraph-product, lifted-product, bivariate-bicycle, quantum Tanner, SHYPS, GALA, and other families.
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

qLDPC requires Python 3.11 or later:

```bash
python -m pip install qldpc
```

The standard installation includes the `ldpc` and `pymatching` decoders.
See the [installation guide](https://qldpc.readthedocs.io/installation.html) for optional decoders, GAP integration, GPU distance estimation with `sqetch`, and troubleshooting.
For a development checkout, follow the [contributor reference](https://qldpc.readthedocs.io/contributing.html#development-installation).

## 🧭 Find your way around

| Goal | Start here |
| --- | --- |
| Understand code representations and how packages fit together | [Library map](https://qldpc.readthedocs.io/library_map.html) |
| Learn the concepts through runnable examples | [Examples](https://qldpc.readthedocs.io/examples/index.html) |
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

## AI usage

AI assistants are used in qLDPC's development and maintenance.
Human maintainers direct and approve changes, but may rely on AI-assisted review and automated checks rather than reviewing every implementation detail.
AI-assisted contributions are held to the same [standards for scientific correctness, testing, documentation, and compatibility](AGENTS.md) as other contributions.
Project maintainers remain responsible for accepting contributions and communicating their assumptions and limitations.

## ⚓ Attribution

If you use qLDPC in your work, please follow the [citation instructions](CITATION.md).

qLDPC is distributed under the [Apache License 2.0](https://github.com/qLDPCOrg/qLDPC/blob/main/LICENSE).
