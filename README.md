# qLDPC

This library contains tools for constructing and analyzing [quantum low density parity check (qLDPC) codes](https://errorcorrectionzoo.org/c/qldpc).  At least, that was the original motivation for this library.  In practice, the tools here work just as well for error-correcting stabilizer and subsystem codes more broadly.

In a nutshell, `qLDPC` provides supports a variety of built-in codes and custom codes constructed from parity check matrices that represent stabilizer or gauge group generators.  Once a code is constructed, `qLDPC` automates various tasks of common interest, and integrates with external tools for analyzing error-correcting codes (including [`QDistRnd`](https://docs.gap-system.org/pkg/qdistrnd/doc/chap1_mj.html), [`ldpc`](https://github.com/quantumgizmos/ldpc), [`stim`](https://github.com/quantumlib/Stim), and [`sinter`](https://pypi.org/project/sinter), among others).  Automated tasks include:
- constructing a canonical basis logical Pauli operators,
- computing (or upper-bounding) code distance,
- computing logical error rates in a code-capacity model,
- constructing various circuits of interest, such as a quantum memory experiment for obtaining circuit-level logical error rates,
- defining custom Pauli noise models,
- plugging a decoder of choice into your workflow.

Where possible, this library strives to support qudit codes over arbitrary finite (Galois) fields, although circuit-related utilities are (at least currently) limited to qubit codes.  See the [`examples`](https://github.com/qLDPCOrg/qLDPC/tree/main/examples) directory for some demonstrations and use-cases.

## 📦 Installation

This library requires Python>=3.10, and can be installed from the Python Package Index (PyPI) with
```
pip install qldpc
```

To install a local version of qLDPC from source:
```
git clone https://github.com/qLDPCOrg/qLDPC.git
pip install -e qLDPC
```
You can also `pip install -e 'qLDPC[dev]'` to additionally install some development tools.

### GAP

Some features in `qLDPC` require an installation of the [GAP](https://www.gap-system.org) computer algebra system.  If you (a) use linux or macOS, and (b) use a `conda` to manage your python environment, then you can obtain GAP by running `conda install -c conda-forge gap` (or `gap-core`).  Installations without `conda` should also work, as long as `gap` is a recognized command in the command line.  Unfortunately, I have not figured out how to install [GAP](https://www.gap-system.org) in a `qLDPC`-compatible way on Windows.  If you figure this out, [please let me know](https://github.com/qLDPCOrg/qLDPC/issues/294)!

### macOS

If you use macOS you may need to install `cvxpy` manually by following the instructions [here](https://www.cvxpy.org/install) before installing `qLDPC`.  If you use `conda` to manage your python environment, you can obtain `cvxpy` by running `conda install -c conda-forge cvxpy`.

## 🚀 Features

Notable features include:
- `ClassicalCode`: class for representing classical linear error-correcting codes over finite fields.
  - Various pre-defined classical code families.
  - Communication with the [GAP](https://www.gap-system.org)/[`GUAVA`](https://www.gap-system.org/Packages/guava.html) package for [even more codes](https://docs.gap-system.org/pkg/guava/doc/chap5.html).
- `QuditCode`: class for constructing [Galois-qudit codes](https://errorcorrectionzoo.org/c/galois_into_galois), including both [stabilizer](https://errorcorrectionzoo.org/c/galois_stabilizer) and [subsystem](https://errorcorrectionzoo.org/c/oecc) codes.
  - `QuditCode.get_logical_ops`: method to construct a complete basis of nontrivial logical Pauli operators for a `QuditCode`.
  - `QuditCode.get_distance`: method to compute the exact code distance of a `QuditCode` (i.e., the minimum weight of a nontrivial logical operator).  Includes options to compute an upper bound on code distance using [`QDistRnd`](https://docs.gap-system.org/pkg/qdistrnd/doc/chap1_mj.html) or (for CSS codes) a decoder-based method introduced in [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
  - `QuditCode.concatenate`: method to [concatenate](https://errorcorrectionzoo.org/c/quantum_concatenated) `QuditCode`s in various ways.
- `CSSCode`: subclass of `QuditCode` for the special case of constructing a [quantum CSS code](https://errorcorrectionzoo.org/c/css) out of two mutually compatible `ClassicalCode`s.  Special cases (subclasses) with specialized constructors and helper methods include:
  - `TBCode`: [two-block quantum codes](https://errorcorrectionzoo.org/c/two_block_quantum).
  - `BBCode`: [bivariate bicycle codes](https://errorcorrectionzoo.org/c/quantum_quasi_cyclic), as in [arXiv:2308.07915](https://arxiv.org/abs/2308.07915) and [arXiv:2311.16980](https://arxiv.org/abs/2311.16980).  See [`examples/bivariate_bicycle_codes.ipynb`](https://github.com/qLDPCOrg/qLDPC/blob/main/examples/bivariate_bicycle_codes.ipynb) for methods to identify...
    - toric layouts of a `BBCode`, in which the code looks like a toric code augmented by some long-distance checks, as in discussed in [arXiv:2308.07915](https://arxiv.org/abs/2308.07915), and
    - qubit layouts that minimize the communication distance for neutral atoms, as discussed in [arXiv:2404.18809](https://arxiv.org/abs/2404.18809).
  - `HGPCode`: [hypergraph product codes](https://errorcorrectionzoo.org/c/hypergraph_product), first introduced in [arXiv:0903.0566](https://arxiv.org/abs/0903.0566).
  - `SHPCode`: [subsystem hypergraph product codes](https://errorcorrectionzoo.org/c/subsystem_quantum_parity), as in [arXiv:2002.06257](https://arxiv.org/abs/2002.06257).
  - `SHYPSCode`: [subsystem hypergraph product simplex codes](https://errorcorrectionzoo.org/c/shyps), as in [arXiv:2502.07150](https://arxiv.org/abs/2502.07150).
  - `LPCode`: [lifted product codes](https://errorcorrectionzoo.org/c/lifted_product), as in [arXiv:2012.04068](https://arxiv.org/abs/2012.04068) and [arXiv:2202.01702](https://arxiv.org/abs/2202.01702).
  - `SLPCode`: [subsystem lifted product codes](https://errorcorrectionzoo.org/c/subsystem_lifted_product), as in [arXiv:2404.18302](https://arxiv.org/abs/2404.18302).
  - `QTCode`: [quantum Tanner codes](https://errorcorrectionzoo.org/c/quantum_tanner), as in [arXiv:2202.13641](https://arxiv.org/abs/2202.13641) and [arXiv:2206.07571](https://arxiv.org/abs/2206.07571).
- `decoders.py`: module for decoding errors with various methods, including BP-OSD, BP-LSD, and belief-find (via [`ldpc`](https://github.com/quantumgizmos/ldpc)), Relay-BP (via [`relay-bp`](https://pypi.org/project/relay-bp)), minimum-weight perfect matching (via [`pymatching`](https://github.com/oscarhiggott/PyMatching)), and others.  Includes an interface for using custom decoders. 
- `qldpc.circuits`: module for [`stim`](https://github.com/quantumlib/Stim) circuits and circuit utilities, including:
  - `get_memory_experiment`: circuit for testing the performance of a code as a quantum memory.  Accepts both predefined and custom-built `SyndromeMeasurementStrategy`s.
  - `NoiseModel`: class for constructing expressive Pauli noise models, which map noiseless circuits to noisy circuits.  Built-in subclasses include a single-parameter `DepolarizingNoiseModel` and a superconducting-inspired `SI1000NoiseModel`.
  - `SinterDecoder`: class to construct circuit-level decoders that are usable by [`sinter`](https://pypi.org/project/sinter).
  - `get_encoding_circuit`: circuit to prepare the all-|0> logical state of a code.  (Warning: generally not fault-tolerant.  Fault-tolerant encoding circuits [pending](https://github.com/qLDPCOrg/qLDPC/issues/327).)
  - `get_transversal_ops`: logical tableaus and physical circuits for the SWAP-transversal logical Clifford gates of a code, constructed via the code automorphism method of [arXiv:2409.18175](https://arxiv.org/abs/2409.18175).  (Warning: exponential complexity.)
  - `get_transversal_circuits`: find a SWAP-transversal physical circuit (if any) that implements a given logical Clifford operation in a code.  (Warning: exponential complexity.)
- `abstract.py`: module for abstract algebra (groups, algebras, modules, and representations thereof).
  - Various pre-defined groups (mostly borrowed from [SymPy](https://docs.sympy.org/latest/modules/combinatorics/named_groups.html)).
  - Communication with the [GAP](https://www.gap-system.org) computer algebra system and [GroupNames.org](https://people.maths.bris.ac.uk/~matyd/GroupNames) for constructing [even more groups](https://docs.gap-system.org/doc/ref/chap50.html).
- `objects.py`: module for constructing helper objects such as Cayley complexes and chain complexes, which are instrumental for the construction of various quantum codes.

## 🤔 Questions and issues

This project aspires to have a [documentation page](https://qldpc.readthedocs.io/en/latest), but at the moment the documentation is out of date and auto-generated from source code that was written to be human-readable in a plain text editor.  For now, I recommend looking at the source code (and comments therein) directly, as well as the [`examples`](https://github.com/qLDPCOrg/qLDPC/tree/main/examples) directory.  Test files (such as [`qldpc/codes/quantum_test.py`](https://github.com/qLDPCOrg/qLDPC/blob/main/qldpc/codes/quantum_test.py)) also contain some examples of using the classes and methods in this library.

If you have any questions, feedback, or requests, please [open an issue on GitHub](https://github.com/qLDPCOrg/qLDPC/issues/new) or email me at [mika.perlin@gmail.com](mailto:mika.perlin@gmail.com)!

## ⚓ Attribution

If you use this software in your work, please cite with:
```
@misc{perlin2023qldpc,
  author = {Perlin, Michael A.},
  title = {{qLDPC}},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/qLDPCOrg/qLDPC}},
}
```
This may require adding `\usepackage{url}` to your LaTeX file header.  Alternatively, you can cite
```
Michael A. Perlin. qLDPC. https://github.com/qLDPCOrg/qLDPC, 2023.
```
