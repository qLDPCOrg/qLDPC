# Workarounds for slow paths of Stim

This guide lists the code in qLDPC that exists only because a path of [Stim](https://github.com/quantumlib/Stim) is slow, with one section for each path, and says how to remove the code once Stim is fast.
Algorithmic choices, ordinary uses of Stim's API, and caches that would stay useful do not belong here.

Each site carries a marker that names its section, so `git grep "TEMPORARY until"` lists every site.
A function that the migration deletes has the marker right after its docstring's summary line.
Anything else that the migration deletes or changes has the marker as a comment above it:

```python
# TEMPORARY until <trigger>: see circuits/stim_workarounds.md, "<section title>".
```

A trigger `stim>=X` holds once qLDPC's `pyproject.toml` requires Stim `X` or later; any other trigger is a behavior of Stim that its section's script checks.
The values below are from stim 1.17.dev1793242987 on macOS on arm64.
The cost of some of Stim's Python bindings varies widely between platforms ([quantumlib/Stim#1087](https://github.com/quantumlib/Stim/issues/1087)), so check a trigger on that platform too.
To migrate, follow the section's steps, delete every import of a name that they delete, check that the call that the section names gets no slower, and delete the section.

## Resets with random outcomes in stim.TableauSimulator

`stim.TableauSimulator` collapses a qubit by transposing its whole tableau, so a reset whose outcome is random takes time quadratic in the number of qubits, even if the qubit is unentangled: about 2.5 ms on 2,000 qubits, while a gate takes 0.3 µs.

**Trigger:** such resets take linear time, so that this script prints at most about 4 rather than about 16.

```python
import time
import stim

def reset_time(num_qubits: int) -> float:
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(num_qubits)
    resets = stim.Circuit("RX 0\nR 0")  # each reset has a random outcome
    start = time.perf_counter()
    for _ in range(10):
        simulator.do(resets)
    return time.perf_counter() - start

print(reset_time(4000) / reset_time(1000))
```

**Migration:**

1. In `encoding.get_state_stabilizers`, replace the marked `if`/`else` by `simulator.do(circuit)`.
2. Delete `_do_with_cheap_resets`, `_MIN_QUBITS_FOR_CHEAP_RESETS`, `_RESET_PAULIS`, `_PAULI_FLIPS`, and `_PAULI_SWAPS` of `encoding`, with their comments, and `test_state_stabilizers_with_cheap_resets` of `encoding_test`.

**Check:** `get_state_stabilizers(circuit, 2000)` for `circuit = stim.Circuit("\n".join(f"RX {q}\nCX {q} {q + 1}" for q in range(0, 2000, 2)))`, which takes about 0.2 s.

## stim.Circuit.append in stim 1.16

In stim 1.16, `stim.Circuit.append` converts each target slowly ([quantumlib/Stim#1087](https://github.com/quantumlib/Stim/issues/1087), fixed in stim 1.17 by [quantumlib/Stim#1090](https://github.com/quantumlib/Stim/pull/1090)): appending `CX 1 2` takes about 300 µs, while parsing its text takes 0.4 µs.
So the marked code builds circuits by parsing their text once.

**Trigger:** `stim>=1.17`.

**Migration:**

1. In `common.get_pauli_product_measurements`, `_post_selection.get_flag_detectors`, `encoding._get_fixed_parities_by_simulation`, and `fault_propagation._get_primitive_fault_signatures`, build each marked circuit with `stim.Circuit.append`, with `stim.target_combiner()` between the Pauli targets of each MPP product.
   Keep the checks of `get_pauli_product_measurements`, whose messages differ from Stim's.
2. Delete `common._get_pauli_product_text` and its call in `common_test`.
3. In `deferred_measurement`, make `_GateTarget.to_stim` return an uninverted qubit as an `int`, and replace `_CircuitInstruction.to_stim` by a method that appends the instruction to a given circuit, with `stim.target_combiner()` between the targets of each Pauli product.
4. Make `_get_circuit` append each instruction with that method.
   Delete its text path, its check for non-finite arguments, the import of `math`, and what its docstring and the comment that opens "instructions as Python objects" say about parsing text.
   Keep `_CircuitInstruction.__str__` and `_escape`, which error messages use, and the reason that `_GateTarget` and `_CircuitInstruction` stay.

**Check:** `certify_bounded_fault_propagation` on the circuit of `test_steane_preparation_meets_css_weight_but_not_physical_weight`, which takes about 1 ms, and `to_deferred_measurement_form` on `stim.Circuit.generated("surface_code:rotated_memory_z", distance=9, rounds=9, after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3, after_reset_flip_probability=1e-3)`, which takes about 60 ms.

## Targets of detector error models

Finding the detectors of the errors of a detector error model through Stim's Python objects takes about 3 times as long as parsing the model's text, so `_get_primitive_fault_signatures` parses the text.

**Trigger:** Stim's Python objects are at least as fast, so that this script, which computes the signature of each error both ways, prints at most 1 rather than about 3.
If Stim adds a way to read the targets of all errors at once, time that instead.

```python
import timeit
import stim

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=9, rounds=9, after_clifford_depolarization=1e-3
)
dem = circuit.detector_error_model().flattened()
by_text = {f"D{jj}": 1 << jj for jj in range(dem.num_detectors)}
by_index = {jj: 1 << jj for jj in range(dem.num_detectors)}

def from_text() -> list[int]:
    signatures = []
    for line in str(dem).split("\n"):
        if line.startswith("error"):
            bits = 0
            for target in line[line.index(")") + 1 :].split():
                bits ^= by_text.get(target, 0)
            signatures.append(bits)
    return signatures

def from_objects() -> list[int]:
    signatures = []
    for instruction in dem:
        if instruction.type == "error":
            bits = 0
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    bits ^= by_index.get(target.val, 0)
            signatures.append(bits)
    return signatures

assert from_text() == from_objects()
print(timeit.timeit(from_objects, number=5) / timeit.timeit(from_text, number=5))
```

**Migration:**

1. In `fault_propagation._get_primitive_fault_signatures`, key `signature_bits` by detector index, replace the parse of `str(dem)` by the loop of `from_objects` above, and delete the comments about the text of targets.
2. In `test_primitive_fault_signatures_follow_the_targets_of_error_mechanisms` of `fault_propagation_test`, delete the marked case, and make `get_signatures` return the sorted signatures without recomputing them from Stim's objects.

**Check:** `_get_primitive_fault_signatures(circuit, [], [], range(circuit.num_detectors))` for the circuit of the script, which takes about 25 ms.

## The stim.CircuitInstruction constructor

The constructor of `stim.CircuitInstruction` takes about 30 µs per `stim.GateTarget`, so `common._new_instruction` appends the instruction to an empty circuit and indexes it, which takes about 2 µs.

**Trigger:** with stim 1.17 or later, the constructor is at least as fast as appending and indexing, so that this script, whose targets are like those that the callers of `_new_instruction` pass, prints at most 1 rather than about 32.

```python
import timeit
import stim

circuit = stim.Circuit("CX 0 1\nMPP[tag] X0*Z1 !Y2\nM(0.01) !4\nCX rec[-1] 5\nDETECTOR(1, 2) rec[-1]")
cases = [(op.name, op.targets_copy(), op.gate_args_copy(), op.tag) for op in circuit]

def construct() -> None:
    for name, targets, args, tag in cases:
        stim.CircuitInstruction(name, targets, args, tag=tag)

def append_and_index() -> None:
    for name, targets, args, tag in cases:
        new = stim.Circuit()
        new.append(name, targets, args, tag=tag)
        new[0]

print(timeit.timeit(construct, number=200) / timeit.timeit(append_and_index, number=200))
```

**Migration:**

1. Replace each call `_new_instruction(name, targets, args, tag)` in `qldpc.circuits.noise_model` by `stim.CircuitInstruction(name, targets, args, tag=tag)`.
2. Delete `common._new_instruction` and `common_test.test_new_instruction`.

**Check:** `SI1000NoiseModel(1e-3).noisy_circuit(stim.Circuit.generated("surface_code:rotated_memory_z", distance=11, rounds=11))` after `scheduling._get_instruction_operations.cache_clear()`, which takes about 30 ms.

## Iterating over stim circuits

Iterating over a `stim.Circuit` takes about 90 µs more than indexing its instructions, which adds up over the many small circuits of noise models, so `common._get_instructions` indexes the circuit.

**Trigger:** iterating is about as fast as indexing, so that this script prints at most 2 rather than about 80.

```python
import timeit
import stim

circuit = stim.Circuit("H 0\nCX 0 1")
iterate = timeit.timeit(lambda: list(circuit), number=1000)
index = timeit.timeit(lambda: [circuit[i] for i in range(len(circuit))], number=1000)
print(iterate / index)
```

**Migration:**

1. Replace each call `_get_instructions(circuit)` in `qldpc.circuits` by `circuit`.
   Keep the loops over `range(len(circuit))` that read a `StimCircuitProtocol`, which declares no iterator.
2. Delete `common._get_instructions`.

**Check:** 200 calls of `SI1000NoiseModel(1e-3).noisy_circuit(stim.Circuit("H 0\nCX 0 1"))`, which take about 10 ms.
