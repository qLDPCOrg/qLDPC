# Workarounds for slow paths of Stim

This guide lists the code in qLDPC that exists only because a path of [Stim](https://github.com/quantumlib/Stim) is slow, with one section for each limitation of Stim.
Code belongs here if it would be deleted or simplified mechanically once Stim is fast.
Algorithmic choices, ordinary uses of Stim's API, and caches that would remain useful do not.

Each site of a workaround carries a marker.
In the docstring of a function, the marker is the paragraph right after the summary line:

```python
def function():
    """Summary line.

    TEMPORARY until <trigger>: see circuits/stim_workarounds.md, "<section title>".

    More description.
    """
```

Elsewhere, such as above a constant or a block of code, the marker is a comment:

```python
# TEMPORARY until <trigger>: see circuits/stim_workarounds.md, "<section title>".
```

So `git grep "TEMPORARY until"` lists every site.
A trigger `stim>=X` holds once the dependency on Stim in qLDPC's `pyproject.toml` requires at least version `X`.
Any other trigger is a behavior of Stim, which its section shows how to check.
Measurements are from macOS on arm64.
The cost of some paths of Stim's Python bindings varies widely between platforms, as [quantumlib/Stim#1087](https://github.com/quantumlib/Stim/issues/1087) reports for appending to circuits, so check a trigger on macOS on arm64 as well.
When the migration of a section lands, delete the section along with its markers.

## Resets with random outcomes in stim.TableauSimulator

**Slow path:**
`stim.TableauSimulator` collapses a qubit by transposing its whole tableau, so a reset instruction whose outcome is random costs time quadratic in the number `n` of qubits, even if the qubit is unentangled.
In stim 1.16.0 and 1.17.dev1793242987, such a reset on 2,000 qubits takes about 2.5 ms, while a single-qubit gate takes about 0.3 µs.

**Trigger:**
`stim.TableauSimulator` resets an unentangled qubit in time at most linear in `n`.
The following script prints about 16 while such resets take quadratic time, and at most about 4 once they take linear time.

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

**Sites:**
In `qldpc.circuits.encoding`, the marked `if`/`else` of `get_state_stabilizers` simulates circuits on at least `_MIN_QUBITS_FOR_CHEAP_RESETS` qubits with `_do_with_cheap_resets`, which replaces the reset of each qubit in an eigenstate of another Pauli by single-qubit gates that it looks up in `_RESET_PAULIS`, `_PAULI_FLIPS`, and `_PAULI_SWAPS`.
In `qldpc.circuits.encoding_test`, `test_state_stabilizers_with_cheap_resets` exists only for this replacement.

**Migration:**

1. In `get_state_stabilizers`, replace the marked `if`/`else`, with its marker, by `simulator.do(circuit)`.
2. Delete `_do_with_cheap_resets`, `_MIN_QUBITS_FOR_CHEAP_RESETS`, `_RESET_PAULIS`, `_PAULI_FLIPS`, and `_PAULI_SWAPS`, with the comments above them.
3. Delete `test_state_stabilizers_with_cheap_resets`.
4. Delete this section.

**Check:**
Time `get_state_stabilizers(circuit, 2000)` before and after the migration, where `circuit = stim.Circuit("\n".join(f"RX {q}\nCX {q} {q + 1}\nTICK" for q in range(0, 2000, 2)))` prepares 1,000 Bell pairs with one reset instruction each.
With the workaround and stim 1.17.dev1793242987, it takes about 0.2 s, and about 3 s without the workaround.
It should take no longer after the migration.

## stim.Circuit.append in stim 1.16

**Slow path:**
In stim 1.16.0, `stim.Circuit.append` converts each target slowly: `circuit.append("CX", [1, 2])` takes about 300 µs, while `stim.Circuit("CX 1 2")` parses the same instruction in about 0.4 µs.
In stim 1.17.dev1793242987, appending takes about 0.3 µs.

**Upstream:**
[quantumlib/Stim#1087](https://github.com/quantumlib/Stim/issues/1087), fixed for stim 1.17 by [quantumlib/Stim#1090](https://github.com/quantumlib/Stim/pull/1090).

**Trigger:**
`stim>=1.17`.

**Sites:**
These functions build circuits by joining the text of their instructions and parsing it once:

- in `qldpc.circuits.common`, `get_pauli_product_measurements`, with `_get_pauli_product_text`, which writes the targets of an MPP instruction;
- `qldpc.circuits._post_selection.get_flag_detectors`;
- `qldpc.circuits.encoding._get_fixed_parities_by_simulation`;
- `qldpc.circuits.certification.fault_propagation._get_primitive_fault_signatures`, for its detectors on the measured stabilizers;
- `qldpc.experimental.circuits.transforms.deferred_measurement._get_circuit`, which builds each output circuit of the module's passes.

In `qldpc.circuits.common_test`, the call of `_get_pauli_product_text` in `test_pauli_product_measurement_edge_cases` exists only for the workaround.

**Migration:**

1. In `get_pauli_product_measurements`, `get_flag_detectors`, `_get_fixed_parities_by_simulation`, and `_get_primitive_fault_signatures`, build each circuit by calling `stim.Circuit.append` for each instruction, with `stim.target_combiner()` between the Pauli targets of each MPP product.
   Keep the checks in `get_pauli_product_measurements`, whose messages differ from Stim's.
2. Delete `_get_pauli_product_text`, and its call in `test_pauli_product_measurement_edge_cases`.
3. In `deferred_measurement`, make `_GateTarget.to_stim` return a plain, uninverted qubit as an `int`, rather than as `stim.GateTarget(index)`, whose constructor is slow.
4. Replace `_CircuitInstruction.to_stim`, which calls the slow constructor of `stim.CircuitInstruction`, by a method that appends the instruction to a circuit with `circuit.append(name, targets, arguments, tag=tag)`.
   Its targets are those of `_GateTarget.to_stim`, with `stim.target_combiner()` between the targets of each Pauli product.
5. Make `_get_circuit` append each instruction with that method.
   Remove what only parsing text needs: the text path of `_get_circuit`, its check for non-finite arguments, and the import of `math`.
   Keep `_CircuitInstruction.__str__` and `_escape`, since error messages use both.
6. In the comment that opens the section "instructions as Python objects" of `deferred_measurement`, and in the docstring of `_get_circuit`, delete what describes building circuits from text.
   Keep the reason that `_GateTarget` and `_CircuitInstruction` stay.
7. Keep `test_non_finite_arguments` and `test_split_measure_resets` of `deferred_measurement_test`, which check results, but delete "which stim cannot parse" from the docstring of the former.
8. Delete this section.

**Check:**
With stim 1.17 or later, time the following before and after the migration; neither should get slower.

- `certify_bounded_fault_propagation` on the noisy Steane-code preparation of `test_steane_preparation_meets_css_weight_but_not_physical_weight`, which takes about 1 ms.
- `to_deferred_measurement_form` on `stim.Circuit.generated("surface_code:rotated_memory_z", distance=9, rounds=9, after_clifford_depolarization=1e-3, before_measure_flip_probability=1e-3, after_reset_flip_probability=1e-3)`, which takes about 60 ms.

## Targets of detector error models

**Slow path:**
Reading the targets of the error instructions of a detector error model (DEM) through Stim's Python objects, with `DemInstruction.targets_copy`, `DemTarget.is_relative_detector_id`, and `DemTarget.val`, takes about 3 times as long as parsing the DEM's text.
For the 13,617 error mechanisms of a rotated surface-code memory circuit of distance 9, it takes 36 ms against 11 ms, in stim 1.16.0 and 1.17.dev1793242987.

**Trigger:**
Stim's Python API reads the targets of a DEM's error instructions at least as fast as `str(dem)` and a split of its lines, whether through `DemInstruction.targets_copy` or through an accessor that returns the targets of all error instructions at once.
The following script prints about 2.3 in stim 1.17.dev1793242987, and at most 1 once this holds for `targets_copy`; time a new accessor in place of `targets_copy`.

```python
import time
import stim

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z", distance=9, rounds=9, after_clifford_depolarization=1e-3
)
dem = circuit.detector_error_model().flattened()
start = time.perf_counter()
from_text = [line.split(")")[1].split() for line in str(dem).split("\n") if line.startswith("error")]
middle = time.perf_counter()
from_objects = [instruction.targets_copy() for instruction in dem if instruction.type == "error"]
end = time.perf_counter()
print((end - middle) / (middle - start))
```

**Sites:**
`qldpc.circuits.certification.fault_propagation._get_primitive_fault_signatures` parses the DEM's text, and maps the text of each target, such as `D3`, to the bits of its signature.
In `qldpc.circuits.certification.fault_propagation_test`, the first case of `test_primitive_fault_signatures_follow_the_targets_of_error_mechanisms`, whose tags hold `]`, `)`, the names of targets, and a form feed, exists only for the parse.

**Migration:**

1. In `_get_primitive_fault_signatures`, key the map `signature_bits` by detector index rather than by text.
   Replace the loop over the lines of `str(dem)` by a loop over the DEM's error instructions that XORs the bits of each detector target.
   Delete the comments about the text of targets and about tags.
2. In `test_primitive_fault_signatures_follow_the_targets_of_error_mechanisms`, delete the first case and its comment.
   Make its helper `get_signatures` return the sorted signatures of `_get_primitive_fault_signatures` without recomputing them from Stim's objects, since the function then reads the same objects.
   Keep the second case, whose expected signatures check results.
3. Delete this section.

**Check:**
Time `_get_primitive_fault_signatures(circuit, [], [], range(circuit.num_detectors))` before and after the migration, for the circuit of the trigger's script, whose DEM has 13,617 error mechanisms.
It takes about 25 ms, and should take no longer after the migration.

## The stim.CircuitInstruction constructor

**Slow path:**
The constructor of `stim.CircuitInstruction` takes about 50 µs per integer target, in stim 1.16.0 and 1.17.dev1793242987.
In stim 1.17.dev1793242987, appending the same instruction to an empty circuit and indexing it takes about 1.3 µs.

**Trigger:**
With stim 1.17 or later, the constructor of `stim.CircuitInstruction` is at least as fast as appending the instruction to an empty `stim.Circuit` and indexing it.
The following script prints about 200 in stim 1.17.dev1793242987, and at most 1 once this holds.
Run it with stim 1.17 or later, since stim 1.16 appends slowly (see "stim.Circuit.append in stim 1.16").

```python
import timeit
import stim

def append_and_index() -> stim.CircuitInstruction:
    circuit = stim.Circuit()
    circuit.append("CX", [0, 1, 2, 3])
    return circuit[0]

constructor = timeit.timeit(lambda: stim.CircuitInstruction("CX", [0, 1, 2, 3]), number=1000)
print(constructor / timeit.timeit(append_and_index, number=1000))
```

**Sites:**
`qldpc.circuits.common._new_instruction` builds an instruction by appending it to an empty circuit.
`qldpc.circuits.common_test.test_new_instruction` exists only for it.

**Migration:**

1. Replace each call `_new_instruction(name, targets, args, tag)`, in the modules `noise_model`, `rules`, and `scheduling` of `qldpc.circuits.noise_model`, by `stim.CircuitInstruction(name, targets, args, tag=tag)`.
2. Delete `_new_instruction` and `test_new_instruction`.
3. Delete this section.

**Check:**
Time `SI1000NoiseModel(1e-3).noisy_circuit(circuit)` before and after the migration, for `circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=11, rounds=11)`, and call `scheduling._get_instruction_operations.cache_clear()` before each run.
In stim 1.17.dev1793242987, it takes about 30 ms with `_new_instruction`, and about 200 ms with the constructor.
It should take no longer after the migration.

## Iterating over stim circuits

**Slow path:**
Iterating over a `stim.Circuit` takes about 90 µs more than indexing its instructions, in stim 1.16.0 and 1.17.dev1793242987.
So iterating over a circuit of one instruction takes more than 100 times as long as indexing it.

**Trigger:**
Iterating over a `stim.Circuit` is about as fast as indexing its instructions.
The following script prints about 80 in stim 1.17.dev1793242987, and at most 2 once this holds.

```python
import timeit
import stim

circuit = stim.Circuit("H 0\nCX 0 1")
iterate = timeit.timeit(lambda: list(circuit), number=1000)
index = timeit.timeit(lambda: [circuit[i] for i in range(len(circuit))], number=1000)
print(iterate / index)
```

**Sites:**
`qldpc.circuits.common._get_instructions` yields the instructions of a circuit by indexing it.

**Migration:**

1. Replace each call `_get_instructions(circuit)`, in `qldpc.circuits.common` and in the modules `noise_model` and `scheduling` of `qldpc.circuits.noise_model`, by `circuit`.
2. Delete `_get_instructions`.
   Keep the loops over `range(len(circuit))` that read circuits of `StimCircuitProtocol`, which declares no iterator.
3. Delete this section.

**Check:**
Time 200 calls of `SI1000NoiseModel(1e-3).noisy_circuit(stim.Circuit("H 0\nCX 0 1"))` before and after the migration.
In stim 1.17.dev1793242987, they take about 10 ms with `_get_instructions`, and about 60 ms with Stim's iterator.
They should take no longer after the migration.
