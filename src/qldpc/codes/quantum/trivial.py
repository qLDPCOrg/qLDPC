# SPDX-License-Identifier: Apache-2.0

"""The trivial code: a direct sum of single-qudit codes."""

from __future__ import annotations

import galois

from qldpc.math import resolve_field

from ..common import CSSCode

####################################################################################################
# trivial code


class TrivialCode(CSSCode):
    """The trivial code with single-qubit logical operators."""

    def __init__(
        self,
        size: int,
        num_stabs_x: int = 0,
        num_stabs_z: int = 0,
        *,
        gauge_dimension: int = 0,
        self_dual: bool = False,
        field: int | type[galois.FieldArray] | None = None,
    ) -> None:
        """Initialize a trivial code from single-qubit stabilizers and gauge operators.

        The code is a direct sum whose ``size`` qubits are laid out, in order, as

        - ``dimension = size - gauge_dimension - num_stabs_x - num_stabs_z`` logical qubits,
        - ``gauge_dimension`` gauge qubits (each carrying a weight-one X and Z gauge operator),
        - ``num_stabs_x`` qubits each stabilized by a single-qubit X-type operator, and
        - ``num_stabs_z`` qubits each stabilized by a single-qubit Z-type operator.

        If ``self_dual is True``, then ``num_stabs_x`` and ``num_stabs_z`` must be equal, and the
        X-type and Z-type stabilizers are paired into weight-two XX and ZZ operators on shared
        qubits, so that ``matrix_x == matrix_z``.  It is enough to specify just one of
        ``num_stabs_x`` and ``num_stabs_z`` in this case; the other defaults to the same value.
        """
        field = resolve_field(field)

        if self_dual:
            if num_stabs_x and num_stabs_z and num_stabs_x != num_stabs_z:
                raise ValueError(
                    "Self-dual trivial codes must have an equal number of X-type and Z-type"
                    " stabilizers"
                )
            num_stabs_x = num_stabs_z = num_stabs_x or num_stabs_z

        dimension = size - gauge_dimension - num_stabs_x - num_stabs_z
        if dimension < 0:
            raise ValueError(
                f"size ({size}) is too small for gauge_dimension + num_stabs_x + num_stabs_z"
                f" ({gauge_dimension + num_stabs_x + num_stabs_z})"
            )

        num_checks_x = gauge_dimension + num_stabs_x
        num_checks_z = gauge_dimension + num_stabs_z
        matrix_x = field.Zeros((num_checks_x, size))
        matrix_z = field.Zeros((num_checks_z, size))

        # Qubit layout: [logical | gauge | X-stabilizer | Z-stabilizer].
        gauge_qubits = range(dimension, dimension + gauge_dimension)
        stab_x_qubits = range(size - num_stabs_x - num_stabs_z, size - num_stabs_z)
        stab_z_qubits = range(size - num_stabs_z, size)

        # Each gauge qubit carries a weight-one X and Z gauge operator.
        matrix_x[range(gauge_dimension), gauge_qubits] = 1
        matrix_z[range(gauge_dimension), gauge_qubits] = 1

        # Each stabilizer is a weight-one Pauli on its own qubit...
        matrix_x[range(gauge_dimension, num_checks_x), stab_x_qubits] = 1
        matrix_z[range(gauge_dimension, num_checks_z), stab_z_qubits] = 1
        if self_dual:
            # ...except when self-dual, where the i-th X and Z stabilizers are paired into
            # weight-two XX and ZZ operators on the same qubits, making matrix_x == matrix_z.
            matrix_x[range(gauge_dimension, num_checks_x), stab_z_qubits] = 1
            matrix_z[range(gauge_dimension, num_checks_z), stab_x_qubits] = 1

        super().__init__(matrix_x, matrix_z, promise_equal_distance_xz=True)
        self._is_subsystem_code = bool(gauge_dimension)  # known by construction
