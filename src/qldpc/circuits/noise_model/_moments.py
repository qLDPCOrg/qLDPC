# SPDX-License-Identifier: Apache-2.0

"""Idle errors accumulated over moments.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import math

from .channels import PauliChannel, PauliChannelSequence

####################################################################################################
# idle errors


def repeated_idle_channel(
    channel: PauliChannel | PauliChannelSequence, repetitions: int
) -> PauliChannel | PauliChannelSequence:
    """Compose ``repetitions`` independent applications of a one-qubit Pauli channel."""
    if isinstance(channel, PauliChannelSequence):
        # an independent Pauli event that fires with probability p in each of n repetitions has a
        # net effect with probability (1 - (1 - 2p)^n) / 2
        return PauliChannelSequence(
            {
                pauli: (1 - (1 - 2 * probability) ** repetitions) / 2
                for pauli, probability in channel.probabilities.items()
            }
        )

    # compose the distribution over (I, X, Y, Z) by repeated squaring
    probabilities = tuple(channel.probabilities.get(pauli, 0.0) for pauli in "XYZ")
    single: tuple[float, ...] = (max(0.0, 1 - math.fsum(probabilities)), *probabilities)
    result: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    while repetitions:
        if repetitions & 1:
            result = _compose_pauli_distributions(result, single)
        repetitions >>= 1
        if repetitions:
            single = _compose_pauli_distributions(single, single)
    return PauliChannel(dict(zip("XYZ", result[1:], strict=True)))


def quadratic_idle_channel(channel: PauliChannel, num_moments: int) -> PauliChannel:
    """The error of a run of consecutive idle moments, given the quadratic idle error of one.

    If ``channel`` applies Paulis with total probability ``p``, the result applies the same Paulis
    in the same proportions with total probability ``(1 - (1 - 2 * p)**(num_moments**2)) / 2``.
    This is the Pauli twirl of a rotation about a fixed axis by an angle that grows linearly over
    the run, at a Gaussian random rate whose spread is set by ``p``.
    """
    total = math.fsum(channel.probabilities.values())
    # compute (1 - 2 * total)**(num_moments**2) - 1 to full relative precision
    decay = math.expm1(num_moments**2 * math.log1p(-2 * total)) if total < 0.5 else -1.0
    scale = -decay / (2 * total)
    return PauliChannel(
        {pauli: probability * scale for pauli, probability in channel.probabilities.items()}
    )


def compose_idle_channels(
    first: PauliChannel | PauliChannelSequence, second: PauliChannel | PauliChannelSequence
) -> PauliChannel:
    """The one-qubit Pauli channel that applies independent errors from two one-qubit channels."""
    distribution: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0)
    for channel in (first, second):
        if isinstance(channel, PauliChannelSequence):
            # a sequence of independent Pauli events, each of which fires with its own probability
            for pauli, probability in channel.probabilities.items():
                event = [1 - probability, 0.0, 0.0, 0.0]
                event["IXYZ".index(pauli)] = probability
                distribution = _compose_pauli_distributions(distribution, tuple(event))
        else:
            probabilities = tuple(channel.probabilities.get(pauli, 0.0) for pauli in "XYZ")
            single = (max(0.0, 1 - math.fsum(probabilities)), *probabilities)
            distribution = _compose_pauli_distributions(distribution, single)
    return PauliChannel(dict(zip("XYZ", distribution[1:], strict=True)))


def _compose_pauli_distributions(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, ...]:
    """Distribution of the product of independent Paulis, indexed as (I, X, Y, Z).

    Up to phase, Paulis multiply by XOR-ing their indices: e.g. X * Z ~ (1 ^ 3) = 2 ~ Y.  The result
    is renormalized to prevent rounding errors from compounding over repeated compositions.
    """
    distribution = [
        math.fsum(left[index] * right[index ^ outcome] for index in range(4))
        for outcome in range(4)
    ]
    total = math.fsum(distribution)
    return tuple(probability / total for probability in distribution)
