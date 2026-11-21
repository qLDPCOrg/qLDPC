# SPDX-License-Identifier: Apache-2.0

"""Brouwer-Zimmermann searches of the distances of nested codes and of the cosets of a code.

Their interfaces are ``qldpc.codes.distance.distances`` and ``qldpc.codes.distance.coset_leaders``.
The counters of ``qldpc.codes.distance.logical_counts`` also run the distance searches, and
``qldpc.codes.distance._auto`` runs them, advancing them one layer at a time, for ``method="auto"``.
"""
