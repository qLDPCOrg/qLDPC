# SPDX-License-Identifier: Apache-2.0

"""Typed decoder specifications for deferred construction."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from typing import (
    Any,
    Concatenate,
    Generic,
    Never,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import stim

from qldpc.math import IntegerArray

from ..adapters.error_decoders import ErrorsToObservablesDecoder as _ErrorsToObservablesDecoder
from ..adapters.observable_decoders import (
    validate_observable_decoder as _validate_observable_decoder,
)
from ..protocols import ErrorDecoder, ObservableDecoder, SupportsDecode
from .factories import DEMDecoderFactory, MatrixDecoderFactory

_DecoderT_co = TypeVar("_DecoderT_co", bound=ErrorDecoder, covariant=True)
_DecoderT = TypeVar("_DecoderT", bound=ErrorDecoder)
_InputT = TypeVar("_InputT")
_Parameters = ParamSpec("_Parameters")
_OptionTransform: TypeAlias = Callable[[dict[str, object], frozenset[str]], dict[str, object]]


PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
"""A parity-check matrix or detector error model from which to build an error decoder."""


# Decoder specifications


@dataclasses.dataclass(frozen=True, slots=True, eq=False, repr=False)
class DecoderSpec(Generic[_DecoderT_co]):
    """A typed specification for building a decoder later.

    A specification builds an error decoder, an observable decoder, or both.  A specification
    without an error builder, such as ``decoders.frontier(...)``, is typed ``DecoderSpec[Never]``.
    """

    _helper_name: str
    _builder: Callable[..., _DecoderT_co] | None
    _options: tuple[tuple[str, object], ...]
    _observable_builder: Callable[..., ObservableDecoder] | None = None
    _defaults: tuple[tuple[str, object], ...] | None = None

    def __post_init__(self) -> None:
        """Require at least one way to build a decoder."""
        if self._builder is None and self._observable_builder is None:
            raise ValueError(
                "A decoder specification needs an error builder or an observable builder"
            )

    def __repr__(self) -> str:
        """Show the helper call that reproduces this specification."""
        if self._defaults is None:
            return f"DecoderSpec({self._helper_name!r}, {self._builder!r}, {self._options!r})"
        defaults = dict(self._defaults)
        options = ", ".join(
            f"{name}={value!r}"
            for name, value in self._options
            if name not in defaults or not _is_default_value(value, defaults[name])
        )
        return f"decoders.{self._helper_name}({options})"

    @property
    def options(self) -> dict[str, object]:
        """Return a copy of the construction options, including defaults."""
        return dict(self._options)

    def build(self, pcm_or_dem: PcmOrDem) -> _DecoderT_co:
        """Build an error decoder for a parity-check matrix or detector error model."""
        if self._builder is None:
            raise TypeError(
                f"decoders.{self._helper_name}(...) predicts observable flips but cannot infer"
                " errors, so it cannot build an error decoder.  Call build_observable_decoder(dem)"
                " instead, or pass it where an observable decoder is accepted, such as to"
                " decoders.SinterDecoder"
            )
        return self._builder(pcm_or_dem, **self.options)

    def build_observable_decoder(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build a decoder that predicts the observable flips of a detector error model."""
        if self._observable_builder is not None:
            return _validate_observable_decoder(
                self._observable_builder(dem, **self.options), "A decoder specification"
            )
        return _ErrorsToObservablesDecoder(self.build(dem), dem)

    @property
    def infers_errors(self) -> bool:
        """Whether this specification can build an error decoder."""
        return self._builder is not None

    @property
    def predicts_observables_natively(self) -> bool:
        """Whether this specification has a native observable-decoding mode."""
        return self._observable_builder is not None


@overload
def decoder_spec(
    helper_name: str,
    builder: Callable[Concatenate[PcmOrDem, _Parameters], _DecoderT],
    observable_builder: Callable[..., ObservableDecoder] | None = None,
    /,
    *,
    signature_source: None = None,
    exclude: frozenset[str] = frozenset(),
    option_transform: _OptionTransform | None = None,
    doc: str | None = None,
) -> Callable[_Parameters, DecoderSpec[_DecoderT]]: ...


@overload
def decoder_spec(
    helper_name: str,
    builder: Callable[..., _DecoderT],
    observable_builder: Callable[..., ObservableDecoder] | None = None,
    /,
    *,
    signature_source: Callable[Concatenate[_InputT, _Parameters], object],
    exclude: frozenset[str] = frozenset(),
    option_transform: _OptionTransform | None = None,
    doc: str | None = None,
) -> Callable[_Parameters, DecoderSpec[_DecoderT]]: ...


def decoder_spec(
    helper_name: str,
    builder: Callable[..., _DecoderT],
    observable_builder: Callable[..., ObservableDecoder] | None = None,
    /,
    *,
    signature_source: Callable[..., object] | None = None,
    exclude: frozenset[str] = frozenset(),
    option_transform: _OptionTransform | None = None,
    doc: str | None = None,
) -> Callable[..., DecoderSpec[_DecoderT]]:
    """Create a typed decoder-specification helper from one construction signature.

    The helper accepts the options of ``signature_source`` (by default, ``builder``), excluding
    its first argument and any names in ``exclude``.  Its public documentation is ``doc``, not
    a rewritten builder docstring.
    """
    source = builder if signature_source is None else signature_source
    helper_signature = _get_helper_signature(source, exclude)
    defaults = tuple(
        (name, parameter.default)
        for name, parameter in helper_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    )

    def make_spec(*args: object, **kwargs: object) -> DecoderSpec[_DecoderT]:
        options = _get_spec_options(helper_name, helper_signature, args, kwargs, option_transform)
        return DecoderSpec(
            helper_name,
            builder,
            tuple(options.items()),
            observable_builder,
            defaults,
        )

    decoder_type = inspect.signature(builder, eval_str=True).return_annotation
    if decoder_type is inspect.Signature.empty:
        decoder_type = ErrorDecoder
    _set_helper_metadata(
        make_spec,
        helper_name,
        builder,
        helper_signature,
        cast(Any, DecoderSpec)[decoder_type],
        doc,
    )
    return make_spec


def observable_decoder_spec(
    helper_name: str,
    observable_builder: Callable[
        Concatenate[stim.DetectorErrorModel, _Parameters], ObservableDecoder
    ],
    /,
    *,
    option_transform: _OptionTransform | None = None,
    doc: str | None = None,
) -> Callable[_Parameters, DecoderSpec[Never]]:
    """Create an observable-only decoder-specification helper, as by decoder_spec."""
    helper_signature = _get_helper_signature(observable_builder, frozenset())
    defaults = tuple(
        (name, parameter.default)
        for name, parameter in helper_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    )

    def make_spec(*args: object, **kwargs: object) -> DecoderSpec[Never]:
        options = _get_spec_options(helper_name, helper_signature, args, kwargs, option_transform)
        return DecoderSpec(
            helper_name,
            None,
            tuple(options.items()),
            observable_builder,
            defaults,
        )

    _set_helper_metadata(
        make_spec, helper_name, observable_builder, helper_signature, DecoderSpec[Never], doc
    )
    return make_spec


# Decoder inputs


class ErrorDecoderConstructor(Protocol):
    """Callable that builds an error decoder from a matrix or detector error model."""

    def __call__(self, pcm_or_dem: PcmOrDem, /) -> ErrorDecoder | SupportsDecode:
        """Build an error decoder."""


class ObservableDecoderConstructor(Protocol):
    """Callable that builds an observable decoder from a detector error model."""

    def __call__(self, dem: stim.DetectorErrorModel, /) -> ObservableDecoder:
        """Build an observable decoder."""


class ObservableDecoderCompiler(Protocol):
    """Object that compiles an observable decoder for a detector error model."""

    def compile_decoder_for_dem(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build an observable decoder specialized to one detector error model."""


DeferredErrorDecoderInput: TypeAlias = (
    DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | MatrixDecoderFactory | None
)
"""A deferred input for an error decoder; use a decoder specification for the model the
receiving method constructs."""


ErrorDecoderInput: TypeAlias = DeferredErrorDecoderInput | ErrorDecoder | SupportsDecode
"""An input for a decoder that infers errors; configure it with a decoder specification."""


DeferredDecoderInput: TypeAlias = (
    DeferredErrorDecoderInput
    | ObservableDecoderConstructor
    | DEMDecoderFactory
    | ObservableDecoderCompiler
)
"""A deferred input for error or observable decoding; configure it with a decoder specification
for the receiving method's matrix or detector error model."""


DecoderInput: TypeAlias = ErrorDecoderInput | DeferredDecoderInput | ObservableDecoder
"""An input for error or observable decoding; use a decoder specification to select a backend."""


# Private helpers


def _set_helper_metadata(
    helper: Callable[..., object],
    helper_name: str,
    builder: Callable[..., object],
    signature: inspect.Signature,
    return_annotation: object,
    doc: str | None,
) -> None:
    """Give a generated helper its documented name and derived public signature."""
    helper.__name__ = helper_name
    helper.__qualname__ = helper_name
    helper.__module__ = builder.__module__
    helper.__doc__ = doc
    vars(helper)["__signature__"] = signature.replace(return_annotation=return_annotation)
    helper.__annotations__ = _get_helper_annotations(signature, return_annotation)


def _get_helper_signature(
    source: Callable[..., object], exclude: frozenset[str]
) -> inspect.Signature:
    """Return a construction signature without its matrix/DEM input."""
    parameters = list(inspect.signature(source, eval_str=True).parameters.values())
    if not parameters:
        raise TypeError("A decoder construction signature must accept a matrix or DEM")
    parameters = [parameter for parameter in parameters[1:] if parameter.name not in exclude]
    return inspect.Signature(parameters)


def _get_helper_annotations(
    signature: inspect.Signature, return_annotation: object
) -> dict[str, object]:
    """Return annotations matching a generated helper's public signature."""
    annotations = {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Parameter.empty
    }
    annotations["return"] = return_annotation
    return annotations


def _get_spec_options(
    helper_name: str,
    signature: inspect.Signature,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    option_transform: _OptionTransform | None,
) -> dict[str, object]:
    """Bind and validate the options supplied to a specification helper."""
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as error:
        raise TypeError(f"{helper_name}() {error}") from None
    explicitly_provided: set[str] = set()
    for name, value in bound.arguments.items():
        if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            explicitly_provided.update(value)
        else:
            explicitly_provided.add(name)
    bound.apply_defaults()
    options: dict[str, object] = {}
    for name, value in bound.arguments.items():
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            options.update(value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                "Decoder specification helpers do not support variadic positional arguments"
            )
        else:
            options[name] = value
    _normalize_backend_options(helper_name, options)
    return (
        option_transform(options, frozenset(explicitly_provided))
        if option_transform is not None
        else options
    )


def _normalize_backend_options(helper_name: str, options: dict[str, object]) -> None:
    """Store backend_options as a plain dict, or None if empty, and reject named duplicates.

    A helper whose construction signature has a ``backend_options`` parameter forwards that mapping
    unchecked to its backend.  An option listed by name must be passed by name instead, so that its
    spelling is checked.
    """
    if (backend_options := options.get("backend_options")) is None:
        return
    if not isinstance(backend_options, Mapping):
        raise TypeError(
            f"{helper_name}() backend_options must be a mapping from option names to values, but"
            f" got {type(backend_options).__name__}"
        )
    if duplicates := sorted(name for name in backend_options if name in options):
        raise ValueError(
            f"{helper_name}() lists {', '.join(duplicates)} by name, so pass "
            + ("it" if len(duplicates) == 1 else "them")
            + " directly rather than in backend_options"
        )
    options["backend_options"] = dict(backend_options) or None


def _is_default_value(value: object, default: object) -> bool:
    """Whether an option is a plain copy of its default value."""
    if value is default:
        return True
    plain_types = (bool, int, float, str)
    return type(value) is type(default) and isinstance(value, plain_types) and value == default
