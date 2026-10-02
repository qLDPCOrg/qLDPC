# SPDX-License-Identifier: Apache-2.0

"""Generic typed settings for deferred decoder construction."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import (
    Concatenate,
    Generic,
    Never,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

import stim

from qldpc.math import IntegerArray

from ..adapters.error_decoders import ErrorsToObservablesDecoder as _ErrorsToObservablesDecoder
from ..adapters.observable_decoders import (
    validate_observable_decoder as _validate_observable_decoder,
)
from ..protocols import ErrorDecoder, ObservableDecoder, SupportsDecode

_DecoderT_co = TypeVar("_DecoderT_co", bound=ErrorDecoder, covariant=True)
_DecoderT = TypeVar("_DecoderT", bound=ErrorDecoder)
_InputT = TypeVar("_InputT")
_Parameters = ParamSpec("_Parameters")
_OptionTransform: TypeAlias = Callable[[dict[str, object], frozenset[str]], dict[str, object]]

PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
"""A parity-check matrix or detector error model from which to build an error decoder."""


@dataclasses.dataclass(frozen=True, slots=True, eq=False, repr=False)
class DecoderSpec(Generic[_DecoderT_co]):
    """Deferred, typed construction settings for a decoder.

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
            raise ValueError("A decoder spec needs an error builder or an observable builder")

    @property
    def options(self) -> dict[str, object]:
        """Return a copy of the construction options, including defaults."""
        return dict(self._options)

    @property
    def infers_errors(self) -> bool:
        """Whether this specification can build an error decoder."""
        return self._builder is not None

    def build(self, pcm_or_dem: PcmOrDem) -> _DecoderT_co:
        """Build an error decoder for a parity-check matrix or detector error model."""
        if self._builder is None:
            raise TypeError(
                f"decoders.{self._helper_name}(...) predicts observable flips but cannot infer"
                " errors, so it cannot build an error decoder.  Pass it where an observable decoder"
                " is accepted, such as to decoders.get_observable_decoder or decoders.SinterDecoder"
            )
        return self._builder(pcm_or_dem, **self.options)

    @property
    def predicts_observables_natively(self) -> bool:
        """Whether this specification has a native observable-decoding mode."""
        return self._observable_builder is not None

    def build_observable_decoder(self, dem: stim.DetectorErrorModel) -> ObservableDecoder:
        """Build a decoder that predicts the observable flips of a detector error model."""
        if self._observable_builder is not None:
            return _validate_observable_decoder(
                self._observable_builder(dem, **self.options), "A decoder spec"
            )
        return _ErrorsToObservablesDecoder(self.build(dem), dem)

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


DeferredErrorDecoderInput: TypeAlias = DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | None
"""A decoder= input that builds an error decoder later, for a matrix or detector error model that
the receiving method constructs: decoder settings, an error-decoder constructor, or None to select
the default decoder.  Prebuilt decoders are excluded, because they are tied to one matrix."""

ErrorDecoderInput: TypeAlias = DeferredErrorDecoderInput | ErrorDecoder | SupportsDecode
"""A decoder= input that yields an error decoder: a DeferredErrorDecoderInput, or a prebuilt error
decoder (an ErrorDecoder, or any object whose decode method returns an inferred error)."""

DeferredDecoderInput: TypeAlias = (
    DeferredErrorDecoderInput | ObservableDecoderConstructor | ObservableDecoderCompiler
)
"""A decoder= input that builds an error or observable decoder later, for a matrix or detector
error model that the receiving method constructs: a DeferredErrorDecoderInput, an
observable-decoder constructor, or an observable-decoder compiler such as a SinterDecoder.
Prebuilt decoders are excluded, because they are tied to one matrix or detector error model."""

DecoderInput: TypeAlias = ErrorDecoderInput | DeferredDecoderInput | ObservableDecoder
"""Any decoder= input: an ErrorDecoderInput, a DeferredDecoderInput, or a prebuilt observable
decoder.  The receiving method adapts the decoder that the input yields to what it needs."""


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
) -> Callable[..., DecoderSpec[_DecoderT]]:
    """Create a typed deferred-settings helper from a decoder construction signature."""
    source = builder if signature_source is None else signature_source
    helper_signature = _get_helper_signature(source, exclude)
    defaults = tuple(
        (name, parameter.default)
        for name, parameter in helper_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    )

    @functools.wraps(source)
    def make_spec(*args: object, **kwargs: object) -> DecoderSpec[_DecoderT]:
        bound = helper_signature.bind(*args, **kwargs)
        explicitly_provided = _get_explicit_option_names(bound, helper_signature)
        bound.apply_defaults()
        options = _get_bound_options(bound, helper_signature)
        if option_transform is not None:
            options = option_transform(options, explicitly_provided)
        return DecoderSpec(
            helper_name,
            builder,
            tuple(options.items()),
            observable_builder,
            defaults,
        )

    make_spec.__name__ = helper_name
    make_spec.__qualname__ = helper_name
    make_spec.__module__ = builder.__module__
    make_spec.__doc__ = f"Configure the decoder built by :func:`{builder.__name__}`."
    return_annotation = DecoderSpec[ErrorDecoder]
    vars(make_spec)["__signature__"] = helper_signature.replace(return_annotation=return_annotation)
    make_spec.__annotations__ = _get_helper_annotations(helper_signature, return_annotation)
    return make_spec


def observable_decoder_spec(
    helper_name: str,
    observable_builder: Callable[
        Concatenate[stim.DetectorErrorModel, _Parameters], ObservableDecoder
    ],
    /,
    *,
    option_transform: _OptionTransform | None = None,
) -> Callable[_Parameters, DecoderSpec[Never]]:
    """Create deferred settings for an observable-only decoder."""
    helper_signature = _get_helper_signature(observable_builder, frozenset())
    defaults = tuple(
        (name, parameter.default)
        for name, parameter in helper_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    )

    @functools.wraps(observable_builder)
    def make_spec(*args: object, **kwargs: object) -> DecoderSpec[Never]:
        bound = helper_signature.bind(*args, **kwargs)
        explicitly_provided = _get_explicit_option_names(bound, helper_signature)
        bound.apply_defaults()
        options = _get_bound_options(bound, helper_signature)
        if option_transform is not None:
            options = option_transform(options, explicitly_provided)
        return DecoderSpec(
            helper_name,
            None,
            tuple(options.items()),
            observable_builder,
            defaults,
        )

    make_spec.__name__ = helper_name
    make_spec.__qualname__ = helper_name
    make_spec.__module__ = observable_builder.__module__
    make_spec.__doc__ = (
        f"Configure the observable decoder built by :func:`{observable_builder.__name__}`."
    )
    vars(make_spec)["__signature__"] = helper_signature.replace(
        return_annotation=DecoderSpec[Never]
    )
    make_spec.__annotations__ = _get_helper_annotations(helper_signature, DecoderSpec[Never])
    return make_spec


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


def _get_explicit_option_names(
    bound: inspect.BoundArguments, signature: inspect.Signature
) -> frozenset[str]:
    """Return option names explicitly supplied to a generated helper."""
    names: set[str] = set()
    for name, value in bound.arguments.items():
        if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            names.update(value)
        else:
            names.add(name)
    return frozenset(names)


def _get_bound_options(
    bound: inspect.BoundArguments, signature: inspect.Signature
) -> dict[str, object]:
    """Flatten bound helper arguments into decoder construction options."""
    options: dict[str, object] = {}
    for name, value in bound.arguments.items():
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            options.update(value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            raise TypeError("Decoder settings helpers do not support variadic positional arguments")
        else:
            options[name] = value
    return options


def _is_default_value(value: object, default: object) -> bool:
    """Whether an option is a plain copy of its default value."""
    if value is default:
        return True
    plain_types = (bool, int, float, str)
    return type(value) is type(default) and isinstance(value, plain_types) and value == default
