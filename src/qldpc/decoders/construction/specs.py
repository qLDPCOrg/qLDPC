# SPDX-License-Identifier: Apache-2.0

"""Generic typed settings for deferred decoder construction."""

from __future__ import annotations

import dataclasses
import inspect
import textwrap
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
from .factories import _DEMDecoderFactory, _MatrixDecoderFactory

_DecoderT_co = TypeVar("_DecoderT_co", bound=ErrorDecoder, covariant=True)
_DecoderT = TypeVar("_DecoderT", bound=ErrorDecoder)
_InputT = TypeVar("_InputT")
_Parameters = ParamSpec("_Parameters")
_OptionTransform: TypeAlias = Callable[[dict[str, object], frozenset[str]], dict[str, object]]


PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
"""A parity-check matrix or detector error model from which to build an error decoder."""


# Decoder settings


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
                self._observable_builder(dem, **self.options), "A decoder spec"
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
    returns: str | None = None,
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
    returns: str | None = None,
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
    returns: str | None = None,
) -> Callable[..., DecoderSpec[_DecoderT]]:
    """Create a typed deferred-settings helper from a decoder construction signature.

    The helper takes the keyword options of ``signature_source`` (by default, ``builder``) other
    than its first argument and the ``exclude`` names, and is documented by the docstring of
    ``builder``, with its summary verb "Build" replaced by "Configure" and its Returns section
    replaced by ``returns``, which describes the settings.
    """
    source = builder if signature_source is None else signature_source
    helper_signature = _get_helper_signature(source, exclude)
    defaults = tuple(
        (name, parameter.default)
        for name, parameter in helper_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    )

    def make_spec(*args: object, **kwargs: object) -> DecoderSpec[_DecoderT]:
        bound = _bind_helper_arguments(helper_name, helper_signature, args, kwargs)
        explicitly_provided = _get_explicit_option_names(bound, helper_signature)
        bound.apply_defaults()
        options = _get_bound_options(bound, helper_signature)
        _normalize_backend_options(helper_name, options)
        if option_transform is not None:
            options = option_transform(options, explicitly_provided)
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
        returns,
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
    returns: str | None = None,
) -> Callable[_Parameters, DecoderSpec[Never]]:
    """Create deferred settings for an observable-only decoder, documented as by decoder_spec."""
    helper_signature = _get_helper_signature(observable_builder, frozenset())
    defaults = tuple(
        (name, parameter.default)
        for name, parameter in helper_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    )

    def make_spec(*args: object, **kwargs: object) -> DecoderSpec[Never]:
        bound = _bind_helper_arguments(helper_name, helper_signature, args, kwargs)
        explicitly_provided = _get_explicit_option_names(bound, helper_signature)
        bound.apply_defaults()
        options = _get_bound_options(bound, helper_signature)
        _normalize_backend_options(helper_name, options)
        if option_transform is not None:
            options = option_transform(options, explicitly_provided)
        return DecoderSpec(
            helper_name,
            None,
            tuple(options.items()),
            observable_builder,
            defaults,
        )

    _set_helper_metadata(
        make_spec, helper_name, observable_builder, helper_signature, DecoderSpec[Never], returns
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
    DecoderSpec[ErrorDecoder] | ErrorDecoderConstructor | _MatrixDecoderFactory | None
)
"""A decoder= input that builds an error decoder later, for a matrix or detector error model that
the receiving method constructs: decoder settings, an error-decoder constructor, a matrix-only
factory, or None to select the default decoder.  Prebuilt decoders are excluded, because they are
tied to one matrix."""


ErrorDecoderInput: TypeAlias = DeferredErrorDecoderInput | ErrorDecoder | SupportsDecode
"""A decoder= input that yields an error decoder: a DeferredErrorDecoderInput, or a prebuilt error
decoder (an ErrorDecoder, or any object whose decode method returns an inferred error)."""


DeferredDecoderInput: TypeAlias = (
    DeferredErrorDecoderInput
    | ObservableDecoderConstructor
    | _DEMDecoderFactory
    | ObservableDecoderCompiler
)
"""A decoder= input that builds an error or observable decoder later, for a matrix or detector
error model that the receiving method constructs: a DeferredErrorDecoderInput, an explicit
observable-decoder factory, or an observable-decoder compiler such as a SinterDecoder.  A bare
constructor is treated as an error-decoder constructor.  Prebuilt decoders are excluded, because
they are tied to one matrix or detector error model."""


DecoderInput: TypeAlias = ErrorDecoderInput | DeferredDecoderInput | ObservableDecoder
"""Any decoder= input: an ErrorDecoderInput, a DeferredDecoderInput, or a prebuilt observable
decoder.  The receiving method adapts the decoder that the input yields to what it needs."""


# Private helpers


def _set_helper_metadata(
    helper: Callable[..., object],
    helper_name: str,
    builder: Callable[..., object],
    signature: inspect.Signature,
    return_annotation: object,
    returns: str | None,
) -> None:
    """Give a generated helper the public name, docs, and signature of a settings helper.

    The docstring of the builder documents the options of the helper.  Unlike functools.wraps, this
    does not copy the attributes of a decoder class that provides the signature.
    """
    helper.__name__ = helper_name
    helper.__qualname__ = helper_name
    helper.__module__ = builder.__module__
    helper.__doc__ = _get_helper_docstring(builder.__doc__, signature, returns)
    vars(helper)["__signature__"] = signature.replace(return_annotation=return_annotation)
    helper.__annotations__ = _get_helper_annotations(signature, return_annotation)


def _get_helper_docstring(
    docstring: str | None, signature: inspect.Signature, returns: str | None = None
) -> str | None:
    """Adapt the docstring of a decoder builder to its settings helper.

    A builder documents the matrix or detector error model that it decodes, which is instead passed
    to DecoderSpec.build, and may document keyword arguments that it forwards to a decoder class.
    The arguments that the helper does not accept are dropped.  A builder summary that starts with
    "Build" starts with "Configure" instead, and if ``returns`` is provided, it replaces the
    description of the built decoder in the Returns section.
    """
    if docstring is None:
        return None
    lines = docstring.splitlines()
    if lines and lines[0].startswith("Build "):
        lines[0] = "Configure " + lines[0].removeprefix("Build ")
    kept_lines: list[str] = []
    section: str | None = None
    section_indent = 0
    entry_indent: int | None = None
    keep_entry = True
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if section is not None and (not line.strip() or indent <= section_indent):
            section = None
        if line.strip() == "Args:":
            section, section_indent, entry_indent = "Args", indent, None
        elif line.strip() == "Returns:" and returns is not None:
            section, section_indent = "Returns", indent
            body_indent = " " * (indent + 4)
            kept_lines.append(line)
            kept_lines.extend(
                textwrap.wrap(
                    returns, width=100, initial_indent=body_indent, subsequent_indent=body_indent
                )
            )
            continue
        elif section == "Returns":
            continue
        elif section == "Args":
            entry_indent = indent if entry_indent is None else entry_indent
            if indent == entry_indent:
                name = line.strip().split(":", maxsplit=1)[0].lstrip("*")
                keep_entry = name in signature.parameters
            if not keep_entry:
                continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


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


def _bind_helper_arguments(
    helper_name: str,
    signature: inspect.Signature,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> inspect.BoundArguments:
    """Bind the arguments of a generated helper, naming the helper in any error."""
    try:
        return signature.bind(*args, **kwargs)
    except TypeError as error:
        raise TypeError(f"{helper_name}() {error}") from None


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
