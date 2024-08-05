from __future__ import annotations

import numpy as np
from numpy.fft import fft, ifft
import math

from abc import ABC
from dataclasses import dataclass
from collections.abc import Sequence
from numbers import Number

##############################################################################
## Syntax for all generalized type systems                                  ##
##############################################################################


class TTExpr(ABC):
    """
    Abstract base class of all expressions/terms for our 
    generalized type system.
    """
    pass


@dataclass
class TTExprIdent(TTExpr):
    """
    Class of terms which are identifiers.
    """
    x: str

    def __repr__(self) -> str:
        return self.x


@dataclass
class TTExprConst(TTExpr):
    """
    Class of terms which are constants.
    """
    c: str

    def __repr__(self) -> str:
        return self.c


@dataclass
class TTExprApp(TTExpr):
    """
    Class of terms applied to other terms, this is the elimination rule 
    for product types.
    """
    m: TTExpr
    n: TTExpr

    def __repr__(self) -> str:
        return f"({self.m} {self.n})"


@dataclass
class TTExprAbs(TTExpr):
    """
    Class of constructors for product types; lambda abstraction.
    """
    x: str
    t: TTExpr
    m: TTExpr

    def __repr__(self) -> str:
        return f"(\\ {self.x} : {self.t} . {self.m})"


@dataclass
class TTExprProd(TTExpr):
    """
    Class of product types, which generalize the notion of conditionals 
    as well as functions.
    """
    x: str
    t: TTExpr
    m: TTExpr

    def __repr__(self) -> str:
        return f"(|~| {self.x} : {self.t} . {self.m})"


##############################################################################
## Holographic Reduced Representations                                      ##
##############################################################################

_default_large = 0.8
_default_small = 0.2
@dataclass
class HRR(Sequence):
    """
    Holographic reduced representation.
    """
    v: np.ndarray
    large: float = _default_large
    small: float = _default_small

    @staticmethod
    def from_data(
        data: np.ndarray | HRR, 
        large: float | None = None, 
        small: float | None = None
    ) -> HRR:
        """ 
        Initialize a holographic reduced representation with 
        vector with provided data.
        """
        h = HRR(np.array([]))

        if large is not None:
            h.large = large
        if small is not None:
            h.small = small
        if isinstance(data, HRR):
            h.v = data.v
        else:
            h.v = data

        return h

    @staticmethod
    def from_size(
        size: int, 
        large: float | None = None,
        small: float | None = None
    ) -> HRR:
        """
        Initialize a holographic reduced representation vector 
        with normal distribution of elements, with a given size 
        `size`.
        """
        h = HRR(np.array([]))

        if large is not None:
            h.large = large
        if small is not None:
            h.small = small

        sd = 1.0 / math.sqrt(size)
        h.v = np.random.normal(scale=sd, size=size)
        h.v /= np.linalg.norm(self.v)
        return h

    @staticmethod
    def zeros(
        size: int,
        large: float | None = None,
        small: float | None = None
    ) -> HRR:
        """
        Initialize a holographic reduced representation vector 
        with normal distribution of elements, with a given size `size`,
        all zero'd out.
        """
        h = HRR(np.array([]))

        if large is not None:
            h.large = large
        if small is not None:
            h.small = small

        h.v = np.zeros(size)
        return h

    def associate(self, rhs: HRR | np.ndarray) -> HRR:
        """
        Associate two vectors.
        """
        if isinstance(rhs, HRR):
            return HRR.from_data(ifft(fft(self.v) * fft(other.v)).real)
        else:
            return HRR.from_data(self.v * rhs)

    def __mul__(self, rhs: HRR | np.ndarray) -> HRR:
        return self.associate(rhs)

    def __rmul__(self, lhs: HRR | np.ndarray) -> HRR:
        return self * lhs

    def fractional_bind(self, exp: Number) -> HRR:
        """
        Fractional binding.
        """
        return HRR(ifft(fft(self.v) ** exponent).real)

    def union(self, rhs: HRR | np.ndarray) -> HRR:
        """ 
        Superpose two vectors.
        """
        if isinstance(rhs, HRR):
            return rhs + self.v
        else:
            return HRR.from_data(rhs + self.v)

    def __neg__(self) -> HRR:
        return HRR(-self.v)

    def __sub__(self, other: HRR) -> HRR:
        return HRR(self.v - other.v)

    def __add__(self, rhs: HRR | np.ndarray) -> HRR:
        return self.union(rhs)

    def __pow__(self, exp: Number) -> HRR:
        return self.fractional_bind(exp)

    def __getitem__(self, i: int) -> float:
        return self.v[i]

    def __len__(self) -> int:
        return self.v.size

