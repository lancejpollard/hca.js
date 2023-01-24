import * as tf from '@tensorflow/tfjs'
import { Tensor } from '@tensorflow/tfjs'

type Idx2SignInputType = {
  neg: boolean
}

type ClampAbsInputType = {
  eps: number
}

type SabsInputType = {
  eps: number
}

type _WeightedMidpointInputType = {
  dim: number
  keepdim: boolean
  lincomb: boolean
  posweight: boolean
  reducedim: OptionalListInt
  weights?: Tensor
}

type WeightedMidpointInputType = {
  dim: number
  keepdim: boolean
  lincomb: boolean
  posweight: boolean
  reducedim: OptionalListInt
  weights?: Tensor
}

type _AntipodeInputType = {
  dim: number
}

type AntipodeInputType = {
  dim: number
}

type _InvSprojInputType = {
  dim: number
}

type InvSprojInputType = {
  dim: number
}

type _SprojInputType = {
  dim: number
}

type SprojInputType = {
  dim: number
}

type _Egrad2RgradInputType = {
  dim: number
}

type _ParallelTransport0BackInputType = {
  dim: number
}

type ParallelTransport0BackInputType = {
  dim: number
}

type _ParallelTransport0InputType = {
  dim: number
}

type _ParallelTransportInputType = {
  dim: number
}

type _Dist2PlaneInputType = {
  dim: number
  keepdim: boolean
  scaled: boolean
  signed: boolean
}

type _MobiusPointwiseMulInputType = {
  dim: number
}

type _MobiusMatvecInputType = {
  dim: number
}

type _Logmap0InputType = {
  dim: number
}

type _LogmapInputType = {
  dim: number
}

type _GeodesicUnitInputType = {
  dim: number
}

type _Expmap0InputType = {
  dim: number
}

type _ExpmapInputType = {
  dim: number
}

type _GeodesicInputType = {
  dim: number
}

type _Dist0InputType = {
  dim: number
  keepdim: boolean
}

type _DistInputType = {
  dim: number
  keepdim: boolean
}

type _MobiusScalarMulInputType = {
  dim: number
}

type _MobiusCosubInputType = {
  dim: number
}

type _MobiusCoaddInputType = {
  dim: number
}

type _GyrationInputType = {
  dim: number
}

type _MobiusSubInputType = {
  dim: number
}

type _MobiusAddInputType = {
  dim: number
}

type _NormInputType = {
  dim: number
  keepdim: boolean
}

type _InnerInputType = {
  dim: number
  keepdim: boolean
}

type _LambdaXInputType = {
  dim: number
  keepdim: boolean
}

type _ProjectInputType = {
  dim: number
  eps: number
}

type SinKZeroTaylorInputType = {
  order: number
}

type ArsinKZeroTaylorInputType = {
  order: number
}

type ArtanKZeroTaylorInputType = {
  order: number
}

type TanKZeroTaylorInputType = {
  order: number
}

/* `
:math:`\kappa`-Stereographic math module.

The functions for the mathematics in gyrovector spaces are taken from the
following resources:

    [1] Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic
           neural networks." Advances in neural information processing systems.
           2018.
    [2] Bachmann, Gregor, Gary Bécigneul, and Octavian-Eugen Ganea. "Constant
           Curvature Graph Convolutional Networks." arXiv preprint
           arXiv:1911.05076 (2019).
    [3] Skopek, Ondrej, Octavian-Eugen Ganea, and Gary Bécigneul.
           "Mixed-curvature Variational Autoencoders." arXiv preprint
           arXiv:1911.08411 (2019).
    [4] Ungar, Abraham A. Analytic hyperbolic geometry: Mathematical
           foundations and applications. World Scientific, 2005.
    [5] Albert, Ungar Abraham. Barycentric calculus in Euclidean and
           hyperbolic geometry: A comparative introduction. World Scientific,
           2010.
` */
function tanh(x: Tensor) {
  return tf.clipByValue(x, -15, 15).tanh()
}

function artanh(x: Tensor) {
  x = tf.clipByValue(x, -1 + 1e-7, 1 - 1e-7)
  return tf.add(1, x).sub(tf.sub(1, x).log()).mul(0.5)
}

function arsinh(x: Tensor): Tensor {
  return tf
    .clipByValue(
      x.add(tf.sqrt(tf.add(1, x.pow(2)))),
      1e-15,
      Number.MAX_SAFE_INTEGER,
    )
    .log()
    .cast(x.dtype)
}

function absZeroGrad(x: Tensor) {
  return x.mul(sign(x))
}

function prod(items) {
  return functools.reduce(operator.mul, items, 1)
}

function sign(x: Tensor) {
  return x.sign().add(0.5)
}

function sabs(x: Tensor, { eps = 1e-15 }: SabsInputType) {
  return x.abs().add(eps)
}

function clampAbs(x: Tensor, { eps = 1e-15 }: ClampAbsInputType) {
  let s = sign(x)

  return s.mul(sabs(x, { eps: eps }))
}

function tanKZeroTaylor(
  x: Tensor,
  k: Tensor,
  { order = -1 }: TanKZeroTaylorInputType,
) {
  if (order == 0) {
    return x
  }

  k = absZeroGrad(k)

  if (order == -1 || order == 5) {
    return x.add(
      tf
        .mul(1 / 3, k)
        .mul(x)
        .pow(3)
        .add(
          tf
            .mul(2 / 15, k)
            .pow(2)
            .mul(x.pow(5)),
        )
        .add(
          tf
            .mul(17 / 315, k)
            .pow(3)
            .mul(x.pow(7)),
        )
        .add(
          tf
            .mul(62 / 2835, k)
            .pow(4)
            .mul(x.pow(9)),
        )
        .add(
          tf
            .mul(1382 / 155925, k)
            .pow(5)
            .mul(x.pow(11)),
        ),
    )
  } else if (order == 1) {
    return x + (1 / 3) * k * x ** 3
  } else if (order == 2) {
    return x + (1 / 3) * k * x ** 3 + (2 / 15) * k ** 2 * x ** 5
  } else if (order == 3) {
    return (
      x +
      (1 / 3) * k * x ** 3 +
      (2 / 15) * k ** 2 * x ** 5 +
      (17 / 315) * k ** 3 * x ** 7
    )
  } else if (order == 4) {
    return (
      x +
      (1 / 3) * k * x ** 3 +
      (2 / 15) * k ** 2 * x ** 5 +
      (17 / 315) * k ** 3 * x ** 7 +
      (62 / 2835) * k ** 4 * x ** 9
    )
  } else {
    throw new runtimeError('order not in [-1, 5]')
  }
}
function artanKZeroTaylor(
  x: Tensor,
  k: Tensor,
  { order = -1 }: ArtanKZeroTaylorInputType,
) {
  if (order == 0) {
    return x
  }
  k = absZeroGrad(k)
  if (order == -1 || order == 5) {
    return (
      x -
      (1 / 3) * k * x ** 3 +
      (1 / 5) * k ** 2 * x ** 5 -
      (1 / 7) * k ** 3 * x ** 7 +
      (1 / 9) * k ** 4 * x ** 9 -
      (1 / 11) * k ** 5 * x ** 11
    )
  } else if (order == 1) {
    return x - (1 / 3) * k * x ** 3
  } else if (order == 2) {
    return x - (1 / 3) * k * x ** 3 + (1 / 5) * k ** 2 * x ** 5
  } else if (order == 3) {
    return (
      x -
      (1 / 3) * k * x ** 3 +
      (1 / 5) * k ** 2 * x ** 5 -
      (1 / 7) * k ** 3 * x ** 7
    )
  } else if (order == 4) {
    return (
      x -
      (1 / 3) * k * x ** 3 +
      (1 / 5) * k ** 2 * x ** 5 -
      (1 / 7) * k ** 3 * x ** 7 +
      (1 / 9) * k ** 4 * x ** 9
    )
  } else {
    throw new runtimeError('order not in [-1, 5]')
  }
}
function arsinKZeroTaylor(
  x: Tensor,
  k: Tensor,
  { order = -1 }: ArsinKZeroTaylorInputType,
) {
  if (order == 0) {
    return x
  }
  k = absZeroGrad(k)
  if (order == -1 || order == 5) {
    return (
      x +
      (k * x ** 3) / 6 +
      (3 / 40) * k ** 2 * x ** 5 +
      (5 / 112) * k ** 3 * x ** 7 +
      (35 / 1152) * k ** 4 * x ** 9 +
      (63 / 2816) * k ** 5 * x ** 11
    )
  } else if (order == 1) {
    return x + (k * x ** 3) / 6
  } else if (order == 2) {
    return x + (k * x ** 3) / 6 + (3 / 40) * k ** 2 * x ** 5
  } else if (order == 3) {
    return (
      x +
      (k * x ** 3) / 6 +
      (3 / 40) * k ** 2 * x ** 5 +
      (5 / 112) * k ** 3 * x ** 7
    )
  } else if (order == 4) {
    return (
      x +
      (k * x ** 3) / 6 +
      (3 / 40) * k ** 2 * x ** 5 +
      (5 / 112) * k ** 3 * x ** 7 +
      (35 / 1152) * k ** 4 * x ** 9
    )
  } else {
    throw new runtimeError('order not in [-1, 5]')
  }
}
function sinKZeroTaylor(
  x: Tensor,
  k: Tensor,
  { order = -1 }: SinKZeroTaylorInputType,
) {
  if (order == 0) {
    return x
  }
  k = absZeroGrad(k)
  if (order == -1 || order == 5) {
    return (
      x -
      (k * x ** 3) / 6 +
      (k ** 2 * x ** 5) / 120 -
      (k ** 3 * x ** 7) / 5040 +
      (k ** 4 * x ** 9) / 362880 -
      (k ** 5 * x ** 11) / 39916800
    )
  } else if (order == 1) {
    return x - (k * x ** 3) / 6
  } else if (order == 2) {
    return x - (k * x ** 3) / 6 + (k ** 2 * x ** 5) / 120
  } else if (order == 3) {
    return (
      x -
      (k * x ** 3) / 6 +
      (k ** 2 * x ** 5) / 120 -
      (k ** 3 * x ** 7) / 5040
    )
  } else if (order == 4) {
    return (
      x -
      (k * x ** 3) / 6 +
      (k ** 2 * x ** 5) / 120 -
      (k ** 3 * x ** 7) / 5040 +
      (k ** 4 * x ** 9) / 362880
    )
  } else {
    throw new runtimeError('order not in [-1, 5]')
  }
}
function tanK(x: Tensor, k: Tensor) {
  let kSign
  let zero
  let kZero
  let kSqrt
  let scaledX
  let tanKNonzero
  kSign = k.sign()
  zero = torch.zeros([], { device: k.device, dtype: k.dtype })
  kZero = k.isclose(zero)
  kSign = torch.maskedFill(kSign, kZero, zero.to(kSign.dtype))
  if (torch.all(kZero)) {
    return tanKZeroTaylor(x, k, { order: 1 })
  }
  kSqrt = sabs(k).sqrt()
  scaledX = x * kSqrt
  if (torch.all(kSign.lt(0))) {
    return kSqrt.reciprocal() * tanh(scaledX)
  } else if (torch.all(kSign.gt(0))) {
    return kSqrt.reciprocal() * scaledX.clampMax(1e38).tan()
  } else {
    tanKNonzero =
      torch.where(
        kSign.gt(0),
        scaledX.clampMax(1e38).tan(),
        tanh(scaledX),
      ) * kSqrt.reciprocal()
    return torch.where(
      kZero,
      tanKZeroTaylor(x, k, { order: 1 }),
      tanKNonzero,
    )
  }
}
function artanK(x: Tensor, k: Tensor) {
  let kSign
  let zero
  let kZero
  let kSqrt
  let scaledX
  let artanKNonzero
  kSign = k.sign()
  zero = torch.zeros([], { device: k.device, dtype: k.dtype })
  kZero = k.isclose(zero)
  kSign = torch.maskedFill(kSign, kZero, zero.to(kSign.dtype))
  if (torch.all(kZero)) {
    return artanKZeroTaylor(x, k, { order: 1 })
  }
  kSqrt = sabs(k).sqrt()
  scaledX = x * kSqrt
  if (torch.all(kSign.lt(0))) {
    return kSqrt.reciprocal() * artanh(scaledX)
  } else if (torch.all(kSign.gt(0))) {
    return kSqrt.reciprocal() * scaledX.atan()
  } else {
    artanKNonzero =
      torch.where(kSign.gt(0), scaledX.atan(), artanh(scaledX)) *
      kSqrt.reciprocal()
    return torch.where(
      kZero,
      artanKZeroTaylor(x, k, { order: 1 }),
      artanKNonzero,
    )
  }
}
function arsinK(x: Tensor, k: Tensor) {
  let kSign
  let zero
  let kZero
  let kSqrt
  let scaledX
  let arsinKNonzero
  kSign = k.sign()
  zero = torch.zeros([], { device: k.device, dtype: k.dtype })
  kZero = k.isclose(zero)
  kSign = torch.maskedFill(kSign, kZero, zero.to(kSign.dtype))
  if (torch.all(kZero)) {
    return arsinKZeroTaylor(x, k)
  }
  kSqrt = sabs(k).sqrt()
  scaledX = x * kSqrt
  if (torch.all(kSign.lt(0))) {
    return kSqrt.reciprocal() * arsinh(scaledX)
  } else if (torch.all(kSign.gt(0))) {
    return kSqrt.reciprocal() * scaledX.asin()
  } else {
    arsinKNonzero =
      torch.where(
        kSign.gt(0),
        scaledX.clamp(-1 + 1e-7, 1 - 1e-7).asin(),
        arsinh(scaledX),
      ) * kSqrt.reciprocal()
    return torch.where(
      kZero,
      arsinKZeroTaylor(x, k, { order: 1 }),
      arsinKNonzero,
    )
  }
}
function sinK(x: Tensor, k: Tensor) {
  let kSign
  let zero
  let kZero
  let kSqrt
  let scaledX
  let sinKNonzero
  kSign = k.sign()
  zero = torch.zeros([], { device: k.device, dtype: k.dtype })
  kZero = k.isclose(zero)
  kSign = torch.maskedFill(kSign, kZero, zero.to(kSign.dtype))
  if (torch.all(kZero)) {
    return sinKZeroTaylor(x, k)
  }
  kSqrt = sabs(k).sqrt()
  scaledX = x * kSqrt
  if (torch.all(kSign.lt(0))) {
    return kSqrt.reciprocal() * torch.sinh(scaledX)
  } else if (torch.all(kSign.gt(0))) {
    return kSqrt.reciprocal() * scaledX.sin()
  } else {
    sinKNonzero =
      torch.where(kSign.gt(0), scaledX.sin(), torch.sinh(scaledX)) *
      kSqrt.reciprocal()
    return torch.where(
      kZero,
      sinKZeroTaylor(x, k, { order: 1 }),
      sinKNonzero,
    )
  }
}
function project(x: Tensor, k: Tensor, { dim = -1, eps = -1 }) {
  /* `
    Safe projection on the manifold for numerical stability.

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension to compute norm
    eps : float
        stability parameter, uses default for dtype if not provided

    Returns
    -------
    tensor
        projected vector on the manifold
    ` */
  return _project(x, k, dim, eps)
}
function _project(x, k, { dim = -1, eps = -1.0 }: _ProjectInputType) {
  let maxnorm
  let norm
  let cond
  let projected
  if (eps < 0) {
    if (x.dtype == torch.float32) {
      eps = 4e-3
    } else {
      eps = 1e-5
    }
  }
  maxnorm = (1 - eps) / sabs(k) ** 0.5
  maxnorm = torch.where(k.lt(0), maxnorm, k.newFull([], 1e15))
  norm = x.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  cond = norm > maxnorm
  projected = (x / norm) * maxnorm
  return torch.where(cond, projected, x)
}
function lambdaX(x: Tensor, k: Tensor, { keepdim = false, dim = -1 }) {
  /* `
    Compute the conformal factor :math:`\lambda^\kappa_x` for a point on the ball.

    .. math::
        \lambda^\kappa_x = \frac{2}{1 + \kappa \|x\|_2^2}

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        conformal factor
    ` */
  return _lambdaX(x, k, { dim: dim, keepdim: keepdim })
}
function _lambdaX(
  x: Tensor,
  k: Tensor,
  { keepdim = false, dim = -1 }: _LambdaXInputType,
) {
  return (
    2 /
    (1 + k * x.pow(2).sum({ dim: dim, keepdim: keepdim })).clampMin(
      1e-15,
    )
  )
}
function inner(
  x: Tensor,
  u: Tensor,
  v: Tensor,
  k,
  { keepdim = false, dim = -1 },
) {
  /* `
    Compute inner product for two vectors on the tangent space w.r.t Riemannian metric on the Poincare ball.

    .. math::

        \langle u, v\rangle_x = (\lambda^\kappa_x)^2 \langle u, v \rangle

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on Poincare ball
    v : tensor
        tangent vector to :math:`x` on Poincare ball
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    ` */
  return _inner(x, u, v, k, { dim: dim, keepdim: keepdim })
}
function _inner(
  x: Tensor,
  u: Tensor,
  v: Tensor,
  k: Tensor,
  { keepdim = false, dim = -1 }: _InnerInputType,
) {
  return (
    _lambdaX(x, k, { dim: dim, keepdim: true }) ** 2 *
    (u * v).sum({ dim: dim, keepdim: keepdim })
  )
}
function norm(
  x: Tensor,
  u: Tensor,
  k: Tensor,
  { keepdim = false, dim = -1 },
) {
  /* `
    Compute vector norm on the tangent space w.r.t Riemannian metric on the Poincare ball.

    .. math::

        \|u\|_x = \lambda^\kappa_x \|u\|_2

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    u : tensor
        tangent vector to :math:`x` on Poincare ball
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        norm of vector
    ` */
  return _norm(x, u, k, { dim: dim, keepdim: keepdim })
}
function _norm(
  x: Tensor,
  u: Tensor,
  k: Tensor,
  { keepdim = false, dim = -1 }: _NormInputType,
) {
  return (
    _lambdaX(x, k, { dim: dim, keepdim: keepdim }) *
    u.norm({ dim: dim, keepdim: keepdim, p: 2 })
  )
}
function mobiusAdd(x: Tensor, y: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the Möbius gyrovector addition.

    .. math::

        x \oplus_\kappa y =
        \frac{
            (1 - 2 \kappa \langle x, y\rangle - \kappa \|y\|^2_2) x +
            (1 + \kappa \|x\|_2^2) y
        }{
            1 - 2 \kappa \langle x, y\rangle + \kappa^2 \|x\|^2_2 \|y\|^2_2
        }

    .. plot:: plots/extended/stereographic/mobius_add.py

    In general this operation is not commutative:

    .. math::

        x \oplus_\kappa y \ne y \oplus_\kappa x

    But in some cases this property holds:

    * zero vector case

    .. math::

        \mathbf{0} \oplus_\kappa x = x \oplus_\kappa \mathbf{0}

    * zero curvature case that is same as Euclidean addition

    .. math::

        x \oplus_0 y = y \oplus_0 x

    Another useful property is so called left-cancellation law:

    .. math::

        (-x) \oplus_\kappa (x \oplus_\kappa y) = y

    Parameters
    ----------
    x : tensor
        point on the manifold
    y : tensor
        point on the manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius addition
    ` */
  return _mobiusAdd(x, y, k, { dim: dim })
}
function _mobiusAdd(
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { dim = -1 }: _MobiusAddInputType,
) {
  let x2
  let y2
  let xy
  let num
  let denom
  x2 = x.pow(2).sum({ dim: dim, keepdim: true })
  y2 = y.pow(2).sum({ dim: dim, keepdim: true })
  xy = (x * y).sum({ dim: dim, keepdim: true })
  num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
  denom = 1 - 2 * k * xy + k ** 2 * x2 * y2
  return num / denom.clampMin(1e-15)
}
function mobiusSub(x: Tensor, y: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the Möbius gyrovector subtraction.

    The Möbius subtraction can be represented via the Möbius addition as
    follows:

    .. math::

        x \ominus_\kappa y = x \oplus_\kappa (-y)

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius subtraction
    ` */
  return _mobiusSub(x, y, k, { dim: dim })
}
function _mobiusSub(
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { dim = -1 }: _MobiusSubInputType,
) {
  return _mobiusAdd(x, -y, k, { dim: dim })
}
function gyration(
  a: Tensor,
  b: Tensor,
  u: Tensor,
  k: Tensor,
  { dim = -1 },
) {
  /* `
    Compute the gyration of :math:`u` by :math:`[a,b]`.

    The gyration is a special operation of gyrovector spaces. The gyrovector
    space addition operation :math:`\oplus_\kappa` is not associative (as
    mentioned in :func:`mobius_add`), but it is gyroassociative, which means

    .. math::

        u \oplus_\kappa (v \oplus_\kappa w)
        =
        (u\oplus_\kappa v) \oplus_\kappa \operatorname{gyr}[u, v]w,

    where

    .. math::

        \operatorname{gyr}[u, v]w
        =
        \ominus (u \oplus_\kappa v) \oplus (u \oplus_\kappa (v \oplus_\kappa w))

    We can simplify this equation using the explicit formula for the Möbius
    addition [1]. Recall,

    .. math::

        A = - \kappa^2 \langle u, w\rangle \langle v, v\rangle
            - \kappa \langle v, w\rangle
            + 2 \kappa^2 \langle u, v\rangle \langle v, w\rangle\\
        B = - \kappa^2 \langle v, w\rangle \langle u, u\rangle
            + \kappa \langle u, w\rangle\\
        D = 1 - 2 \kappa \langle u, v\rangle
            + \kappa^2 \langle u, u\rangle \langle v, v\rangle\\

        \operatorname{gyr}[u, v]w = w + 2 \frac{A u + B v}{D}.

    Parameters
    ----------
    a : tensor
        first point on manifold
    b : tensor
        second point on manifold
    u : tensor
        vector field for operation
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of automorphism

    References
    ----------
    [1]  A. A. Ungar (2009), A Gyrovector Space Approach to Hyperbolic Geometry
    ` */
  return _gyration(a, b, u, k, { dim: dim })
}
function _gyration(
  u: Tensor,
  v: Tensor,
  w: Tensor,
  k: Tensor,
  { dim = -1 }: _GyrationInputType,
) {
  let u2
  let v2
  let uv
  let uw
  let vw
  let k2
  let a
  let b
  let d
  u2 = u.pow(2).sum({ dim: dim, keepdim: true })
  v2 = v.pow(2).sum({ dim: dim, keepdim: true })
  uv = (u * v).sum({ dim: dim, keepdim: true })
  uw = (u * w).sum({ dim: dim, keepdim: true })
  vw = (v * w).sum({ dim: dim, keepdim: true })
  k2 = k ** 2
  a = -K2 * uw * v2 - k * vw + 2 * k2 * uv * vw
  b = -K2 * vw * u2 + k * uw
  d = 1 - 2 * k * uv + k2 * u2 * v2
  return w + (2 * (a * u + b * v)) / d.clampMin(1e-15)
}
function mobiusCoadd(x: Tensor, y: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the Möbius gyrovector coaddition.

    The addition operation :math:`\oplus_\kappa` is neither associative, nor
    commutative. In contrast, the coaddition :math:`\boxplus_\kappa` (or
    cooperation) is an associative operation that is defined as follows.

    .. math::

        a \boxplus_\kappa b
        =
        b \boxplus_\kappa a
        =
        a\operatorname{gyr}[a, -b]b\\
        = \frac{
            (1 + \kappa \|y\|^2_2) x + (1 + \kappa \|x\|_2^2) y
            }{
            1 + \kappa^2 \|x\|^2_2 \|y\|^2_2
        },

    where :math:`\operatorname{gyr}[a, b]v = \ominus_\kappa (a \oplus_\kappa b)
    \oplus_\kappa (a \oplus_\kappa (b \oplus_\kappa v))`

    The following right cancellation property holds

    .. math::

        (a \boxplus_\kappa b) \ominus_\kappa b = a\\
        (a \oplus_\kappa b) \boxminus_\kappa b = a

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius coaddition

    ` */
  return _mobiusCoadd(x, y, k, { dim: dim })
}
function _mobiusCoadd(
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { dim = -1 }: _MobiusCoaddInputType,
) {
  return _mobiusAdd(x, _gyration(x, -y, y, { dim: dim, k: k }), k, {
    dim: dim,
  })
}
function mobiusCosub(x: Tensor, y: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the Möbius gyrovector cosubtraction.

    The Möbius cosubtraction is defined as follows:

    .. math::

        a \boxminus_\kappa b = a \boxplus_\kappa -b

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius cosubtraction

    ` */
  return _mobiusCosub(x, y, k, { dim: dim })
}
function _mobiusCosub(
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { dim = -1 }: _MobiusCosubInputType,
) {
  return _mobiusCoadd(x, -y, k, { dim: dim })
}
function mobiusScalarMul(
  r: Tensor,
  x: Tensor,
  k: Tensor,
  { dim = -1 },
) {
  /* `
    Compute the Möbius scalar multiplication.

    .. math::

        r \otimes_\kappa x
        =
        \tan_\kappa(r\tan_\kappa^{-1}(\|x\|_2))\frac{x}{\|x\|_2}

    This operation has properties similar to the Euclidean scalar multiplication

    * `n-addition` property

    .. math::

         r \otimes_\kappa x = x \oplus_\kappa \dots \oplus_\kappa x

    * Distributive property

    .. math::

         (r_1 + r_2) \otimes_\kappa x
         =
         r_1 \otimes_\kappa x \oplus r_2 \otimes_\kappa x

    * Scalar associativity

    .. math::

         (r_1 r_2) \otimes_\kappa x = r_1 \otimes_\kappa (r_2 \otimes_\kappa x)

    * Monodistributivity

    .. math::

         r \otimes_\kappa (r_1 \otimes x \oplus r_2 \otimes x) =
         r \otimes_\kappa (r_1 \otimes x) \oplus r \otimes (r_2 \otimes x)

    * Scaling property

    .. math::

        |r| \otimes_\kappa x / \|r \otimes_\kappa x\|_2 = x/\|x\|_2

    Parameters
    ----------
    r : tensor
        scalar for multiplication
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the result of the Möbius scalar multiplication
    ` */
  return _mobiusScalarMul(r, x, k, { dim: dim })
}
function _mobiusScalarMul(
  r: Tensor,
  x: Tensor,
  k: Tensor,
  { dim = -1 }: _MobiusScalarMulInputType,
) {
  let xNorm
  let resC
  xNorm = x.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  resC = tanK(r * artanK(xNorm, k), k) * (x / xNorm)
  return resC
}
function dist(
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { keepdim = false, dim = -1 },
) {
  /* `
    Compute the geodesic distance between :math:`x` and :math:`y` on the manifold.

    .. math::

        d_\kappa(x, y) = 2\tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)

    .. plot:: plots/extended/stereographic/distance.py

    Parameters
    ----------
    x : tensor
        point on manifold
    y : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    ` */
  return _dist(x, y, k, { dim: dim, keepdim: keepdim })
}
function _dist(
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { keepdim = false, dim = -1 }: _DistInputType,
) {
  return (
    2.0 *
    artanK(
      _mobiusAdd(-x, y, k, { dim: dim }).norm({
        dim: dim,
        keepdim: keepdim,
        p: 2,
      }),
      k,
    )
  )
}
function dist0(x: Tensor, k: Tensor, { keepdim = false, dim = -1 }) {
  /* `
    Compute geodesic distance to the manifold's origin.

    Parameters
    ----------
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`0`
    ` */
  return _dist0(x, k, { dim: dim, keepdim: keepdim })
}
function _dist0(
  x: Tensor,
  k: Tensor,
  { keepdim = false, dim = -1 }: _Dist0InputType,
) {
  return 2.0 * artanK(x.norm({ dim: dim, keepdim: keepdim, p: 2 }), k)
}
function geodesic(
  t: Tensor,
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { dim = -1 },
) {
  /* `
    Compute the point on the path connecting :math:`x` and :math:`y` at time :math:`x`.

    The path can also be treated as an extension of the line segment to an
    unbounded geodesic that goes through :math:`x` and :math:`y`. The equation
    of the geodesic is given as:

    .. math::

        \gamma_{x\to y}(t)
        =
        x \oplus_\kappa t \otimes_\kappa ((-x) \oplus_\kappa y)

    The properties of the geodesic are the following:

    .. math::

        \gamma_{x\to y}(0) = x\\
        \gamma_{x\to y}(1) = y\\
        \dot\gamma_{x\to y}(t) = v

    Furthermore, the geodesic also satisfies the property of local distance
    minimization:

    .. math::

         d_\kappa(\gamma_{x\to y}(t_1), \gamma_{x\to y}(t_2)) = v|t_1-t_2|

    "Natural parametrization" of the curve ensures unit speed geodesics which
    yields the above formula with :math:`v=1`.

    However, we can always compute the constant speed :math:`v` from the points
    that the particular path connects:

    .. math::

        v = d_\kappa(\gamma_{x\to y}(0), \gamma_{x\to y}(1)) = d_\kappa(x, y)


    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        point on the geodesic going through x and y
    ` */
  return _geodesic(t, x, y, k, { dim: dim })
}
function _geodesic(
  t: Tensor,
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { dim = -1 }: _GeodesicInputType,
) {
  let v
  let tv
  let gammaT
  v = _mobiusAdd(-x, y, k, { dim: dim })
  tv = _mobiusScalarMul(t, v, k, { dim: dim })
  gammaT = _mobiusAdd(x, tv, k, { dim: dim })
  return gammaT
}
function expmap(x: Tensor, u: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the exponential map of :math:`u` at :math:`x`.

    The expmap is tightly related with :func:`geodesic`. Intuitively, the
    expmap represents a smooth travel along a geodesic from the starting point
    :math:`x`, into the initial direction :math:`u` at speed :math:`\|u\|_x` for
    the duration of one time unit. In formulas one can express this as the
    travel along the curve :math:`\gamma_{x, u}(t)` such that

    .. math::

        \gamma_{x, u}(0) = x\\
        \dot\gamma_{x, u}(0) = u\\
        \|\dot\gamma_{x, u}(t)\|_{\gamma_{x, u}(t)} = \|u\|_x

    The existence of this curve relies on uniqueness of the differential
    equation solution, that is local. For the universal manifold the solution
    is well defined globally and we have.

    .. math::

        \operatorname{exp}^\kappa_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_\kappa \tan_\kappa(\|u\|_x/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    x : tensor
        starting point on manifold
    u : tensor
        speed vector in tangent space at x
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    ` */
  return _expmap(x, u, k, { dim: dim })
}
function _expmap(
  x: Tensor,
  u: Tensor,
  k: Tensor,
  { dim = -1 }: _ExpmapInputType,
) {
  let uNorm
  let lam
  let secondTerm
  let y
  uNorm = u.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  lam = _lambdaX(x, k, { dim: dim, keepdim: true })
  secondTerm = tanK((lam / 2.0) * uNorm, k) * (u / uNorm)
  y = _mobiusAdd(x, secondTerm, k, { dim: dim })
  return y
}
function expmap0(u: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the exponential map of :math:`u` at the origin :math:`0`.

    .. math::

        \operatorname{exp}^\kappa_0(u)
        =
        \tan_\kappa(\|u\|_2/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    u : tensor
        speed vector on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    ` */
  return _expmap0(u, k, { dim: dim })
}
function _expmap0(
  u: Tensor,
  k: Tensor,
  { dim = -1 }: _Expmap0InputType,
) {
  let uNorm
  let gamma1
  uNorm = u.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  gamma1 = tanK(uNorm, k) * (u / uNorm)
  return gamma1
}
function geodesicUnit(
  t: Tensor,
  x: Tensor,
  u: Tensor,
  k: Tensor,
  { dim = -1 },
) {
  /* `
    Compute the point on the unit speed geodesic.

    The point on the unit speed geodesic at time :math:`t`, starting
    from :math:`x` with initial direction :math:`u/\|u\|_x` is computed
    as follows:

    .. math::

        \gamma_{x,u}(t) = x\oplus_\kappa \tan_\kappa(t/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    t : tensor
        travelling time
    x : tensor
        initial point on manifold
    u : tensor
        initial direction in tangent space at x
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        the point on the unit speed geodesic
    ` */
  return _geodesicUnit(t, x, u, k, { dim: dim })
}
function _geodesicUnit(
  t: Tensor,
  x: Tensor,
  u: Tensor,
  k: Tensor,
  { dim = -1 }: _GeodesicUnitInputType,
) {
  let uNorm
  let secondTerm
  let gamma1
  uNorm = u.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  secondTerm = tanK(t / 2.0, k) * (u / uNorm)
  gamma1 = _mobiusAdd(x, secondTerm, k, { dim: dim })
  return gamma1
}
function logmap(x: Tensor, y: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the logarithmic map of :math:`y` at :math:`x`.

    .. math::

        \operatorname{log}^\kappa_x(y) = \frac{2}{\lambda_x^\kappa}
        \tan_\kappa^{-1}(\|(-x)\oplus_\kappa y\|_2)
        * \frac{(-x)\oplus_\kappa y}{\|(-x)\oplus_\kappa y\|_2}

    The result of the logmap is a vector :math:`u` in the tangent space of
    :math:`x` such that

    .. math::

        y = \operatorname{exp}^\kappa_x(\operatorname{log}^\kappa_x(y))


    Parameters
    ----------
    x : tensor
        starting point on manifold
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_x M` that transports :math:`x` to :math:`y`
    ` */
  return _logmap(x, y, k, { dim: dim })
}
function _logmap(
  x: Tensor,
  y: Tensor,
  k: Tensor,
  { dim = -1 }: _LogmapInputType,
) {
  let sub
  let subNorm
  let lam
  sub = _mobiusAdd(-x, y, k, { dim: dim })
  subNorm = sub.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  lam = _lambdaX(x, k, { dim: dim, keepdim: true })
  return 2.0 * artanK(subNorm, k) * (sub / (lam * subNorm))
}
function logmap0(y: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the logarithmic map of :math:`y` at the origin :math:`0`.

    .. math::

        \operatorname{log}^\kappa_0(y)
        =
        \tan_\kappa^{-1}(\|y\|_2) \frac{y}{\|y\|_2}

    The result of the logmap at the origin is a vector :math:`u` in the tangent
    space of the origin :math:`0` such that

    .. math::

        y = \operatorname{exp}^\kappa_0(\operatorname{log}^\kappa_0(y))

    Parameters
    ----------
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_0 M` that transports :math:`0` to :math:`y`
    ` */
  return _logmap0(y, k, { dim: dim })
}
function _logmap0(y: Tensor, k, { dim = -1 }: _Logmap0InputType) {
  let yNorm
  yNorm = y.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  return (y / yNorm) * artanK(yNorm, k)
}
function mobiusMatvec(m: Tensor, x: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Compute the generalization of matrix-vector multiplication in gyrovector spaces.

    The Möbius matrix vector operation is defined as follows:

    .. math::

        M \otimes_\kappa x = \tan_\kappa\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tan_\kappa^{-1}(\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}

    .. plot:: plots/extended/stereographic/mobius_matvec.py

    Parameters
    ----------
    m : tensor
        matrix for multiplication. Batched matmul is performed if
        ``m.dim() > 2``, but only last dim reduction is supported
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius matvec result
    ` */
  return _mobiusMatvec(m, x, k, { dim: dim })
}
function _mobiusMatvec(
  m: Tensor,
  x: Tensor,
  k: Tensor,
  { dim = -1 }: _MobiusMatvecInputType,
) {
  let xNorm
  let mx
  let mxNorm
  let resC
  let cond
  let res0
  let res
  if (m.dim() > 2 && dim != -1) {
    throw new runtimeError(
      'broadcasted Möbius matvec is supported for the last dim only',
    )
  }
  xNorm = x.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  if (dim != -1 || m.dim() == 2) {
    mx = torch.tensordot(x, m, [[dim], [1]])
  } else {
    mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
  }
  mxNorm = mx.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  resC = tanK((mxNorm / xNorm) * artanK(xNorm, k), k) * (mx / mxNorm)
  cond = (mx == 0).prod({ dim: dim, dtype: torch.bool, keepdim: true })
  res0 = torch.zeros(1, { device: resC.device, dtype: resC.dtype })
  res = torch.where(cond, res0, resC)
  return res
}
function mobiusPointwiseMul(
  w: Tensor,
  x: Tensor,
  k: Tensor,
  { dim = -1 },
) {
  /* `
    Compute the generalization for point-wise multiplication in gyrovector spaces.

    The Möbius pointwise multiplication is defined as follows

    .. math::

        \operatorname{diag}(w) \otimes_\kappa x = \tan_\kappa\left(
            \frac{\|\operatorname{diag}(w)x\|_2}{x}\tanh^{-1}(\|x\|_2)
        \right)\frac{\|\operatorname{diag}(w)x\|_2}{\|x\|_2}


    Parameters
    ----------
    w : tensor
        weights for multiplication (should be broadcastable to x)
    x : tensor
        point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Möbius point-wise mul result
    ` */
  return _mobiusPointwiseMul(w, x, k, { dim: dim })
}
function _mobiusPointwiseMul(
  w: Tensor,
  x: Tensor,
  k: Tensor,
  { dim = -1 }: _MobiusPointwiseMulInputType,
) {
  let xNorm
  let wx
  let wxNorm
  let resC
  let zero
  let cond
  let res
  xNorm = x.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  wx = w * x
  wxNorm = wx.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  resC = tanK((wxNorm / xNorm) * artanK(xNorm, k), k) * (wx / wxNorm)
  zero = torch.zeros([], { device: resC.device, dtype: resC.dtype })
  cond = wx
    .isclose(zero)
    .prod({ dim: dim, dtype: torch.bool, keepdim: true })
  res = torch.where(cond, zero, resC)
  return res
}
function mobiusFnApplyChain(
  x: Tensor,
  fns: Callable,
  k: Tensor,
  { dim = -1 },
) {
  let ex
  let fn
  let y
  /* `
    Compute the generalization of sequential function application in gyrovector spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{log}^\kappa_0` and then the sequence of functions is
    applied to the vector in the tangent space. The resulting tangent vector is
    then mapped back with :math:`\operatorname{exp}^\kappa_0`.

    .. math::

        f^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(f(\operatorname{log}^\kappa_0(y)))

    The definition of mobius function application allows chaining as

    .. math::

        y = \operatorname{exp}^\kappa_0(\operatorname{log}^\kappa_0(y))

    Resulting in

    .. math::

        (f \circ g)^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(
            (f \circ g) (\operatorname{log}^\kappa_0(y))
        )

    Parameters
    ----------
    x : tensor
        point on manifold
    fns : callable[]
        functions to apply
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Apply chain result
    ` */
  if (!fns) {
    return x
  } else {
    ex = _logmap0(x, k, { dim: dim })
    for (fn of fns) {
      ex = fn(ex)
    }
    y = _expmap0(ex, k, { dim: dim })
    return y
  }
}
function mobiusFnApply(
  fn: Callable,
  x: Tensor,
  args,
  k: Tensor,
  kwargs,
  { dim = -1 },
) {
  let ex
  let y
  /* `
    Compute the generalization of function application in gyrovector spaces.

    First, a gyrovector is mapped to the tangent space (first-order approx.) via
    :math:`\operatorname{log}^\kappa_0` and then the function is applied
    to the vector in the tangent space. The resulting tangent vector is then
    mapped back with :math:`\operatorname{exp}^\kappa_0`.

    .. math::

        f^{\otimes_\kappa}(x)
        =
        \operatorname{exp}^\kappa_0(f(\operatorname{log}^\kappa_0(y)))

    .. plot:: plots/extended/stereographic/mobius_sigmoid_apply.py

    Parameters
    ----------
    x : tensor
        point on manifold
    fn : callable
        function to apply
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Result of function in hyperbolic space
    ` */
  ex = _logmap0(x, k, { dim: dim })
  ex = fn(ex, args, kwargs)
  y = _expmap0(ex, k, { dim: dim })
  return y
}
function mobiusify(fn: Callable) {
  /* `
    Wrap a function such that is works in gyrovector spaces.

    Parameters
    ----------
    fn : callable
        function in Euclidean space

    Returns
    -------
    callable
        function working in gyrovector spaces

    Notes
    -----
    New function will accept additional argument ``k`` and ``dim``.
    ` */
  function mobiusFn(x, args, k, kwargs, { dim = -1 }) {
    let ex
    let y
    ex = _logmap0(x, k, { dim: dim })
    ex = fn(ex, args, kwargs)
    y = _expmap0(ex, k, { dim: dim })
    return y
  }
  return mobiusFn
}
function dist2Plane(
  x: Tensor,
  p: Tensor,
  a: Tensor,
  k: Tensor,
  { keepdim = false, signed = false, scaled = false, dim = -1 },
) {
  /* `
    Geodesic distance from :math:`x` to a hyperplane :math:`H_{a, b}`.

    The hyperplane is such that its set of points is orthogonal to :math:`a` and
    contains :math:`p`.

    .. plot:: plots/extended/stereographic/distance2plane.py

    To form an intuition what is a hyperplane in gyrovector spaces, let's first
    consider an Euclidean hyperplane

    .. math::

        H_{a, b} = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - b = 0
        \right\},

    where :math:`a\in \mathbb{R}^n\backslash \{\mathbf{0}\}` and
    :math:`b\in \mathbb{R}^n`.

    This formulation of a hyperplane is hard to generalize,
    therefore we can rewrite :math:`\langle x, a\rangle - b`
    utilizing orthogonal completion.
    Setting any :math:`p` s.t. :math:`b=\langle a, p\rangle` we have

    .. math::

        H_{a, b} = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - b = 0
        \right\}\\
        =H_{a, \langle a, p\rangle} = \tilde{H}_{a, p}\\
        = \left\{
            x \in \mathbb{R}^n\;:\;\langle x, a\rangle - \langle a, p\rangle = 0
        \right\}\\
        =\left\{
            x \in \mathbb{R}^n\;:\;\langle -p + x, a\rangle = 0
        \right\}\\
        = p + \{a\}^\perp

    Naturally we have a set :math:`\{a\}^\perp` with applied :math:`+` operator
    to each element. Generalizing a notion of summation to the gyrovector space
    we replace :math:`+` with :math:`\oplus_\kappa`.

    Next, we should figure out what is :math:`\{a\}^\perp` in the gyrovector
    space.

    First thing that we should acknowledge is that notion of orthogonality is
    defined for vectors in tangent spaces. Let's consider now
    :math:`p\in \mathcal{M}_\kappa^n` and
    :math:`a\in T_p\mathcal{M}_\kappa^n\backslash \{\mathbf{0}\}`.

    Slightly deviating from traditional notation let's write
    :math:`\{a\}_p^\perp` highlighting the tight relationship of
    :math:`a\in T_p\mathcal{M}_\kappa^n\backslash \{\mathbf{0}\}`
    with :math:`p \in \mathcal{M}_\kappa^n`. We then define

    .. math::

        \{a\}_p^\perp := \left\{
            z\in T_p\mathcal{M}_\kappa^n \;:\; \langle z, a\rangle_p = 0
        \right\}

    Recalling that a tangent vector :math:`z` for point :math:`p` yields
    :math:`x = \operatorname{exp}^\kappa_p(z)` we rewrite the above equation as

    .. math::
        \{a\}_p^\perp := \left\{
            x\in \mathcal{M}_\kappa^n \;:\; \langle
            \operatorname{log}_p^\kappa(x), a\rangle_p = 0
        \right\}

    This formulation is something more pleasant to work with.
    Putting all together

    .. math::

        \tilde{H}_{a, p}^\kappa = p + \{a\}^\perp_p\\
        = \left\{
            x \in \mathcal{M}_\kappa^n\;:\;\langle
            \operatorname{log}^\kappa_p(x),
            a\rangle_p = 0
        \right\} \\
        = \left\{
            x \in \mathcal{M}_\kappa^n\;:\;\langle -p \oplus_\kappa x, a\rangle
            = 0
        \right\}

    To compute the distance :math:`d_\kappa(x, \tilde{H}_{a, p}^\kappa)` we find

    .. math::

        d_\kappa(x, \tilde{H}_{a, p}^\kappa)
        =
        \inf_{w\in \tilde{H}_{a, p}^\kappa} d_\kappa(x, w)\\
        =
        \sin^{-1}_\kappa\left\{
            \frac{
            2 |\langle(-p)\oplus_\kappa x, a\rangle|
            }{
            (1+\kappa\|(-p)\oplus_\kappa \|x\|^2_2)\|a\|_2
            }
        \right\}

    Parameters
    ----------
    x : tensor
        point on manifold to compute distance for
    a : tensor
        hyperplane normal vector in tangent space of :math:`p`
    p : tensor
        point on manifold lying on the hyperplane
    k : tensor
        sectional curvature of manifold
    keepdim : bool
        retain the last dim? (default: false)
    signed : bool
        return signed distance
    scaled : bool
        scale distance by tangent norm
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        distance to the hyperplane
    ` */
  return _dist2Plane(x, a, p, k, {
    dim: dim,
    keepdim: keepdim,
    scaled: scaled,
    signed: signed,
  })
}
function _dist2Plane(
  x: Tensor,
  a: Tensor,
  p: Tensor,
  k: Tensor,
  {
    keepdim = false,
    signed = false,
    scaled = false,
    dim = -1,
  }: _Dist2PlaneInputType,
) {
  let diff
  let diffNorm2
  let scDiffA
  let aNorm
  let num
  let denom
  let distance
  diff = _mobiusAdd(-p, x, k, { dim: dim })
  diffNorm2 = diff
    .pow(2)
    .sum({ dim: dim, keepdim: keepdim })
    .clampMin(1e-15)
  scDiffA = (diff * a).sum({ dim: dim, keepdim: keepdim })
  if (!signed) {
    scDiffA = scDiffA.abs()
  }
  aNorm = a.norm({ dim: dim, keepdim: keepdim, p: 2 })
  num = 2.0 * scDiffA
  denom = clampAbs((1 + k * diffNorm2) * aNorm)
  distance = arsinK(num / denom, k)
  if (scaled) {
    distance = distance * aNorm
  }
  return distance
}
function parallelTransport(
  x: Tensor,
  y: Tensor,
  v: Tensor,
  k: Tensor,
  { dim = -1 },
) {
  /* `
    Compute the parallel transport of :math:`v` from :math:`x` to :math:`y`.

    The parallel transport is essential for adaptive algorithms on Riemannian
    manifolds. For gyrovector spaces the parallel transport is expressed through
    the gyration.

    .. plot:: plots/extended/stereographic/gyrovector_parallel_transport.py

    To recover parallel transport we first need to study isomorphisms between
    gyrovectors and vectors. The reason is that originally, parallel transport
    is well defined for gyrovectors as

    .. math::

        P_{x\to y}(z) = \operatorname{gyr}[y, -x]z,

    where :math:`x,\:y,\:z \in \mathcal{M}_\kappa^n` and
    :math:`\operatorname{gyr}[a, b]c = \ominus (a \oplus_\kappa b)
    \oplus_\kappa (a \oplus_\kappa (b \oplus_\kappa c))`

    But we want to obtain parallel transport for vectors, not for gyrovectors.
    The blessing is the isomorphism mentioned above. This mapping is given by

    .. math::

        U^\kappa_p \: : \: T_p\mathcal{M}_\kappa^n \to \mathbb{G}
        =
        v \mapsto \lambda^\kappa_p v


    Finally, having the points :math:`x,\:y \in \mathcal{M}_\kappa^n` and a
    tangent vector :math:`u\in T_x\mathcal{M}_\kappa^n` we obtain

    .. math::

        P^\kappa_{x\to y}(v)
        =
        (U^\kappa_y)^{-1}\left(\operatorname{gyr}[y, -x] U^\kappa_x(v)\right)\\
        =
        \operatorname{gyr}[y, -x] v \lambda^\kappa_x / \lambda^\kappa_y

    .. plot:: plots/extended/stereographic/parallel_transport.py


    Parameters
    ----------
    x : tensor
        starting point
    y : tensor
        end point
    v : tensor
        tangent vector at x to be transported to y
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        transported vector
    ` */
  return _parallelTransport(x, y, v, k, { dim: dim })
}
function _parallelTransport(
  x: Tensor,
  y: Tensor,
  u: Tensor,
  k: Tensor,
  { dim = -1 }: _ParallelTransportInputType,
) {
  return (
    (_gyration(y, -x, u, k, { dim: dim }) *
      _lambdaX(x, k, { dim: dim, keepdim: true })) /
    _lambdaX(y, k, { dim: dim, keepdim: true })
  )
}
function parallelTransport0(
  y: Tensor,
  v: Tensor,
  k: Tensor,
  { dim = -1 },
) {
  /* `
    Compute the parallel transport of :math:`v` from the origin :math:`0` to :math:`y`.

    This is just a special case of the parallel transport with the starting
    point at the origin that can be computed more efficiently and more
    numerically stable.

    Parameters
    ----------
    y : tensor
        target point
    v : tensor
        vector to be transported from the origin to y
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    ` */
  return _parallelTransport0(y, v, k, { dim: dim })
}
function _parallelTransport0(
  y: Tensor,
  v: Tensor,
  k: Tensor,
  { dim = -1 }: _ParallelTransport0InputType,
) {
  return (
    v *
    (1 + k * y.pow(2).sum({ dim: dim, keepdim: true })).clampMin(1e-15)
  )
}
function parallelTransport0Back(
  x: Tensor,
  v: Tensor,
  k: Tensor,
  { dim = -1 }: ParallelTransport0BackInputType,
) {
  /* `
    Perform parallel transport to the zero point.

    Special case parallel transport with last point at zero that
    can be computed more efficiently and numerically stable

    Parameters
    ----------
    x : tensor
        target point
    v : tensor
        vector to be transported
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
    ` */
  return _parallelTransport0Back(x, v, { dim: dim, k: k })
}
function _parallelTransport0Back(
  x: Tensor,
  v: Tensor,
  k: Tensor,
  { dim = -1 }: _ParallelTransport0BackInputType,
) {
  return (
    v /
    (1 + k * x.pow(2).sum({ dim: dim, keepdim: true })).clampMin(1e-15)
  )
}
function egrad2Rgrad(x: Tensor, grad: Tensor, k: Tensor, { dim = -1 }) {
  /* `
    Convert the Euclidean gradient to the Riemannian gradient.

    .. math::

        \nabla_x = \nabla^E_x / (\lambda_x^\kappa)^2

    Parameters
    ----------
    x : tensor
        point on the manifold
    grad : tensor
        Euclidean gradient for :math:`x`
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        Riemannian gradient :math:`u\in T_x\mathcal{M}_\kappa^n`
    ` */
  return _egrad2Rgrad(x, grad, k, { dim: dim })
}
function _egrad2Rgrad(
  x: Tensor,
  grad: Tensor,
  k: Tensor,
  { dim = -1 }: _Egrad2RgradInputType,
) {
  return grad / _lambdaX(x, k, { dim: dim, keepdim: true }) ** 2
}
function sproj(x: Tensor, k: Tensor, { dim = -1 }: SprojInputType) {
  /* `
    Stereographic Projection from hyperboloid or sphere.

    Parameters
    ----------
    x : tensor
        point to be projected
    k : tensor
        constant sectional curvature
    dim : int
        dimension to operate on

    Returns
    -------
    tensor
        the result of the projection
    ` */
  return _sproj(x, k, { dim: dim })
}
function _sproj(x: Tensor, k: Tensor, { dim = -1 }: _SprojInputType) {
  let invR
  let factor
  let proj
  invR = torch.sqrt(sabs(k))
  factor = 1.0 / (1.0 + invR * x.narrow(dim, -1, 1))
  proj = factor * x.narrow(dim, 0, x.size(dim) - 1)
  return proj
}
function invSproj(
  x: Tensor,
  k: Tensor,
  { dim = -1 }: InvSprojInputType,
) {
  /* `
    Inverse of Stereographic Projection to hyperboloid or sphere.

    Parameters
    ----------
    x : tensor
        point to be projected
    k : tensor
        constant sectional curvature
    dim : int
        dimension to operate on

    Returns
    -------
    tensor
        the result of the projection
    ` */
  return _invSproj(x, k, { dim: dim })
}
function _invSproj(
  x: Tensor,
  k: Tensor,
  { dim = -1 }: _InvSprojInputType,
) {
  let invR
  let lamX
  let a
  let b
  let proj
  invR = torch.sqrt(sabs(k))
  lamX = _lambdaX(x, k, { dim: dim, keepdim: true })
  a = lamX * x
  b = (1.0 / invR) * (lamX - 1.0)
  proj = torch.cat([a, b], { dim: dim })
  return proj
}
function antipode(
  x: Tensor,
  k: Tensor,
  { dim = -1 }: AntipodeInputType,
) {
  /* `
    Compute the antipode of a point :math:`x_1,...,x_n` for :math:`\kappa > 0`.

    Let :math:`x` be a point on some sphere. Then :math:`-x` is its antipode.
    Since we're dealing with stereographic projections, for :math:`sproj(x)` we
    get the antipode :math:`sproj(-x)`. Which is given as follows:

    .. math::

        \text{antipode}(x)
        =
        \frac{1+\kappa\|x\|^2_2}{2\kappa\|x\|^2_2}{}(-x)

    Parameters
    ----------
    x : tensor
        points :math:`x_1,...,x_n` on manifold to compute antipode for
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        antipode
    ` */
  return _antipode(x, k, { dim: dim })
}
function _antipode(
  x: Tensor,
  k: Tensor,
  { dim = -1 }: _AntipodeInputType,
) {
  let v
  let r
  let pi
  let a
  if (torch.all(k.le(0))) {
    return -x
  }
  v = x / x.norm({ dim: dim, keepdim: true, p: 2 }).clampMin(1e-15)
  r = sabs(k).sqrt().reciprocal()
  pi = 3.141592653589793
  a = _geodesicUnit(pi * r, x, v, k, { dim: dim })
  return torch.where(k.gt(0), a, -x)
}
function weightedMidpoint(
  xs: Tensor,
  k: Tensor,
  {
    weights = null,
    reducedim = null,
    dim = -1,
    keepdim = false,
    lincomb = false,
    posweight = false,
  }: WeightedMidpointInputType,
) {
  /* `
    Compute weighted Möbius gyromidpoint.

    The weighted Möbius gyromidpoint of a set of points
    :math:`x_1,...,x_n` according to weights
    :math:`\alpha_1,...,\alpha_n` is computed as follows:

    The weighted Möbius gyromidpoint is computed as follows

    .. math::

        m_{\kappa}(x_1,\ldots,x_n,\alpha_1,\ldots,\alpha_n)
        =
        \frac{1}{2}
        \otimes_\kappa
        \left(
        \sum_{i=1}^n
        \frac{
        \alpha_i\lambda_{x_i}^\kappa
        }{
        \sum_{j=1}^n\alpha_j(\lambda_{x_j}^\kappa-1)
        }
        x_i
        \right)

    where the weights :math:`\alpha_1,...,\alpha_n` do not necessarily need
    to sum to 1 (only their relative weight matters). Note that this formula
    also requires to choose between the midpoint and its antipode for
    :math:`\kappa > 0`.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
    reducedim : int|list|tuple
        reduce dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    k : tensor
        constant sectional curvature
    keepdim : bool
        retain the last dim? (default: false)
    lincomb : bool
        linear combination implementation
    posweight : bool
        make all weights positive. Negative weight will weight antipode of entry with positive weight instead.
        This will give experimentally better numerics and nice interpolation
        properties for linear combination and averaging

    Returns
    -------
    tensor
        Einstein midpoint in poincare coordinates
    ` */
  return _weightedMidpoint({
    dim: dim,
    k: k,
    keepdim: keepdim,
    lincomb: lincomb,
    posweight: posweight,
    reducedim: reducedim,
    weights: weights,
    xs: xs,
  })
}
function _weightedMidpoint(
  xs: Tensor,
  k: Tensor,
  {
    weights = null,
    reducedim = null,
    dim = -1,
    keepdim = false,
    lincomb = false,
    posweight = false,
  }: _WeightedMidpointInputType,
) {
  let gamma
  let denominator
  let nominator
  let twoMean
  let aMean
  let bMean
  let aDist
  let bDist
  let better
  let alpha
  let d
  let _
  if (reducedim == null) {
    reducedim = listRange(xs.dim())
    reducedim.pop(dim)
  }
  gamma = _lambdaX(xs, { dim: dim, k: k, keepdim: true })
  if (weights == null) {
    weights = torch.tensor(1.0, { device: xs.device, dtype: xs.dtype })
  } else {
    weights = weights.unsqueeze(dim)
  }
  if (posweight && weights.lt(0).any()) {
    xs = torch.where(
      weights.lt(0),
      _antipode(xs, { dim: dim, k: k }),
      xs,
    )
    weights = weights.abs()
  }
  denominator = ((gamma - 1) * weights).sum(reducedim, {
    keepdim: true,
  })
  nominator = (gamma * weights * xs).sum(reducedim, { keepdim: true })
  twoMean = nominator / clampAbs(denominator, 1e-10)
  aMean = _mobiusScalarMul(
    torch.tensor(0.5, { device: xs.device, dtype: xs.dtype }),
    twoMean,
    { dim: dim, k: k },
  )
  if (torch.any(k.gt(0))) {
    bMean = _antipode(aMean, k, { dim: dim })
    aDist = _dist(aMean, xs, { dim: dim, k: k, keepdim: true }).sum(
      reducedim,
      { keepdim: true },
    )
    bDist = _dist(bMean, xs, { dim: dim, k: k, keepdim: true }).sum(
      reducedim,
      { keepdim: true },
    )
    better = k.gt(0) & (bDist < aDist)
    aMean = torch.where(better, bMean, aMean)
  }
  if (lincomb) {
    if (weights.numel() == 1) {
      alpha = weights.clone()
      for (d of reducedim) {
        alpha *= xs.size(d)
      }
    } else {
      ;[weights, _] = torch.broadcastTensors(weights, gamma)
      alpha = weights.sum(reducedim, { keepdim: true })
    }
    aMean = _mobiusScalarMul(alpha, aMean, { dim: dim, k: k })
  }
  if (!keepdim) {
    aMean = dropDims(aMean, reducedim)
  }
  return aMean
}
