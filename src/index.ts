let geometry: EGeometry
let variation: EVariation
function squar(x: Ld): Ld {
  return x * x
}
function sig(z: number): number {
  return ginf[geometry].g.sig[z]
}
function curvature(): number {
  switch (cgclass) {
    case gcEuclid: {
      return 0
    }
    case gcHyperbolic: {
      return -1
    }
    case gcSphere: {
      return 1
    }
    case gcProduct: {
      return PIU(curvature())
    }
    default: {
      return 0
    }
  }
}
function sinAuto(x: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return x
    }
    case gcHyperbolic: {
      return sinh(x)
    }
    case gcSphere: {
      return sin(x)
    }
    case gcProduct: {
      return PIU(sin_auto(x))
    }
    case gcSL2: {
      return sinh(x)
    }
    default: {
      return x
    }
  }
}
function asinAuto(x: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return x
    }
    case gcHyperbolic: {
      return asinh(x)
    }
    case gcSphere: {
      return asin(x)
    }
    case gcProduct: {
      return PIU(asin_auto(x))
    }
    case gcSL2: {
      return asinh(x)
    }
    default: {
      return x
    }
  }
}
function acosAuto(x: Ld): Ld {
  switch (cgclass) {
    case gcHyperbolic: {
      return acosh(x)
    }
    case gcSphere: {
      return acos(x)
    }
    case gcProduct: {
      return PIU(acos_auto(x))
    }
    case gcSL2: {
      return acosh(x)
    }
    default: {
      return x
    }
  }
}
function volumeAuto(r: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return r * r * r * _deg(240)
    }
    case gcHyperbolic: {
      return M_PI * (sinh(2 * r) - 2 * r)
    }
    case gcSphere: {
      return M_PI * (2 * r - sin(2 * r))
    }
    default: {
      return 0
    }
  }
}
function areaAuto(r: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return r * r * M_PI
    }
    case gcHyperbolic: {
      return TAU * (cosh(r) - 1)
    }
    case gcSphere: {
      return TAU * (1 - cos(r))
    }
    default: {
      return 0
    }
  }
}
function wvolareaAuto(r: Ld): Ld {
  if (WDIM == 3) {
    return volume_auto(r)
  } else {
    return area_auto(r)
  }
}
function asinClamp(x: Ld): Ld {
  return x > 1
    ? _deg(90)
    : x < -1
    ? -_deg(90)
    : std_isnan(x)
    ? 0
    : asin(x)
}
function acosClamp(x: Ld): Ld {
  return x > 1 ? 0 : x < -1 ? M_PI : std_isnan(x) ? 0 : acos(x)
}
function asinAutoClamp(x: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return x
    }
    case gcHyperbolic: {
      return asinh(x)
    }
    case gcSL2: {
      return asinh(x)
    }
    case gcSphere: {
      return asin_clamp(x)
    }
    case gcProduct: {
      return PIU(asin_auto_clamp(x))
    }
    default: {
      return x
    }
  }
}
function acosAutoClamp(x: Ld): Ld {
  switch (cgclass) {
    case gcHyperbolic: {
      return x < 1 ? 0 : acosh(x)
    }
    case gcSL2: {
      return x < 1 ? 0 : acosh(x)
    }
    case gcSphere: {
      return acos_clamp(x)
    }
    case gcProduct: {
      return PIU(acos_auto_clamp(x))
    }
    default: {
      return x
    }
  }
}
function cosAuto(x: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return 1
    }
    case gcHyperbolic: {
      return cosh(x)
    }
    case gcSL2: {
      return cosh(x)
    }
    case gcSphere: {
      return cos(x)
    }
    case gcProduct: {
      return PIU(cos_auto(x))
    }
    default: {
      return 1
    }
  }
}
function tanAuto(x: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return x
    }
    case gcHyperbolic: {
      return tanh(x)
    }
    case gcSphere: {
      return tan(x)
    }
    case gcProduct: {
      return PIU(tan_auto(x))
    }
    case gcSL2: {
      return tanh(x)
    }
    default: {
      return 1
    }
  }
}
function atanAuto(x: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return x
    }
    case gcHyperbolic: {
      return atanh(x)
    }
    case gcSphere: {
      return atan(x)
    }
    case gcProduct: {
      return PIU(atan_auto(x))
    }
    case gcSL2: {
      return atanh(x)
    }
    default: {
      return x
    }
  }
}
function atan2Auto(y: Ld, x: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return y / x
    }
    case gcHyperbolic: {
      return atanh(y / x)
    }
    case gcSL2: {
      return atanh(y / x)
    }
    case gcSphere: {
      return atan2(y, x)
    }
    case gcProduct: {
      return PIU(atan2_auto(y, x))
    }
    default: {
      return y / x
    }
  }
}
function edgeOfTriangleWithAngles(alpha: Ld, beta: Ld, gamma: Ld): Ld {
  return acos_auto(
    (cos(alpha) + cos(beta) * cos(gamma)) / (sin(beta) * sin(gamma)),
  )
}
function hpxy(x: Ld, y: Ld): Hyperpoint {
  if (embedded_plane) {
    geom3_light_flip(true)
    let h: Hyperpoint = hpxy(x, y)
    geom3_light_flip(false)
    swapmatrix(h)
    return h
  }
  if (sl2) {
    return hyperpoint(x, y, 0, sqrt(1 + x * x + y * y))
  }
  if (rotspace) {
    return hyperpoint(x, y, 0, sqrt(1 - x * x - y * y))
  }
  return PIU(
    hpxyz(
      x,
      y,
      translatable
        ? 1
        : sphere
        ? sqrt(1 - x * x - y * y)
        : sqrt(1 + x * x + y * y),
    ),
  )
}
function hpxy3(x: Ld, y: Ld, z: Ld): Hyperpoint {
  return hpxyz3(
    x,
    y,
    z,
    sl2
      ? sqrt(1 + x * x + y * y - z * z)
      : translatable
      ? 1
      : sphere
      ? sqrt(1 - x * x - y * y - z * z)
      : sqrt(1 + x * x + y * y + z * z),
  )
}
function zeroD(d: number, h: Hyperpoint): Bool {
  for (let i: number = 0; i < d; i++) {
    if (h[i]) {
      return
    }
  }
  return
}
function intval(h1: Hyperpoint, h2: Hyperpoint): Ld {
  let res: Ld = 0
  for (let i: number = 0; i < MDIM; i++) {
    res += squar(h1[i] - h2[i]) * sig(i)
  }
  if (elliptic) {
    let res2: Ld = 0
    for (let i: number = 0; i < MDIM; i++) {
      res2 += squar(h1[i] + h2[i]) * sig(i)
    }
    return min(res, res2)
  }
  return res
}
function quickdist(h1: Hyperpoint, h2: Hyperpoint): Ld {
  if (gproduct) {
    return hdist(h1, h2)
  }
  return intval(h1, h2)
}
function sqhypotD(d: number, h: Hyperpoint): Ld {
  let sum: Ld = 0
  for (let i: number = 0; i < d; i++) {
    sum += h[i] * h[i]
  }
  return sum
}
function hypotD(d: number, h: Hyperpoint): Ld {
  return sqrt(sqhypot_d(d, h))
}
function toOtherSide(h1: Hyperpoint, h2: Hyperpoint): Transmatrix {
  if (geom3_sph_in_low() && !geom3_flipped) {
    geom3_light_flip(true)
    h1 = normalize(h1)
    h2 = normalize(h2)
    let t: Transmatrix = to_other_side(h1, h2)
    for (let i: number = 0; i < 4; i++) {
      T[i][3] = T[3][i] = i == 3
    }
    geom3_light_flip(false)
    return T
  }
  let d: Ld = hdist(h1, h2)
  let v: Hyperpoint
  if (euclid) {
    v = (h2 - h1) / d
  } else {
    v = (h1 * cos_auto(d) - h2) / sin_auto(d)
  }
  let d1: Ld
  if (euclid) {
    d1 = -(v | h1) / (v | v)
  } else {
    d1 = atan_auto(-v[LDIM] / h1[LDIM])
  }
  let hm: Hyperpoint =
    h1 * cos_auto(d1) + (sphere ? -1 : 1) * v * sin_auto(d1)
  return rspintox(hm) * xpush(-hdist0(hm) * 2) * spintox(hm)
}
function material(h: Hyperpoint): Ld {
  if (sphere || in_s2xe()) {
    return intval(h, Hypc)
  } else {
    if (hyperbolic || in_h2xe()) {
      return -intval(h, Hypc)
    } else {
      if (sl2) {
        return h[2] * h[2] + h[3] * h[3] - h[0] * h[0] - h[1] * h[1]
      } else {
        return h[LDIM]
      }
    }
  }
}
function safeClassifyIdeals(h: Hyperpoint): number {
  if (hyperbolic || in_h2xe()) {
    h /= h[LDIM]
    let x: Ld =
      MDIM == 3
        ? 1 - (h[0] * h[0] + h[1] * h[1])
        : 1 - (h[0] * h[0] + h[1] * h[1] + h[2] * h[2])
    if (x > 1e-6) {
      return 1
    }
    if (x < -1e-6) {
      return -1
    }
    return 0
  }
  return 1
}
let idealLimit = 10
let idealEach = degree
function safeApproximationOfIdeal(h: Hyperpoint): Hyperpoint {
  return towards_inf(C0, h, ideal_limit)
}
function closestToZero(a: Hyperpoint, b: Hyperpoint): Hyperpoint {
  if (sqhypot_d(MDIM, a - b) < 1e-9) {
    return a
  }
  if (isnan(a[0])) {
    return a
  }
  a /= a[LDIM]
  b /= b[LDIM]
  let mulA: Ld = 0
  let mulB = 0
  for (let i: number = 0; i < LDIM; i++) {
    let z: Ld = a[i] - b[i]
    mul_a += a[i] * z
    mul_b -= b[i] * z
  }
  return (mul_b * a + mul_a * b) / (mul_a + mul_b)
}
function zlevel(h: Hyperpoint): Ld {
  if (sl2) {
    return sqrt(-intval(h, Hypc))
  } else {
    if (translatable) {
      return h[LDIM]
    } else {
      if (sphere) {
        return sqrt(intval(h, Hypc))
      } else {
        if (in_e2xe()) {
          return log(h[2])
        } else {
          if (gproduct) {
            return log(sqrt(abs(intval(h, Hypc))))
          } else {
            return (h[LDIM] < 0 ? -1 : 1) * sqrt(-intval(h, Hypc))
          }
        }
      }
    }
  }
}
function hypotAuto(x: Ld, y: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return hypot(x, y)
    }
    case gcHyperbolic: {
      return acosh(cosh(x) * cosh(y))
    }
    case gcSphere: {
      return acos(cos(x) * cos(y))
    }
    default: {
      return hypot(x, y)
    }
  }
}
function normalize(h: Hyperpoint): Hyperpoint {
  if (gproduct) {
    return H
  }
  let z: Ld = zlevel(H)
  for (let c: number = 0; c < MXDIM; c++) {
    H[c] /= Z
  }
  return H
}
function ultraNormalize(h: Hyperpoint): Hyperpoint {
  if (material(H) <= 0) {
    H[LDIM] = hypot_d(LDIM, H) + 1e-10
  }
  return normalize(H)
}
function esl2Zpush(z: Ld): Transmatrix {
  return cspin(2, 3, z) * cspin(0, 1, z)
}
function esl2Ita0(h1: Hyperpoint): Hyperpoint {
  return esl2_zpush(h1[2]) * xpush(h1[0]) * ypush0(h1[1])
}
function esl2Ita(h1: Hyperpoint): Transmatrix {
  return esl2_zpush(h1[2]) * xpush(h1[0]) * ypush(h1[1])
}
function esl2Ati(h: Hyperpoint): Hyperpoint {
  let a1: Ld =
    (h[0] * h[3] - h[1] * h[2]) /
    (-h[2] * h[2] - h[1] * h[1] - h[0] * h[0] - h[3] * h[3])
  let a: Ld = a1 * a1
  let b: Ld = 4 * a - 1
  let u: Ld = sqrt(0.25 - a / b) - 0.5
  let s: Ld = sqrt(U) * (a1 > 0 ? 1 : -1)
  let x: Ld = -asinh(S)
  h = lorentz(0, 3, -x) * lorentz(1, 2, x) * h
  let y: Ld =
    h[3] * h[3] > h[2] * h[2] ? atanh(h[1] / h[3]) : atanh(h[0] / h[2])
  h = lorentz(0, 2, -y) * lorentz(1, 3, -y) * h
  let z: Ld = atan2(h[2], h[3])
  return hyperpoint(x, y, z, 0)
}
function normalizeFlat(h: Hyperpoint): Hyperpoint {
  if (gproduct) {
    if (geom3_euc_in_product()) {
      let bz: Ld = zlevel(h)
      let h1 = h / exp(bz)
      let bx: Ld = atan_auto(h1[0] / h1[2])
      return zpush(bz) * xpush(bx) * C0
    }
    return product_decompose(h).second
  }
  if (geom3_euc_in_nil()) {
    h[1] = 0
  }
  if (geom3_euc_in_sl2()) {
    let h1: Hyperpoint = esl2_ati(h)
    h1[1] = 0
    return esl2_ita0(h1)
  } else {
    if (sl2) {
      h = slr_translate(h) * zpush0(-atan2(h[2], h[3]))
    }
  }
  if (geom3_euc_in_solnih()) {
    h[2] = 0
  }
  if (geom3_hyp_in_solnih()) {
    h[0] = 0
  }
  if (geom3_euc_in_sph()) {
    let tx: Ld = hypot(h[0], h[2])
    let ty: Ld = hypot(h[1], h[3])
    h[0] = (h[0] / tx) * sin(1)
    h[1] = (h[1] / ty) * cos(1)
    h[2] = (h[2] / tx) * sin(1)
    h[3] = (h[3] / ty) * cos(1)
    return h
  }
  if (geom3_euc_in_hyp()) {
    h = normalize(h)
    let h1 = deparabolic13(h)
    h1[2] = 0
    return parabolic13(h1)
  }
  if (geom3_sph_in_euc()) {
    let z: Ld = hypot_d(3, h)
    if (z > 0) {
      h[0] /= z
      h[1] /= z
      h[2] /= z
    }
    h[3] = 1
    return h
  }
  if (geom3_sph_in_hyp()) {
    let z: Ld = hypot_d(3, h)
    z = sinh(1) / z
    if (z > 0) {
      h[0] *= z
      h[1] *= z
      h[2] *= z
    }
    h[3] = cosh(1)
    return h
  }
  return normalize(h)
}
function mid(h1: Hyperpoint, h2: Hyperpoint): Hyperpoint {
  if (gproduct) {
    let d1 = product_decompose(H1)
    let d2 = product_decompose(H2)
    let res1: Hyperpoint = PIU(mid(d1.second, d2.second))
    let res: Hyperpoint = res1 * exp((d1.first + d2.first) / 2)
    return res
  }
  return normalize(H1 + H2)
}
function mid(h1: Shiftpoint, h2: Shiftpoint): Shiftpoint {
  return shiftless(mid(H1.h, H2.h), (H1.shift + H2.shift) / 2)
}
function midz(h1: Hyperpoint, h2: Hyperpoint): Hyperpoint {
  if (gproduct) {
    return mid(H1, H2)
  }
  let h3: Hyperpoint = H1 + H2
  let z: Ld = 2
  if (!euclid) {
    Z = (zlevel(H3) * 2) / (zlevel(H1) + zlevel(H2))
  }
  for (let c: number = 0; c < MXDIM; c++) {
    H3[c] /= Z
  }
  return H3
}
function cspin(a: number, b: number, alpha: Ld): Transmatrix {
  let t: Transmatrix = Id
  T[a][a] = +cos(alpha)
  T[a][b] = +sin(alpha)
  T[b][a] = -sin(alpha)
  T[b][b] = +cos(alpha)
  return T
}
function lorentz(a: number, b: number, v: Ld): Transmatrix {
  let t: Transmatrix = Id
  T[a][a] = T[b][b] = cosh(v)
  T[a][b] = T[b][a] = sinh(v)
  return T
}
function cspin90(a: number, b: number): Transmatrix {
  let t: Transmatrix = Id
  T[a][a] = 0
  T[a][b] = 1
  T[b][a] = -1
  T[b][b] = 0
  return T
}
function cspin180(a: number, b: number): Transmatrix {
  let t: Transmatrix = Id
  T[a][a] = T[b][b] = -1
  return T
}
function spin(alpha: Ld): Transmatrix {
  if (embedded_plane && geom3_euc_in_product()) {
    return Id
  }
  if (embedded_plane && geom3_euc_in_sl2()) {
    return Id
  }
  if (embedded_plane && geom3_euc_vertical()) {
    return cspin(0, 2, alpha)
  }
  if (embedded_plane && geom3_hyp_in_solnih()) {
    return cspin(1, 2, alpha)
  }
  return cspin(0, 1, alpha)
}
function unswapSpin(t: Transmatrix): Transmatrix {
  return (
    cgi.intermediate_to_logical_scaled *
    T *
    cgi.logical_scaled_to_intemediate
  )
}
function spin90(): Transmatrix {
  if (embedded_plane && geom3_euc_in_product()) {
    return Id
  }
  if (embedded_plane && geom3_euc_vertical()) {
    return cspin90(0, 2)
  }
  if (embedded_plane && geom3_hyp_in_solnih()) {
    return cspin90(1, 2)
  }
  return cspin90(0, 1)
}
function spin180(): Transmatrix {
  if (embedded_plane && geom3_euc_in_product()) {
    return Id
  }
  if (embedded_plane && geom3_euc_vertical()) {
    return cspin180(0, 2)
  }
  if (embedded_plane && geom3_hyp_in_solnih()) {
    return cspin180(1, 2)
  }
  return cspin180(0, 1)
}
function spin270(): Transmatrix {
  if (embedded_plane && geom3_euc_in_product()) {
    return Id
  }
  if (embedded_plane && geom3_euc_vertical()) {
    return cspin90(2, 0)
  }
  if (embedded_plane && geom3_hyp_in_solnih()) {
    return cspin90(2, 1)
  }
  return cspin90(1, 0)
}
function randomSpin3(): Transmatrix {
  let alpha2: Ld = asin(randd() * 2 - 1)
  let alpha: Ld = randd() * TAU
  let alpha3: Ld = randd() * TAU
  return cspin(0, 1, alpha) * cspin(0, 2, alpha2) * cspin(1, 2, alpha3)
}
function randomSpin(): Transmatrix {
  if (WDIM == 2) {
    return spin(randd() * TAU)
  } else {
    return random_spin3()
  }
}
function eupush(x: Ld, y: Ld): Transmatrix {
  let t: Transmatrix = Id
  T[0][LDIM] = x
  T[1][LDIM] = y
  return T
}
function euclideanTranslate(x: Ld, y: Ld, z: Ld): Transmatrix {
  let t: Transmatrix = Id
  T[0][LDIM] = x
  T[1][LDIM] = y
  T[2][LDIM] = z
  return T
}
function euscale(x: Ld, y: Ld): Transmatrix {
  let t: Transmatrix = Id
  T[0][0] = x
  T[1][1] = y
  return T
}
function euscale3(x: Ld, y: Ld, z: Ld): Transmatrix {
  let t: Transmatrix = Id
  T[0][0] = x
  T[1][1] = y
  T[2][2] = z
  return T
}
function eupush(h: Hyperpoint, co): Transmatrix {
  if (nonisotropic) {
    return nisot_translate(h, co)
  }
  if (hyperbolic) {
    return co
      ? parabolic13_at(deparabolic13(h))
      : inverse(parabolic13_at(deparabolic13(h)))
  }
  let t: Transmatrix = Id
  for (let i: number = 0; i < GDIM; i++) {
    T[i][LDIM] = h[i] * co
  }
  return T
}
function eupush3(x: Ld, y: Ld, z: Ld): Transmatrix {
  if (sl2) {
    return slr_translate(slr_xyz_point(x, y, z))
  }
  return eupush(point3(x, y, z))
}
function euscalezoom(h: Hyperpoint): Transmatrix {
  let t: Transmatrix = Id
  T[0][0] = h[0]
  T[0][1] = -h[1]
  T[1][0] = h[1]
  T[1][1] = h[0]
  return T
}
function euaffine(h: Hyperpoint): Transmatrix {
  let t: Transmatrix = Id
  T[0][1] = h[0]
  T[1][1] = exp(h[1])
  return T
}
function cpush(cid: number, alpha: Ld): Transmatrix {
  if (gproduct && cid == 2) {
    return scale_matrix(Id, exp(alpha))
  }
  let t: Transmatrix = Id
  if (nonisotropic) {
    return eupush3(
      cid == 0 ? alpha : 0,
      cid == 1 ? alpha : 0,
      cid == 2 ? alpha : 0,
    )
  }
  T[LDIM][LDIM] = T[cid][cid] = cos_auto(alpha)
  T[cid][LDIM] = sin_auto(alpha)
  T[LDIM][cid] = -curvature() * sin_auto(alpha)
  return T
}
function lzpush(z: Ld): Transmatrix {
  if (geom3_hyp_in_solnih()) {
    return cpush(0, z)
  }
  if (geom3_euc_vertical()) {
    return cpush(1, z)
  }
  return cpush(2, z)
}
function cmirror(cid: number): Transmatrix {
  let t: Transmatrix = Id
  T[cid][cid] = -1
  return T
}
function xpush(alpha: Ld): Transmatrix {
  return cpush(0, alpha)
}
function lxpush(alpha: Ld): Transmatrix {
  if (embedded_plane) {
    geom3_light_flip(true)
    let t = cpush(0, alpha)
    geom3_light_flip(false)
    swapmatrix(t)
    return t
  }
  return cpush(0, alpha)
}
function eqmatrix(a: Transmatrix, b: Transmatrix, eps): Bool {
  for (let i: number = 0; i < MXDIM; i++) {
    for (let j: number = 0; j < MXDIM; j++) {
      if (std_abs(A[i][j] - B[i][j]) > eps) {
        return
      }
    }
  }
  return
}
function ypush(alpha: Ld): Transmatrix {
  return cpush(1, alpha)
}
function zpush(z: Ld): Transmatrix {
  return cpush(2, z)
}
function matrix3(
  a: Ld,
  b: Ld,
  c: Ld,
  d: Ld,
  e: Ld,
  f: Ld,
  g: Ld,
  h: Ld,
  i: Ld,
): Transmatrix {}
function matrix4(
  a: Ld,
  b: Ld,
  c: Ld,
  d: Ld,
  e: Ld,
  f: Ld,
  g: Ld,
  h: Ld,
  i: Ld,
  j: Ld,
  k: Ld,
  l: Ld,
  m: Ld,
  n: Ld,
  o: Ld,
  p: Ld,
): Transmatrix {}
function parabolic1(u: Ld): Transmatrix {
  if (euclid) {
    return ypush(u)
  } else {
    if (geom3_hyp_in_solnih() && !geom3_flipped) {
      return ypush(u)
    } else {
      let diag: Ld = (u * u) / 2
      return matrix3(-diag + 1, u, diag, -u, 1, u, -diag, u, diag + 1)
    }
  }
}
function parabolic13(u: Ld, v: Ld): Transmatrix {
  if (euclid) {
    return eupush3(0, u, v)
  } else {
    if (geom3_euc_in_hyp()) {
      let diag: Ld = (u * u + v * v) / 2
      return matrix4(
        1,
        0,
        -u,
        u,
        0,
        1,
        -v,
        v,
        u,
        v,
        -diag + 1,
        diag,
        u,
        v,
        -diag,
        diag + 1,
      )
    } else {
      let diag: Ld = (u * u + v * v) / 2
      return matrix4(
        -diag + 1,
        u,
        v,
        diag,
        -u,
        1,
        0,
        u,
        -v,
        0,
        1,
        v,
        -diag,
        u,
        v,
        diag + 1,
      )
    }
  }
}
function deparabolic13(h: Hyperpoint): Hyperpoint {
  if (euclid) {
    return h
  }
  if (geom3_euc_in_hyp()) {
    h /= 1 + h[LDIM]
    h[2] -= 1
    h /= sqhypot_d(LDIM, h)
    h[2] += 0.5
    return point3(h[0] * 2, h[1] * 2, log(2) + log(-h[2]))
  }
  h /= 1 + h[LDIM]
  h[0] -= 1
  h /= sqhypot_d(LDIM, h)
  h[0] += 0.5
  return point3(log(2) + log(-h[0]), h[1] * 2, LDIM == 3 ? h[2] * 2 : 0)
}
function parabolic13(h: Hyperpoint): Hyperpoint {
  if (euclid) {
    return h
  } else {
    if (geom3_euc_in_hyp()) {
      return parabolic13(h[0], h[1]) * cpush0(2, h[2])
    } else {
      if (LDIM == 3) {
        return parabolic13(h[1], h[2]) * xpush0(h[0])
      } else {
        return parabolic1(h[1]) * xpush0(h[0])
      }
    }
  }
}
function parabolic13At(h: Hyperpoint): Transmatrix {
  if (euclid) {
    return rgpushxto0(h)
  } else {
    if (geom3_euc_in_hyp()) {
      return parabolic13(h[0], h[1]) * cpush(2, h[2])
    } else {
      if (LDIM == 3) {
        return parabolic13(h[1], h[2]) * xpush(h[0])
      } else {
        return parabolic1(h[1]) * xpush(h[0])
      }
    }
  }
}
function spintoc(h: Hyperpoint, t: number, f: number): Transmatrix {
  let t: Transmatrix = Id
  let r: Ld = hypot(H[f], H[t])
  if (R >= 1e-15) {
    T[t][t] = +H[t] / R
    T[t][f] = +H[f] / R
    T[f][t] = -H[f] / R
    T[f][f] = +H[t] / R
  }
  return T
}
function rspintoc(h: Hyperpoint, t: number, f: number): Transmatrix {
  let t: Transmatrix = Id
  let r: Ld = hypot(H[f], H[t])
  if (R >= 1e-15) {
    T[t][t] = +H[t] / R
    T[t][f] = -H[f] / R
    T[f][t] = +H[f] / R
    T[f][f] = +H[t] / R
  }
  return T
}
function spintox(h: Hyperpoint): Transmatrix {
  if (GDIM == 2 || gproduct) {
    return spintoc(H, 0, 1)
  }
  let t1: Transmatrix = spintoc(H, 0, 1)
  return spintoc(T1 * H, 0, 2) * T1
}
function rspintox(h: Hyperpoint): Transmatrix {
  if (GDIM == 2 || gproduct) {
    return rspintoc(H, 0, 1)
  }
  let t1: Transmatrix = spintoc(H, 0, 1)
  return rspintoc(H, 0, 1) * rspintoc(T1 * H, 0, 2)
}
function lspintox(h: Hyperpoint): Transmatrix {
  if (geom3_euc_in_product()) {
    return Id
  }
  if (geom3_euc_in_sl2()) {
    return Id
  }
  if (geom3_euc_vertical()) {
    return spintoc(H, 0, 2)
  }
  if (geom3_hyp_in_solnih()) {
    return spintoc(H, 1, 2)
  }
  if (WDIM == 2 || gproduct) {
    return spintoc(H, 0, 1)
  }
  let t1: Transmatrix = spintoc(H, 0, 1)
  return spintoc(T1 * H, 0, 2) * T1
}
function lrspintox(h: Hyperpoint): Transmatrix {
  if (geom3_euc_in_product()) {
    return Id
  }
  if (geom3_euc_in_sl2()) {
    return Id
  }
  if (geom3_euc_vertical()) {
    return rspintoc(H, 0, 2)
  }
  if (geom3_hyp_in_solnih()) {
    return rspintoc(H, 2, 1)
  }
  if (WDIM == 2 || gproduct) {
    return rspintoc(H, 0, 1)
  }
  let t1: Transmatrix = spintoc(H, 0, 1)
  return rspintoc(H, 0, 1) * rspintoc(T1 * H, 0, 2)
}
function pushxto0(h: Hyperpoint): Transmatrix {
  let t: Transmatrix = Id
  T[0][0] = +H[LDIM]
  T[0][LDIM] = -H[0]
  T[LDIM][0] = curvature() * H[0]
  T[LDIM][LDIM] = +H[LDIM]
  return T
}
function setColumn(t: Transmatrix, i: number, h: Hyperpoint): void {
  for (let j: number = 0; j < MXDIM; j++) {
    T[j][i] = H[j]
  }
}
function getColumn(t: Transmatrix, i: number): Hyperpoint {
  let h: Hyperpoint
  for (let j: number = 0; j < MXDIM; j++) {
    h[j] = T[j][i]
  }
  return h
}
function buildMatrix(
  h1: Hyperpoint,
  h2: Hyperpoint,
  h3: Hyperpoint,
  h4: Hyperpoint,
): Transmatrix {
  let t: Transmatrix
  for (let i: number = 0; i < MXDIM; i++) {
    T[i][0] = h1[i]
    T[i][1] = h2[i]
    T[i][2] = h3[i]
    if (MAXMDIM == 4) {
      T[i][3] = h4[i]
    }
  }
  return T
}
function rpushxto0(h: Hyperpoint): Transmatrix {
  let t: Transmatrix = Id
  T[0][0] = +H[LDIM]
  T[0][LDIM] = H[0]
  T[LDIM][0] = -curvature() * H[0]
  T[LDIM][LDIM] = +H[LDIM]
  return T
}
function ggpushxto0(h: Hyperpoint, co: Ld): Transmatrix {
  if (translatable) {
    return eupush(H, co)
  }
  if (gproduct) {
    let d = product_decompose(H)
    return scale_matrix(
      PIU(ggpushxto0(d.second, co)),
      exp(d.first * co),
    )
  }
  let res: Transmatrix = Id
  if (sqhypot_d(GDIM, H) < 1e-16) {
    return res
  }
  let fac: Ld = -curvature() / (H[LDIM] + 1)
  for (let i: number = 0; i < GDIM; i++) {
    for (let j: number = 0; j < GDIM; j++) {
      res[i][j] += H[i] * H[j] * fac
    }
  }
  for (let d: number = 0; d < GDIM; d++) {
    res[d][LDIM] = co * H[d]
    res[LDIM][d] = -curvature() * co * H[d]
  }
  res[LDIM][LDIM] = H[LDIM]
  return res
}
function gpushxto0(h: Hyperpoint): Transmatrix {
  return ggpushxto0(H, -1)
}
function rgpushxto0(h: Hyperpoint): Transmatrix {
  return ggpushxto0(H, 1)
}
function rgpushxto0(h: Shiftpoint): Shiftmatrix {
  return shiftless(rgpushxto0(H.h), H.shift)
}
function fixmatrix(t: Transmatrix): void {
  if (nonisotropic) {
  } else {
    if (cgflags & qAFFINE) {
    } else {
      if (gproduct) {
        let z = zlevel(tC0(T))
        T = scale_matrix(T, exp(-z))
        PIU(fixmatrix(T))
        T = scale_matrix(T, exp(+z))
      } else {
        if (euclid) {
          fixmatrix_euclid(T)
        } else {
          orthonormalize(T)
        }
      }
    }
  }
}
function fixmatrixEuclid(t: Transmatrix): void {
  for (let x: number = 0; x < GDIM; x++) {
    for (let y: number = 0; y <= x; y++) {
      let dp: Ld = 0
      for (let z: number = 0; z < GDIM; z++) {
        dp += T[z][x] * T[z][y]
      }
      if (y == x) {
        dp = 1 - sqrt(1 / dp)
      }
      for (let z: number = 0; z < GDIM; z++) {
        T[z][x] -= dp * T[z][y]
      }
    }
  }
  for (let x: number = 0; x < GDIM; x++) {
    T[LDIM][x] = 0
  }
  T[LDIM][LDIM] = 1
}
function orthonormalize(t: Transmatrix): void {
  for (let x: number = 0; x < MDIM; x++) {
    for (let y: number = 0; y <= x; y++) {
      let dp: Ld = 0
      for (let z: number = 0; z < MXDIM; z++) {
        dp += T[z][x] * T[z][y] * sig(z)
      }
      if (y == x) {
        dp = 1 - sqrt(sig(x) / dp)
      }
      for (let z: number = 0; z < MXDIM; z++) {
        T[z][x] -= dp * T[z][y]
      }
    }
  }
}
function fixRotation(rot: Transmatrix): void {
  // FIX: dynamicval<eGeometry> g(geometry, gSphere);
  fixmatrix(rot)
  for (let i: number = 0; i < 3; i++) {
    rot[i][3] = rot[3][i] = 0
  }
  rot[3][3] = 1
}
function det2(t: Transmatrix): Ld {
  return T[0][0] * T[1][1] - T[0][1] * T[1][0]
}
function det3(t: Transmatrix): Ld {
  let det: Ld = 0
  for (let i: number = 0; i < 3; i++) {
    det += T[0][i] * T[1][(i + 1) % 3] * T[2][(i + 2) % 3]
  }
  for (let i: number = 0; i < 3; i++) {
    det -= T[0][i] * T[1][(i + 2) % 3] * T[2][(i + 1) % 3]
  }
  return det
}
function det(t: Transmatrix): Ld {
  if (MDIM == 3) {
    return det3(T)
  } else {
    let det: Ld = 1
    let m: Transmatrix = T
    for (let a: number = 0; a < MDIM; a++) {
      let maxAt: number = a
      for (let b: number = a; b < MDIM; b++) {
        if (abs(M[b][a]) > abs(M[max_at][a])) {
          max_at = b
        }
      }
      if (max_at != a) {
        for (let c: number = a; c < MDIM; c++) {
          tie(M[max_at][c], M[a][c]) = make_pair(-M[a][c], M[max_at][c])
        }
      }
      if (!M[a][a]) {
        return 0
      }
      for (let b: number = a + 1; b < MDIM; b++) {
        let co: Ld = -M[b][a] / M[a][a]
        for (let c: number = a; c < MDIM; c++) {
          M[b][c] += M[a][c] * co
        }
      }
      det *= M[a][a]
    }
    return det
  }
}
function inverseError(t: Transmatrix): void {
  println(hlog, 'Warning: inverting a singular matrix: ', T)
}
function inverse3(t: Transmatrix): Transmatrix {
  let d: Ld = det(T)
  let t2: Transmatrix
  if (d == 0) {
    inverse_error(T)
    return Id
  }
  for (let i: number = 0; i < 3; i++) {
    for (let j: number = 0; j < 3; j++) {
      T2[j][i] =
        (T[(i + 1) % 3][(j + 1) % 3] * T[(i + 2) % 3][(j + 2) % 3] -
          T[(i + 1) % 3][(j + 2) % 3] * T[(i + 2) % 3][(j + 1) % 3]) /
        d
    }
  }
  return T2
}
function inverse(t: Transmatrix): Transmatrix {
  if (MDIM == 3) {
    return inverse3(T)
  } else {
    let t1: Transmatrix = T
    let t2: Transmatrix = Id
    for (let a: number = 0; a < MDIM; a++) {
      let best: number = a
      for (let b: number = a + 1; b < MDIM; b++) {
        if (abs(T1[b][a]) > abs(T1[best][a])) {
          best = b
        }
      }
      let b: number = best
      if (b != a) {
        for (let c: number = 0; c < MDIM; c++) {
          swap(T1[b][c], T1[a][c])
          swap(T2[b][c], T2[a][c])
        }
      }
      if (!T1[a][a]) {
        inverse_error(T)
        return Id
      }
      for (let b: number = a + 1; b <= GDIM; b++) {
        let co: Ld = -T1[b][a] / T1[a][a]
        for (let c: number = 0; c < MDIM; c++) {
          T1[b][c] += T1[a][c] * co
          T2[b][c] += T2[a][c] * co
        }
      }
    }
    for (let a: number = MDIM - 1; a >= 0; a--) {
      for (let b: number = 0; b < a; b++) {
        let co: Ld = -T1[b][a] / T1[a][a]
        for (let c: number = 0; c < MDIM; c++) {
          T1[b][c] += T1[a][c] * co
          T2[b][c] += T2[a][c] * co
        }
      }
      let co: Ld = 1 / T1[a][a]
      for (let c: number = 0; c < MDIM; c++) {
        T1[a][c] *= co
        T2[a][c] *= co
      }
    }
    return T2
  }
}
function orthoInverse(t: Transmatrix): Transmatrix {
  for (let i: number = 1; i < MDIM; i++) {
    for (let j: number = 0; j < i; j++) {
      swap(T[i][j], T[j][i])
    }
  }
  return T
}
function pseudoOrthoInverse(t: Transmatrix): Transmatrix {
  for (let i: number = 1; i < MXDIM; i++) {
    for (let j: number = 0; j < i; j++) {
      swap(T[i][j], T[j][i])
    }
  }
  for (let i: number = 0; i < MDIM - 1; i++) {
    T[i][MDIM - 1] = -T[i][MDIM - 1]
    T[MDIM - 1][i] = -T[MDIM - 1][i]
  }
  return T
}
function isoInverse(t: Transmatrix): Transmatrix {
  if (hyperbolic) {
    return pseudo_ortho_inverse(T)
  }
  if (sphere) {
    return ortho_inverse(T)
  }
  if (nil) {
    let u: Transmatrix = Id
    U[2][LDIM] = T[0][LDIM] * T[1][LDIM] - T[2][LDIM]
    U[1][LDIM] = -T[1][LDIM]
    U[2][1] = U[0][LDIM] = -T[0][LDIM]
    return U
  }
  if (euclid && !(cgflags & qAFFINE)) {
    let u: Transmatrix = Id
    for (let i: number = 0; i < MDIM - 1; i++) {
      for (let j: number = 0; j < MDIM - 1; j++) {
        U[i][j] = T[j][i]
      }
    }
    let h: Hyperpoint = U * tC0(T)
    for (let i: number = 0; i < MDIM - 1; i++) {
      U[i][MDIM - 1] = -h[i]
    }
    return U
  }
  return inverse(T)
}
function zInverse(t: Transmatrix): Transmatrix {
  return inverse(T)
}
function viewInverse(t: Transmatrix): Transmatrix {
  if (nonisotropic) {
    return inverse(T)
  }
  if (gproduct) {
    return z_inverse(T)
  }
  return iso_inverse(T)
}
function iviewInverse(t: Transmatrix): Transmatrix {
  if (nonisotropic) {
    return inverse(T)
  }
  if (gproduct) {
    return z_inverse(T)
  }
  return iso_inverse(T)
}
function hyperpoint(h: Hyperpoint): PairLd {
  let z: Ld = zlevel(h)
  return make_pair(z, scale_point(h, exp(-z)))
}
function hdist0(mh: Hyperpoint): Ld {
  switch (cgclass) {
    case gcHyperbolic: {
      return acosh(mh[LDIM])
    }
    case gcEuclid: {
      return hypot_d(GDIM, mh)
    }
    case gcSphere: {
      let res: Ld =
        mh[LDIM] >= 1 ? 0 : mh[LDIM] <= -1 ? M_PI : acos(mh[LDIM])
      return res
    }
    case gcProduct: {
      let d1 = product_decompose(mh)
      return hypot(PIU(hdist0(d1.second)), d1.first)
    }
    default: {
      return hypot_d(GDIM, mh)
    }
  }
}
function hdist0(mh: Shiftpoint): Ld {
  return hdist0(unshift(mh))
}
function circlelength(r: Ld): Ld {
  switch (cgclass) {
    case gcEuclid: {
      return TAU * r
    }
    case gcHyperbolic: {
      return TAU * sinh(r)
    }
    case gcSphere: {
      return TAU * sin(r)
    }
    default: {
      return TAU * r
    }
  }
}
function hdist(h1: Hyperpoint, h2: Hyperpoint): Ld {
  let iv: Ld = intval(h1, h2)
  switch (cgclass) {
    case gcEuclid: {
      return sqrt(iv)
    }
    case gcHyperbolic: {
      return 2 * asinh(sqrt(iv) / 2)
    }
    case gcSphere: {
      return 2 * asin_auto_clamp(sqrt(iv) / 2)
    }
    case gcProduct: {
      let d1 = product_decompose(h1)
      let d2 = product_decompose(h2)
      return hypot(
        PIU(hdist(d1.second, d2.second)),
        d1.first - d2.first,
      )
    }
    case gcSL2: {
      return hdist0(stretch_itranslate(h1) * h2)
    }
    default: {
      return sqrt(iv)
    }
  }
}
function hdist(h1: Shiftpoint, h2: Shiftpoint): Ld {
  return hdist(h1.h, unshift(h2, h1.shift))
}
function orthogonalMoveFol(h: Hyperpoint, fol: Double): Hyperpoint {
  if (GDIM == 2) {
    return scale_point(h, fol)
  } else {
    return orthogonal_move(h, fol)
  }
}
function orthogonalMoveFol(t: Transmatrix, fol: Double): Transmatrix {
  if (GDIM == 2) {
    return scale_matrix(T, fol)
  } else {
    return orthogonal_move(T, fol)
  }
}
function orthogonalMoveFol(t: Shiftmatrix, fol: Double): Shiftmatrix {
  if (GDIM == 2) {
    return scale_matrix(T, fol)
  } else {
    return orthogonal_move(T, fol)
  }
}
function scaleMatrix(t: Transmatrix, scaleFactor: Ld): Transmatrix {
  let res: Transmatrix
  for (let i: number = 0; i < MXDIM; i++) {
    for (let j: number = 0; j < MDIM; j++) {
      res[i][j] = t[i][j] * scale_factor
    }
    for (let j: number = MDIM; j < MXDIM; j++) {
      res[i][j] = t[i][j]
    }
  }
  return res
}
function scaleMatrix(t: Shiftmatrix, scaleFactor: Ld): Shiftmatrix {
  return shiftless(scale_matrix(t.T, scale_factor), t.shift)
}
function scalePoint(h: Hyperpoint, scaleFactor: Ld): Hyperpoint {
  let res: Hyperpoint
  for (let j: number = 0; j < MDIM; j++) {
    res[j] = h[j] * scale_factor
  }
  for (let j: number = MDIM; j < MXDIM; j++) {
    res[j] = h[j]
  }
  return res
}
function movedCenter(): Bool {
  if (geom3_sph_in_euc()) {
    return
  }
  if (geom3_sph_in_hyp()) {
    return
  }
  if (geom3_euc_in_sph()) {
    return
  }
  return
}
function tileCenter(): Hyperpoint {
  if (geom3_sph_in_euc()) {
    return C02 + C03
  }
  if (geom3_euc_in_sph()) {
    return zpush0(1)
  }
  if (geom3_sph_in_hyp()) {
    return zpush0(1)
  }
  return C0
}
function orthogonalMove(t: Transmatrix, level: Double): Transmatrix {
  if (gproduct && !geom3_euc_in_product()) {
    return scale_matrix(t, exp(level))
  }
  if (GDIM == 3) {
    return t * lzpush(level)
  }
  return scale_matrix(t, geom3_lev_to_factor(level))
}
function orthogonalMove(t: Shiftmatrix, level: Double): Shiftmatrix {
  return shiftless(orthogonal_move(t.T, level), t.shift)
}
function xyscale(t: Transmatrix, fac: Double): Transmatrix {
  let res: Transmatrix
  for (let i: number = 0; i < MXDIM; i++) {
    for (let j: number = 0; j < GDIM; j++) {
      res[i][j] = t[i][j] * fac
    }
    for (let j: number = GDIM; j < MXDIM; j++) {
      res[i][j] = t[i][j]
    }
  }
  return res
}
function xyzscale(
  t: Transmatrix,
  fac: Double,
  facz: Double,
): Transmatrix {
  let res: Transmatrix
  for (let i: number = 0; i < MXDIM; i++) {
    for (let j: number = 0; j < GDIM; j++) {
      res[i][j] = t[i][j] * fac
    }
    res[i][LDIM] = t[i][LDIM] * facz
    for (let j: number = LDIM + 1; j < MXDIM; j++) {
      res[i][j] = t[i][j]
    }
  }
  return res
}
function xyzscale(
  t: Shiftmatrix,
  fac: Double,
  facz: Double,
): Shiftmatrix {
  return shiftless(xyzscale(t.T, fac, facz), t.shift)
}
function mzscale(t: Transmatrix, fac: Double): Transmatrix {
  if (GDIM == 3) {
    return t * cpush(2, fac)
  }
  let tcentered: Transmatrix = gpushxto0(tC0(t)) * t
  fac -= 1
  let res: Transmatrix =
    t * inverse(tcentered) * ypush(-fac) * tcentered
  fac *= 0.2
  fac += 1
  for (let i: number = 0; i < MXDIM; i++) {
    for (let j: number = 0; j < MXDIM; j++) {
      res[i][j] = res[i][j] * fac
    }
  }
  return res
}
function mzscale(t: Shiftmatrix, fac: Double): Shiftmatrix {
  return shiftless(mzscale(t.T, fac), t.shift)
}
function mid3(
  h1: Hyperpoint,
  h2: Hyperpoint,
  h3: Hyperpoint,
): Hyperpoint {
  return mid(h1 + h2 + h3, h1 + h2 + h3)
}
function midAt(h1: Hyperpoint, h2: Hyperpoint, v: Ld): Hyperpoint {
  let h: Hyperpoint = h1 * (1 - v) + h2 * v
  return mid(h, h)
}
function midAtActual(h: Hyperpoint, v: Ld): Hyperpoint {
  return rspintox(h) * xpush0(hdist0(h) * v)
}
function orthogonalOfC0(
  h0: Hyperpoint,
  h1: Hyperpoint,
  h2: Hyperpoint,
): Hyperpoint {
  h0 /= h0[3]
  h1 /= h1[3]
  h2 /= h2[3]
  let w: Hyperpoint = h0
  let d1: Hyperpoint = h1 - h0
  let d2: Hyperpoint = h2 - h0
  let denom: Ld = (d1 | d1) * (d2 | d2) - (d1 | d2) * (d1 | d2)
  let a1: Ld = (d2 | w) * (d1 | d2) - (d1 | w) * (d2 | d2)
  let a2: Ld = (d1 | w) * (d1 | d2) - (d2 | w) * (d1 | d1)
  let h: Hyperpoint = w * denom + d1 * a1 + d2 * a2
  return normalize(h)
}
function hpxd(d: Ld, x: Ld, y: Ld, z: Ld): Hyperpoint {
  let h: Hyperpoint = hpxyz(d * x, d * y, z)
  H = mid(H, H)
  return H
}
function signum(x: Ld): Ld {
  return x < 0 ? -1 : x > 0 ? 1 : 0
}
function asign(y1: Ld, y2: Ld): Bool {
  return signum(y1) != signum(y2)
}
function xcross(x1: Ld, y1: Ld, x2: Ld, y2: Ld): Ld {
  return x1 + ((x2 - x1) * y1) / (y1 - y2)
}
let eEmbeddedShiftMethodChoiceEmbeddedShiftMethodChoice: Ex = smcBoth
function useEmbeddedShift(sma: EShiftMethodApplication): Bool {
  switch (sma) {
    case smaAutocenter:
    case smaAnimation: {
      return embedded_shift_method_choice != smcNone
    }
    case smaManualCamera: {
      return embedded_shift_method_choice == smcBoth
    }
    case smaObject: {
      return
    }
    case smaWallRadar: {
      return among(pmodel, mdLiePerspective, mdLieOrthogonal)
    }
    default: {
      throw hr_exception('unknown sma')
    }
  }
}
function shiftMethod(sma: EShiftMethodApplication): EShiftMethod {
  if (gproduct) {
    return smProduct
  }
  if (embedded_plane && sma == smaObject) {
    return geom3_same_in_same() ? smIsotropic : smEmbedded
  }
  if (embedded_plane && use_embedded_shift(sma)) {
    return sl2 ? smESL2 : nonisotropic ? smLie : smEmbedded
  }
  if (
    !nonisotropic &&
    !stretch_in() &&
    !(!nisot_geodesic_movement && hyperbolic && bt_in())
  ) {
    return smIsotropic
  }
  if (!nisot_geodesic_movement && !embedded_plane) {
    return smLie
  }
  return smGeodesic
}
function shiftObject(
  position: Transmatrix,
  ori: Transmatrix,
  direction: Hyperpoint,
  sm,
): Transmatrix {
  switch (sm) {
    case smGeodesic: {
      return nisot_parallel_transport(Position, direction)
    }
    case smLie: {
      return nisot_lie_transport(Position, direction)
    }
    case smProduct: {
      let h: Hyperpoint = product_direct_exp(ori * direction)
      return Position * rgpushxto0(h)
    }
    case smIsotropic: {
      return Position * rgpushxto0(direct_exp(direction))
    }
    case smEmbedded: {
      if (geom3_euc_in_hyp() || geom3_sph_in_low()) {
        geom3_light_flip(true)
        let t: Transmatrix = rgpushxto0(direct_exp(direction))
        geom3_light_flip(false)
        swapmatrix(T)
        return Position * T
      }
      if (geom3_euc_in_sph()) {
        Position = inverse(View) * Position
      }
      let rot: Transmatrix =
        inverse(map_relative_push(Position * tile_center())) * Position
      if (moved_center()) {
        rot = rot * lzpush(1)
      }
      let urot: Transmatrix = unswap_spin(rot)
      geom3_light_flip(true)
      let t: Transmatrix = rgpushxto0(direct_exp(urot * direction))
      geom3_light_flip(false)
      swapmatrix(T)
      let res = Position * inverse(rot) * T * rot
      if (geom3_euc_in_sph()) {
        res = View * res
      }
      return res
    }
    default: {
      throw hr_exception('unknown shift method in shift_object')
    }
  }
}
function applyShiftObject(
  position: Transmatrix,
  orientation: Transmatrix,
  direction: Hyperpoint,
  sm,
): void {
  Position = shift_object(Position, orientation, direction, sm)
}
function rotateObject(
  position: Transmatrix,
  orientation: Transmatrix,
  r: Transmatrix,
): void {
  if (geom3_euc_in_product()) {
    orientation = orientation * R
  } else {
    if (gproduct && WDIM == 3) {
      orientation = orientation * R
    } else {
      Position = Position * R
    }
  }
}
function spinTowards(
  position: Transmatrix,
  ori: Transmatrix,
  goal: Hyperpoint,
  dir: number,
  back: number,
): Transmatrix {
  let t: Transmatrix
  let alpha: Ld = 0
  if (nonisotropic && nisot_geodesic_movement) {
    T = nisot_spin_towards(Position, goal)
  } else {
    let u: Hyperpoint = inverse(Position) * goal
    if (gproduct) {
      let h: Hyperpoint = product_inverse_exp(U)
      alpha = asin_clamp(h[2] / hypot_d(3, h))
      U = product_decompose(U).second
    }
    T = rspintox(U)
  }
  if (back < 0) {
    T = T * spin180()
    alpha = -alpha
  }
  if (gproduct) {
    if (dir == 0) {
      ori = cspin(2, 0, alpha)
    }
    if (dir == 2) {
      ori = cspin(2, 0, alpha - _deg(90))
      dir = 0
    }
  }
  if (dir) {
    T = T * cspin(dir, 0)
  }
  T = Position * T
  return T
}
function spinTowards(
  position: Shiftmatrix,
  ori: Transmatrix,
  goal: Shiftpoint,
  dir: number,
  back: number,
): Shiftmatrix {
  return shiftless(
    spin_towards(
      Position.T,
      ori,
      unshift(goal, Position.shift),
      dir,
      back,
    ),
    Position.shift,
  )
}
function orthoError(t: Transmatrix): Ld {
  let err: Ld = 0
  for (let x: number = 0; x < 3; x++) {
    for (let y: number = 0; y < 3; y++) {
      let s: Ld = 0
      for (let z: number = 0; z < 3; z++) {
        s += T[z][x] * T[z][y]
      }
      s -= x == y
      err += s * s
    }
  }
  return err
}
function transpose(t: Transmatrix): Transmatrix {
  let result: Transmatrix
  for (let i: number = 0; i < MXDIM; i++) {
    for (let j: number = 0; j < MXDIM; j++) {
      result[j][i] = T[i][j]
    }
  }
  return result
}
function lspinpush0(alpha: Ld, x: Ld): Hyperpoint {
  let f: Bool = embedded_plane
  if (f) {
    geom3_light_flip(true)
  }
  if (embedded_plane) {
    throw hr_exception('still embedded plane')
  }
  let h: Hyperpoint = xspinpush0(alpha, x)
  if (f) {
    geom3_light_flip(false)
  }
  if (f) {
    swapmatrix(h)
  }
  return h
}
function xspinpush0(alpha: Ld, x: Ld): Hyperpoint {
  if (embedded_plane) {
    return lspinpush0(alpha, x)
  }
  if (sl2) {
    return slr_polar(x, -alpha, 0)
  }
  let h: Hyperpoint = Hypc
  h[LDIM] = cos_auto(x)
  h[0] = sin_auto(x) * cos(alpha)
  h[1] = sin_auto(x) * -sin(alpha)
  return h
}
function ctangent(c: number, x: Ld): Hyperpoint {
  return point3(c == 0 ? x : 0, c == 1 ? x : 0, c == 2 ? x : 0)
}
function xtangent(x: Ld): Hyperpoint {
  return ctangent(0, x)
}
function ztangent(z: Ld): Hyperpoint {
  return ctangent(2, z)
}
function lztangent(z: Ld): Hyperpoint {
  if (geom3_hyp_in_solnih()) {
    return ctangent(0, z)
  }
  if (geom3_euc_vertical()) {
    return ctangent(1, z)
  }
  return ctangent(2, z)
}
function tangentLength(dir: Hyperpoint, length: Ld): Hyperpoint {
  let r: Ld = hypot_d(GDIM, dir)
  if (!r) {
    return dir
  }
  return dir * (length / r)
}
function directExp(v: Hyperpoint): Hyperpoint {
  if (gproduct) {
    return product_direct_exp(v)
  }
  let d: Ld = hypot_d(GDIM, v)
  if (d > 0) {
    for (let i: number = 0; i < GDIM; i++) {
      v[i] = (v[i] * sin_auto(d)) / d
    }
  }
  v[LDIM] = cos_auto(d)
  return v
}
function inverseExp(h: Shiftpoint, prec): Hyperpoint {
  if (nil) {
    return nilv_get_inverse_exp(h.h, prec)
  }
  if (sl2) {
    return slr_get_inverse_exp(h)
  }
  if (gproduct) {
    return product_inverse_exp(h.h)
  }
  let d: Ld = acos_auto_clamp(h[GDIM])
  let v: Hyperpoint = Hypc
  if (d && sin_auto(d)) {
    for (let i: number = 0; i < GDIM; i++) {
      v[i] = (h[i] * d) / sin_auto(d)
    }
  }
  v[3] = 0
  return v
}
function geoDist(h1: Hyperpoint, h2: Hyperpoint, prec): Ld {
  if (!nonisotropic) {
    return hdist(h1, h2)
  }
  return hypot_d(
    3,
    inverse_exp(shiftless(nisot_translate(h1, -1) * h2, prec)),
  )
}
function geoDist(h1: Shiftpoint, h2: Shiftpoint, prec): Ld {
  if (!nonisotropic) {
    return hdist(h1, h2)
  }
  return hypot_d(
    3,
    inverse_exp(
      shiftless(nisot_translate(h1.h, -1) * h2.h, h2.shift - h1.shift),
      prec,
    ),
  )
}
function geoDistQ(h1: Hyperpoint, h2: Hyperpoint, prec): Ld {
  let d = geo_dist(h1, h2, prec)
  if (elliptic && d > _deg(90)) {
    return M_PI - d
  }
  return d
}
function lpIapply(h: Hyperpoint): Hyperpoint {
  return nisot_local_perspective_used ? inverse(NLP) * h : h
}
function lpApply(h: Hyperpoint): Hyperpoint {
  return nisot_local_perspective_used ? NLP * h : h
}
function smalltangent(): Hyperpoint {
  return xtangent(0.1)
}
function cyclefix(a: Ld, b: Ld): void {
  while (a > b + M_PI) {
    a -= TAU
  }
  while (a < b - M_PI) {
    a += TAU
  }
}
function raddif(a: Ld, b: Ld): Ld {
  let d: Ld = a - b
  if (d < 0) {
    d = -d
  }
  if (d > TAU) {
    d -= TAU
  }
  if (d > M_PI) {
    d = TAU - d
  }
  return d
}
function bucketer(x: Ld): Unsigned {
  return unsigned(-100000)
}
function bucketer(h: Hyperpoint): Unsigned {
  let dx = 0
  if (gproduct) {
    let d = product_decompose(h)
    h = d.second
    dx += bucketer(d.first) * 50
    if (geom3_euc_in_product() && in_h2xe()) {
      h /= h[2]
    }
  }
  dx +=
    bucketer(h[0]) + 1000 * bucketer(h[1]) + 1000000 * bucketer(h[2])
  if (MDIM == 4) {
    dx += bucketer(h[3]) * 1000000001
  }
  if (elliptic) {
    dx = min(dx, -dx)
  }
  return dx
}
function lerp(a0: Hyperpoint, a1: Hyperpoint, x: Ld): Hyperpoint {
  return a0 + (a1 - a0) * x
}
function linecross(
  a: Hyperpoint,
  b: Hyperpoint,
  c: Hyperpoint,
  d: Hyperpoint,
): Hyperpoint {
  a /= a[LDIM]
  b /= b[LDIM]
  c /= c[LDIM]
  d /= d[LDIM]
  let bax: Ld = b[0] - a[0]
  let dcx: Ld = d[0] - c[0]
  let cax: Ld = c[0] - a[0]
  let bay: Ld = b[1] - a[1]
  let dcy: Ld = d[1] - c[1]
  let cay: Ld = c[1] - a[1]
  let res: Hyperpoint
  res[0] =
    (cay * dcx * bax + a[0] * bay * dcx - c[0] * dcy * bax) /
    (bay * dcx - dcy * bax)
  res[1] =
    (cax * dcy * bay + a[1] * bax * dcy - c[1] * dcx * bay) /
    (bax * dcy - dcx * bay)
  res[2] = 0
  res[3] = 0
  res[GDIM] = 1
  return normalize(res)
}
function inner2(h1: Hyperpoint, h2: Hyperpoint): Ld {
  return hyperbolic
    ? h1[LDIM] * h2[LDIM] - h1[0] * h2[0] - h1[1] * h2[1]
    : sphere
    ? h1[LDIM] * h2[LDIM] + h1[0] * h2[0] + h1[1] * h2[1]
    : h1[0] * h2[0] + h1[1] * h2[1]
}
function circumscribe(
  a: Hyperpoint,
  b: Hyperpoint,
  c: Hyperpoint,
): Hyperpoint {
  let h: Hyperpoint = C0
  b = b - a
  c = c - a
  if (euclid) {
    let b2: Ld = inner2(b, b) / 2
    let c2: Ld = inner2(c, c) / 2
    let det: Ld = c[1] * b[0] - b[1] * c[0]
    h = a
    h[1] += (c2 * b[0] - b2 * c[0]) / det
    h[0] += (c2 * b[1] - b2 * c[1]) / -det
    return h
  }
  if (inner2(b, b) < 0) {
    b = b / sqrt(-inner2(b, b))
    c = c + b * inner2(c, b)
    h = h + b * inner2(h, b)
  } else {
    b = b / sqrt(inner2(b, b))
    c = c - b * inner2(c, b)
    h = h - b * inner2(h, b)
  }
  if (inner2(c, c) < 0) {
    c = c / sqrt(-inner2(c, c))
    h = h + c * inner2(h, c)
  } else {
    c = c / sqrt(inner2(c, c))
    h = h - c * inner2(h, c)
  }
  if (h[LDIM] < 0) {
    h[0] = -h[0]
    h[1] = -h[1]
    h[LDIM] = -h[LDIM]
  }
  let i: Ld = inner2(h, h)
  if (i > 0) {
    h /= sqrt(i)
  } else {
    h /= -sqrt(-i)
  }
  return h
}
function inner3(h1: Hyperpoint, h2: Hyperpoint): Ld {
  return hyperbolic
    ? h1[LDIM] * h2[LDIM] -
        h1[0] * h2[0] -
        h1[1] * h2[1] -
        h1[2] * h2[2]
    : sphere
    ? h1[LDIM] * h2[LDIM] +
      h1[0] * h2[0] +
      h1[1] * h2[1] +
      h1[2] * h2[2]
    : h1[0] * h2[0] + h1[1] * h2[1]
}
function circumscribe(
  a: Hyperpoint,
  b: Hyperpoint,
  c: Hyperpoint,
  d: Hyperpoint,
): Hyperpoint {
  let ds
  for (let i: number = 0; i < 3; i++) {
    if (inner3(ds[i], ds[i]) < 0) {
      ds[i] = ds[i] / sqrt(-inner3(ds[i], ds[i]))
      for (let j: number = i + 1; j < 4; j++) {
        ds[j] = ds[j] + ds[i] * inner3(ds[i], ds[j])
      }
    } else {
      ds[i] = ds[i] / sqrt(inner3(ds[i], ds[i]))
      for (let j: number = i + 1; j < 4; j++) {
        ds[j] = ds[j] - ds[i] * inner3(ds[i], ds[j])
      }
    }
  }
  let: Hyperpoint = ds[3]
  if (h[3] < 0) {
    h = -h
  }
  let i: Ld = inner3(h, h)
  if (i > 0) {
    h /= sqrt(i)
  } else {
    h /= -sqrt(-i)
  }
  return h
}
function towardsInf(
  material: Hyperpoint,
  dir: Hyperpoint,
  dist,
): Hyperpoint {
  let t: Transmatrix = gpushxto0(material)
  let id: Hyperpoint = T * dir
  return rgpushxto0(material) * rspintox(id) * xpush0(dist)
}
function clockwise(h1: Hyperpoint, h2: Hyperpoint): Bool {
  return h1[0] * h2[1] > h1[1] * h2[0]
}
let worstPrecisionError
function samePointMayWarn(a: Hyperpoint, b: Hyperpoint): Bool {
  let d: Ld = hdist(a, b)
  if (d > 1e-2) {
    return
  }
  if (d > 1e-3) {
    throw hr_precision_error()
  }
  if (d > 1e-6 && worst_precision_error <= 1e-6) {
    addMessage('warning: precision errors are building up!')
  }
  if (d > worst_precision_error) {
    worst_precision_error = d
  }
  return
}
