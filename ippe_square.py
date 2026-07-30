import numpy as np
from camera import Camera


# ---------------------------------------------------------------------
# Algorithm 1 : IPPE(v, J) -> gamma, R1, R2
# ---------------------------------------------------------------------

def rotation_aligning_e3_to(v):
    """R_v (Eq. 16): the minimal rotation with R_v @ (0,0,1) = v (unit vector),
    via the closed-form Rodrigues 'shortest rotation between two vectors'."""
    e3 = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    k = np.cross(e3, v)
    cos_ang = e3 @ v
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    # to avoid denominator = 0
    if 1 + cos_ang < 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + cos_ang))


def rank1(X):
    """rank1(): X is symmetric 2x2 and (in exact arithmetic) exactly rank-1,
    positive semi-definite: X = b b^T. Recovers b via the dominant
    eigenpair (robust to small numerical noise pushing X off exact rank-1)."""
    eigvals, eigvecs = np.linalg.eigh(X)   # ascending order
    lam = max(eigvals[-1], 0.0)
    u = eigvecs[:, -1]
    return np.sqrt(lam) * u


def ippe_core(v, J):
    """Algorithm 1, verbatim: IPPE(v, J) -> gamma, R1, R2."""
    # step 1
    n = np.array([v[0], v[1], 1.0])
    Rv = rotation_aligning_e3_to(n)

    # step 2
    B = np.array([[1.0, 0.0, -v[0]], [0.0, 1.0, -v[1]]]) @ Rv[:, :2]

    # step 3
    A = np.linalg.solve(B, J)

    # step 4
    _, S, _ = np.linalg.svd(A)
    gamma = S[0]

    # step 5
    R22 = A / gamma

    # step 6
    b = rank1(np.eye(2) - R22.T @ R22)

    # step 7: r3 = r1 x r2 using the "+b" candidate columns
    r1u = np.array([R22[0, 0], R22[1, 0], b[0]])
    r2u = np.array([R22[0, 1], R22[1, 1], b[1]])
    r3 = np.cross(r1u, r2u)
    c, a = r3[:2], r3[2]

    # step 8
    def assemble(b_sign, c_sign):
        M = np.empty((3, 3))
        M[:2, :2] = R22
        M[:2, 2] = c_sign * c
        M[2, :2] = b_sign * b
        M[2, 2] = a
        return Rv @ M

    R1 = assemble(+1.0, +1.0)
    R2 = assemble(-1.0, -1.0)
    return gamma, R1, R2


# ---------------------------------------------------------------------
# Algorithm 2 : correspondence-based pipeline
# ---------------------------------------------------------------------

def dlt_homography(obj_xy, img_norm):
    """Step 2: exact planar homography (DLT / SVD), H33 normalized to 1."""
    A = []
    for (X, Y), (u, v) in zip(obj_xy, img_norm):
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def solve_translation(R, obj_pts, img_norm):
    """Step 10: given fixed R, solve t linearly and least-squares over
    ALL correspondences via q_i x (R X_i + t) = 0, i.e. [q_i]_x t = -[q_i]_x (R X_i).
    Stacking these (rank-2-per-point, redundant) constraints and solving by
    least squares is the concrete form of t = (W^T W)^-1 W^T b."""
    rows, rhs = [], []
    for Xi, qi in zip(obj_pts, img_norm):
        q = np.array([qi[0], qi[1], 1.0])
        Qx = np.array([[0, -q[2], q[1]],
                        [q[2], 0, -q[0]],
                        [-q[1], q[0], 0]])
        rows.append(Qx)
        rhs.append(-Qx @ (R @ Xi))
    W = np.vstack(rows)
    d = np.concatenate(rhs)
    t, *_ = np.linalg.lstsq(W, d, rcond=None)
    return t


def project(R, t, K, obj_pts):
    Xc = (R @ obj_pts.T).T + t
    uvw = (K @ Xc.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def refine_gauss_newton(R0, t0, K, obj_pts, img_pts, iters=15):
    """Final nonlinear polish on true pixel reprojection error (hand-rolled
    Gauss-Newton; rotation updated via a right-multiplied Rodrigues step so
    it stays on the rotation manifold)."""
    R, t = R0.copy(), t0.copy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def rodrigues(w):
        theta = np.linalg.norm(w)
        if theta < 1e-12:
            return np.eye(3)
        k = w / theta
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)

    for _ in range(iters):
        rows, res = [], []
        for Xw, uv in zip(obj_pts, img_pts):
            Xc = R @ Xw + t
            x, y, z = Xc
            res.append(fx * x / z + cx - uv[0])
            res.append(fy * y / z + cy - uv[1])

            dU_dXc = np.array([fx / z, 0, -fx * x / z ** 2])
            dV_dXc = np.array([0, fy / z, -fy * y / z ** 2])

            Xw_hat = np.array([[0, -Xw[2], Xw[1]],
                                [Xw[2], 0, -Xw[0]],
                                [-Xw[1], Xw[0], 0]])
            dXc_dtheta = -R @ Xw_hat
            dXc_dt = np.eye(3)

            rows.append(np.concatenate([dU_dXc @ dXc_dtheta, dU_dXc @ dXc_dt]))
            rows.append(np.concatenate([dV_dXc @ dXc_dtheta, dV_dXc @ dXc_dt]))

        Jm = np.array(rows)
        r = np.array(res)
        delta, *_ = np.linalg.lstsq(Jm, -r, rcond=None)
        R = R @ rodrigues(delta[:3])
        t = t + delta[3:]

    return R, t


def ippe_square(camera, obj_pts, img_pts):
    """
    Parameters
    ----------
    K : (3,3) array_like    Camera intrinsic matrix.
    obj_pts : (4,3) array_like
        Square marker corners, canonical (zero-centered) order:
        [(-s/2, s/2, 0), (s/2, s/2, 0), (s/2, -s/2, 0), (-s/2, -s/2, 0)]
    img_pts : (4,2) array_like   Corresponding detected pixel coordinates.

    Returns
    -------
    R : (3,3) ndarray   Estimated rotation (lowest reprojection-error candidate).
    t : (3,)  ndarray   Estimated translation.
    info : dict         info["solutions"] holds ALL candidates (R, t, err),
                        sorted by error, exposing the two-fold ambiguity;
                        info["gamma"] is the scale value from Algorithm 1.
    """
    K = camera.K
    obj_pts = np.asarray(obj_pts, dtype=np.float64)
    img_pts = np.asarray(img_pts, dtype=np.float64)
    assert obj_pts.shape == (4, 3) and img_pts.shape == (4, 2), \
        "IPPE_SQUARE requires exactly 4 zero-centered coplanar points"

    # step 1: normalize pixels by K^-1
    K_inv = camera.K_inv
    img_norm = (K_inv @ np.hstack([img_pts, np.ones((4, 1))]).T).T[:, :2]

    # step 2: homography, object plane (x,y) -> normalized image coords
    H = dlt_homography(obj_pts[:, :2], img_norm)

    # steps 3-7: J, the Jacobian of the homography's action at u0=0
    J = np.array([[H[0, 0] - H[0, 2] * H[2, 0], H[0, 1] - H[1, 2] * H[2, 1]],
                  [H[1, 0] - H[1, 2] * H[2, 0], H[1, 1] - H[1, 2] * H[2, 1]]])

    # step 8: v <- pi(H [0 0 1]^T)   (u0 = 0, the square's own center)
    v = np.array([H[0, 2], H[1, 2]])

    # step 9: Algorithm 1
    gamma, R1, R2 = ippe_core(v, J)

    # step 10: translation per candidate, then rank by reprojection error
    solutions = []
    for R0 in (R1, R2):
        t0 = solve_translation(R0, obj_pts, img_norm)
        R_ref, t_ref = refine_gauss_newton(R0, t0, K, obj_pts, img_pts)
        proj = project(R_ref, t_ref, K, obj_pts)
        err = np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1)))
        solutions.append((R_ref, t_ref, err))

    solutions.sort(key=lambda s: s[2])
    R_best, t_best, e_best = solutions[0]
    return R_best, t_best, e_best, {"solutions": solutions, "gamma": gamma}


if __name__ == "__main__":
    import cv2
    camera = Camera(800.0, 800.0, 320.0, 240.0)
    dist = np.zeros(5)
    s = 0.05
    objp = np.array([[-s/2, s/2, 0], [s/2, s/2, 0], [s/2, -s/2, 0], [-s/2, -s/2, 0]])

    cases = [
        ("Oblique view",         np.array([0.5, 0.7, 0.1]),  np.array([0.02, -0.01, 0.30])),
        ("Near fronto-parallel", np.array([0.03, 0.02, 0.0]), np.array([0.0, 0.0, 0.35])),
        ("Random", np.array([1.2, 0.9, -0.1]), np.array([0.12, -0.11, 0.50])),
        ("Random", np.array([0.53, 0.67, 0.98]), np.array([0.72, -0.91, 0.35])),
        ("Random", np.array([0.51, 0.72, 0.41]), np.array([0.02, -0.01, 0.30])),
    ]

    for name, rvec_t, tvec_t in cases:
        imgp, _ = cv2.projectPoints(objp, rvec_t, tvec_t, camera.K, dist)
        imgp = imgp.reshape(-1, 2)

        R_mine, t_mine, e_mine, info = ippe_square(camera, objp, imgp)
        rvec_mine, _ = cv2.Rodrigues(R_mine)

        _, rvecs_cv, tvecs_cv, errs_cv = cv2.solvePnPGeneric(
            objp, imgp, camera.K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        print(f"\n=== {name} ===")
        print("Ground truth   rvec/tvec:", rvec_t, tvec_t)
        print("From-scratch   rvec/tvec:", rvec_mine.ravel(), t_mine,
              f"(err={info['solutions'][0][2]:.6f} px)")
        print("OpenCV IPPE    rvec/tvec:", rvecs_cv[0].ravel(), tvecs_cv[0].ravel(),
              f"(err={errs_cv[0][0]:.6f} px)")
        print("From-scratch all-candidate errors:",
              [round(sol[2], 4) for sol in info["solutions"]])
        print("OpenCV        all-candidate errors:",
              [round(e[0], 4) for e in errs_cv])