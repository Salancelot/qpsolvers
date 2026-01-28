#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors
# Copyright 2021 Dustin Kenefake

"""Solver interface for `COPT <https://www.shanshu.ai/solver>`__.

The COPT Optimizer suite ships several solvers for mathematical programming,
including problems that have linear constraints, bound constraints, integrality
constraints, cone constraints, or quadratic constraints. It targets modern CPU/GPU
architectures and multi-core processors,

See the :ref:`installation page <copt-install>` for additional instructions
on installing this solver.
"""

import warnings
from typing import Optional, Union

import coptpy
import numpy as np
import scipy.sparse as spa
from coptpy import COPT

from ..problem import Problem
from ..solution import Solution


def copt_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using COPT.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        This argument is not used by COPT.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Notes
    -----
    Keyword arguments are forwarded to COPT as parameters. For instance, we
    can call ``copt_solve_qp(P, q, G, h, u, FeasTol=1e-8,
    DualTol=1e-8)``. COPT settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``FeasTol``
         - Primal feasibility tolerance.
       * - ``DualTol``
         - Dual feasibility tolerance.
       * - ``TimeLimit``
         - Run time limit in seconds, 0 to disable.

    Check out the `Parameter Descriptions
    <https://guide.coap.online/copt/en-doc/parameter.html>`_
    documentation for all available COPT parameters.

    Lower values for primal or dual tolerances yield more precise solutions at
    the cost of computation time. See *e.g.* [Caron2022]_ for a primer of
    solver tolerances.
    """
    if initvals is not None:
        warnings.warn("warm-start values are ignored by this wrapper")

    env_config = coptpy.EnvrConfig()
    if not verbose:
        env_config.set("nobanner", "1")

    env = coptpy.Envr(env_config)
    model = env.createModel()

    if not verbose:
        model.setParam(COPT.Param.Logging, 0)
    for param, value in kwargs.items():
        model.setParam(param, value)

    P, q, G, h, A, b, lb, ub = problem.unpack()
    num_vars = P.shape[0]
    identity = np.eye(num_vars)
    x = model.addMVar(
        num_vars, lb=-COPT.INFINITY, ub=COPT.INFINITY, vtype=COPT.CONTINUOUS
    )
    ineq_constr, eq_constr, lb_constr, ub_constr = None, None, None, None
    if G is not None:
        ineq_constr = model.addMConstr(G, x, COPT.LESS_EQUAL, h)
    if A is not None:
        eq_constr = model.addMConstr(A, x, COPT.EQUAL, b)
    if lb is not None:
        lb_constr = model.addMConstr(identity, x, COPT.GREATER_EQUAL, lb)
    if ub is not None:
        ub_constr = model.addMConstr(identity, x, COPT.LESS_EQUAL, ub)
    objective = 0.5 * (x @ P @ x) + q @ x
    model.setObjective(objective, sense=COPT.MINIMIZE)
    model.solve()

    solution = Solution(problem)
    solution.extras["status"] = model.status
    solution.found = model.status in (COPT.OPTIMAL, COPT.IMPRECISE)
    if solution.found:
        # COPT v8.0.0+ Changed the default Python matrix modeling API
        #  from `numpy` to its own implementation.
        #  `coptpy.NdArray` does not support operators such as ">=",
        #  so convert to `np.ndarray`
        if hasattr(x.X, "tonumpy"):
            solution.x = x.X.tonumpy()
        else:
            solution.x = x.X
        __retrieve_dual(solution, ineq_constr, eq_constr, lb_constr, ub_constr)
    return solution


def __retrieve_dual(
    solution: Solution,
    ineq_constr: Optional[coptpy.MConstr],
    eq_constr: Optional[coptpy.MConstr],
    lb_constr: Optional[coptpy.MConstr],
    ub_constr: Optional[coptpy.MConstr],
) -> None:
    solution.z = -ineq_constr.Pi if ineq_constr is not None else np.empty((0,))
    solution.y = -eq_constr.Pi if eq_constr is not None else np.empty((0,))
    if lb_constr is not None and ub_constr is not None:
        solution.z_box = -ub_constr.Pi - lb_constr.Pi
    elif ub_constr is not None:  # lb_constr is None
        solution.z_box = -ub_constr.Pi
    elif lb_constr is not None:  # ub_constr is None
        solution.z_box = -lb_constr.Pi
    else:  # lb_constr is None and ub_constr is None
        solution.z_box = np.empty((0,))


def copt_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using COPT.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac{1}{2} x^T P x + q^T x \\
            \mbox{subject to}
                & G x \leq h                \\
                & A x = b                   \\
                & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `COPT <https://www.shanshu.ai/solver>`__.

    Parameters
    ----------
    P :
        Primal quadratic cost matrix.
    q :
        Primal quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        This argument is not used by COPT.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to COPT as parameters. For instance, we
    can call ``COPT_solve_qp(P, q, G, h, u, FeasTol=1e-8,
    DualTol=1e-8)``. COPT settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``FeasTol``
         - Primal feasibility tolerance.
       * - ``DualTol``
         - Dual feasibility tolerance.
       * - ``TimeLimit``
         - Run time limit in seconds, 0 to disable.

    Check out the `Parameter Descriptions
    <https://guide.coap.online/copt/en-doc/parameter.html>`_
    documentation for all available COPT parameters.

    Lower values for primal or dual tolerances yield more precise solutions at
    the cost of computation time. See *e.g.* [Caron2022]_ for a primer of
    solver tolerances.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = copt_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
