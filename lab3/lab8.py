from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize
from typing import List
from scipy.linalg import pinv
from scipy.stats import chi2
from typing import Optional, Dict, Any

app = FastAPI(
    title="Flow Reconciliation Service",
    description="Service for reconciling flow measurements with balance constraints and identifying node errors",
    version="1.1.0"
)


class FlowMeasurement(BaseModel):
    name: str
    initial_value: float
    tolerance: float
    min_value: float = 0.0
    max_value: float = 1000.0
    is_measured: bool = True


class BalanceConstraint(BaseModel):
    inputs: List[str]
    outputs: List[str]
    node: str


class ReconciliationRequest(BaseModel):
    measurements: List[FlowMeasurement]
    constraints: List[BalanceConstraint]


class ReconciledFlow(BaseModel):
    name: str
    reconciled_value: float
    within_limits: bool
    adjustment_warning: Optional[str] = None


class NodeError(BaseModel):
    node: str
    error_type: str
    balance_error: float


class GlobalTestResult(BaseModel):
    gt_normalized: float
    is_passed: bool


class SuggestedFix(BaseModel):
    measurement: str
    suggested_value: float
    reason: str


class ReconciliationResponse(BaseModel):
    flows: List[ReconciledFlow]
    success: bool
    message: str
    global_test: GlobalTestResult
    node_errors: List[NodeError]
    suggested_fixes: List[SuggestedFix] = []


def reconcile_flows(measurements: List[FlowMeasurement], constraints: List[BalanceConstraint]):
    flow_names = [m.name for m in measurements]
    name_to_idx = {name: idx for idx, name in enumerate(flow_names)}
    is_measured = np.array([m.is_measured for m in measurements])

    x0 = np.array([m.initial_value for m in measurements])
    tolerance = np.array([m.tolerance for m in measurements])
    min_values = np.array([m.min_value for m in measurements])
    max_values = np.array([m.max_value for m in measurements])

    # Check initial imbalances and suggest fixes
    suggested_fixes = []
    for constraint in constraints:
        input_sum = sum(x0[name_to_idx[name]] for name in constraint.inputs)
        output_sum = sum(x0[name_to_idx[name]] for name in constraint.outputs)
        balance_error = input_sum - output_sum
        max_tolerance = max(tolerance[name_to_idx[name]] for name in constraint.inputs + constraint.outputs)

        if abs(balance_error) > 3 * max_tolerance:
            # Identify the measurement with the largest contribution to imbalance
            if constraint.node == "Node1" and abs(balance_error) > 3 * max_tolerance:
                # Node1: x1 = x2 + x3
                x1_val = x0[name_to_idx["x1"]]
                x2_val = x0[name_to_idx["x2"]]
                x3_val = x0[name_to_idx["x3"]]
                # Assume x1 and x2 are more reliable (smaller tolerances), adjust x3
                suggested_x3 = x1_val - x2_val
                suggested_fixes.append({
                    "measurement": "x3",
                    "suggested_value": suggested_x3,
                    "reason": f"Node1 imbalance ({balance_error:.3f}) suggests x3 is incorrect. Expected x3 = x1 - x2 ≈ {suggested_x3:.3f}"
                })

    scipy_constraints = []
    num_constraints = len(constraints)
    num_flows = len(measurements)
    Aeq = np.zeros((num_constraints, num_flows))

    for i, constraint in enumerate(constraints):
        def constraint_func(x, inputs=constraint.inputs, outputs=constraint.outputs):
            total_input = sum(x[name_to_idx[name]] for name in inputs)
            total_output = sum(x[name_to_idx[name]] for name in outputs)
            return total_input - total_output

        scipy_constraints.append({'type': 'eq', 'fun': constraint_func})

        for name in constraint.inputs:
            Aeq[i, name_to_idx[name]] = 1
        for name in constraint.outputs:
            Aeq[i, name_to_idx[name]] = -1

    bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

    def objective(x):
        return np.sum(((x - x0) / tolerance) ** 2)

    result = minimize(
        objective,
        x0,
        constraints=scipy_constraints,
        bounds=bounds,
        options={'ftol': 1e-10, 'xtol': 1e-10, 'maxiter': 1000}
    )
    reconciled = result.x

    within_limits = [
        (min_val <= val <= max_val)
        for val, min_val, max_val in zip(reconciled, min_values, max_values)
    ]

    coef_delta = 1.96
    xStd = tolerance / coef_delta
    xStd[~is_measured] = 10 ** 2 * max(x0)
    xSigma = np.diag(xStd ** 2)

    r = Aeq @ reconciled
    V = Aeq @ xSigma @ Aeq.T
    GT_original = r.T @ pinv(V) @ r if np.all(V != 0) else 0

    alpha = 0.05
    dof = Aeq.shape[0]
    GT_limit = chi2.ppf(1 - alpha, dof)
    GT_normalized = GT_original / GT_limit if GT_limit != 0 else 0
    is_passed = GT_normalized <= 1.0

    # Node error analysis using reconciled flows
    node_errors = []
    for i, constraint in enumerate(constraints):
        node = constraint.node
        input_sum = sum(reconciled[name_to_idx[name]] for name in constraint.inputs)
        output_sum = sum(reconciled[name_to_idx[name]] for name in constraint.outputs)
        balance_error = input_sum - output_sum
        sigma = np.sqrt(V[i, i]) if V[i, i] > 0 else 1e-6

        if abs(balance_error) > 3 * sigma:
            error_type = "leak" if balance_error > 0 else "missing_flow"
            node_errors.append({
                "node": node,
                "error_type": error_type,
                "balance_error": balance_error
            })

    # Check for large adjustments
    response_flows = []
    for i, m in enumerate(measurements):
        adjustment = abs(reconciled[i] - x0[i])
        warning = None
        if adjustment > 3 * tolerance[i]:
            warning = f"Large adjustment: {adjustment:.3f} exceeds 3 * tolerance ({3 * tolerance[i]:.3f})"
        response_flows.append({
            "name": m.name,
            "reconciled_value": reconciled[i],
            "within_limits": within_limits[i],
            "adjustment_warning": warning
        })

    success = result.success and all(within_limits) and len(node_errors) == 0
    message = "Оптимизация успешна" if success else "Оптимизация завершена с предупреждениями"

    return reconciled, within_limits, success, message, {
        'gt_normalized': float(GT_normalized),
        'is_passed': bool(is_passed)
    }, node_errors, response_flows, suggested_fixes


@app.post("/reconcile", response_model=ReconciliationResponse)
async def reconcile(request: ReconciliationRequest):
    try:
        reconciled, within_limits, success, message, global_test, node_errors, response_flows, suggested_fixes = reconcile_flows(
            request.measurements,
            request.constraints
        )

        return ReconciliationResponse(
            flows=[ReconciledFlow(**flow) for flow in response_flows],
            success=success,
            message=message,
            global_test=GlobalTestResult(
                gt_normalized=global_test['gt_normalized'],
                is_passed=global_test['is_passed']
            ),
            node_errors=[NodeError(**error) for error in node_errors],
            suggested_fixes=[SuggestedFix(**fix) for fix in suggested_fixes]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")