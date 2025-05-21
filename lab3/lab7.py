from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize
from typing import List
from scipy.linalg import pinv
from scipy.stats import chi2
from typing import Optional, Dict, Any

# Инициализация FastAPI
app = FastAPI(
    title="Сервис сверки потоков",
    description="Сервис для сверки измерений потоков",
    version="1.1.0"
)

# Модель измерения потока
class FlowMeasurement(BaseModel):
    name: str
    initial_value: float
    tolerance: float
    min_value: float = 0.0
    max_value: float = 1000.0
    is_measured: bool = True

# Модель балансовых ограничений
class BalanceConstraint(BaseModel):
    inputs: List[str]
    outputs: List[str]
    node: str

# Модель запроса
class ReconciliationRequest(BaseModel):
    measurements: List[FlowMeasurement]
    constraints: List[BalanceConstraint]

# Модель результата потока
class ReconciledFlow(BaseModel):
    name: str
    reconciled_value: float
    within_limits: bool

# Модель ошибки узла
class NodeError(BaseModel):
    node: str
    error_type: str
    imbalance_value: float

# Модель результата глобального теста
class GlobalTestResult(BaseModel):
    gt_normalized: float
    is_passed: bool

# Модель ответа
class ReconciliationResponse(BaseModel):
    flows: List[ReconciledFlow]
    success: bool
    message: str
    global_test: GlobalTestResult
    node_errors: List[NodeError]

# Функция сверки потоков
def reconcile_flows(measurements: List[FlowMeasurement], constraints: List[BalanceConstraint]):
    # Извлечение имен и создание массивов
    flow_names = [m.name for m in measurements]
    name_to_idx = {name: idx for idx, name in enumerate(flow_names)}
    is_measured = np.array([m.is_measured for m in measurements])
    x0 = np.array([m.initial_value for m in measurements])
    tolerance = np.array([m.tolerance for m in measurements])
    min_values = np.array([m.min_value for m in measurements])
    max_values = np.array([m.max_value for m in measurements])

    # Формирование ограничений
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

    # Определение границ и целевой функции
    bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]
    def objective(x):
        return np.sum(((x - x0) / tolerance) ** 2)

    # Оптимизация
    result = minimize(objective, x0, constraints=scipy_constraints, bounds=bounds)
    reconciled = result.x

    # Проверка границ
    within_limits = [
        (min_val <= val <= max_val)
        for val, min_val, max_val in zip(reconciled, min_values, max_values)
    ]

    # Глобальный тест
    coef_delta = 1.96
    xStd = tolerance / coef_delta
    xStd[~is_measured] = 10 ** 2 * max(x0)
    xSigma = np.diag(xStd ** 2)
    r = Aeq @ x0
    V = Aeq @ xSigma @ Aeq.T
    GT_original = r.T @ pinv(V) @ r
    alpha = 0.05
    dof = Aeq.shape[0]
    GT_limit = chi2.ppf(1 - alpha, dof)
    GT_normalized = GT_original / GT_limit if GT_limit != 0 else 0
    is_passed = GT_normalized <= 1.0

    # Анализ ошибок узлов
    node_errors = []
    for i, constraint in enumerate(constraints):
        node = constraint.node
        input_sum = sum(x0[name_to_idx[name]] for name in constraint.inputs)
        output_sum = sum(x0[name_to_idx[name]] for name in constraint.outputs)
        balance_error = input_sum - output_sum
        sigma = np.sqrt(V[i, i]) if V[i, i] > 0 else 1e-6

        if abs(balance_error) > 3 * sigma:
            error_type = "leak" if balance_error > 0 else "missing_flow"
            node_errors.append({
                "node": node,
                "error_type": error_type
            })

    # Формирование результата
    success = result.success and all(within_limits) and is_passed and len(node_errors) == 0
    message = "Оптимизация успешна" if success else "Оптимизация завершена с нарушениями"

    return reconciled, within_limits, success, message, {
        'gt_normalized': float(GT_normalized),
        'is_passed': bool(is_passed)
    }, node_errors

# Эндпоинт для сверки
@app.post("/reconcile", response_model=ReconciliationResponse)
async def reconcile(request: ReconciliationRequest):
    try:
        # Вызов функции сверки
        reconciled, within_limits, success, message, global_test, node_errors = reconcile_flows(
            request.measurements,
            request.constraints
        )

        # Формирование ответа
        response_flows = [
            ReconciledFlow(
                name=m.name,
                reconciled_value=reconciled[i],
                within_limits=within_limits[i]
            ) for i, m in enumerate(request.measurements)
        ]

        return ReconciliationResponse(
            flows=response_flows,
            success=success,
            message=message,
            global_test=GlobalTestResult(
                gt_normalized=global_test['gt_normalized'],
                is_passed=global_test['is_passed']
            ),
            node_errors=node_errors
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")