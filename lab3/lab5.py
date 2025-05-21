from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.optimize import minimize
from typing import List
from scipy.linalg import pinv
from scipy.stats import chi2

app = FastAPI(
    title="Flow Reconciliation Service",
    description="Service for reconciling flow measurements with balance constraints",
    version="1.0.0"
)


class FlowMeasurement(BaseModel):
    """Модель измерения потока"""
    name: str
    initial_value: float
    tolerance: float
    min_value: float = 0.0
    max_value: float = 1000.0
    is_measured: bool = True


class BalanceConstraint(BaseModel):
    """Модель балансового ограничения"""
    inputs: List[str]
    outputs: List[str]


class ReconciliationRequest(BaseModel):
    """Модель запроса на согласование"""
    measurements: List[FlowMeasurement]
    constraints: List[BalanceConstraint]


class ReconciledFlow(BaseModel):
    """Модель результата согласованного потока"""
    name: str
    reconciled_value: float
    within_limits: bool


class GlobalTestResult(BaseModel):
    """Результат глобального теста"""
    gt_normalized: float
    is_passed: bool


class ReconciliationResponse(BaseModel):
    """Итоговый ответ сервиса"""
    flows: List[ReconciledFlow]
    success: bool
    message: str
    global_test: GlobalTestResult


def reconcile_flows(measurements: List[FlowMeasurement], constraints: List[BalanceConstraint]):
    """Основная функция согласования потоков"""

    # Подготовка данных измерений
    flow_names = [m.name for m in measurements]
    name_to_idx = {name: idx for idx, name in enumerate(flow_names)}
    is_measured = np.array([m.is_measured for m in measurements])

    # Инициализация начальных значений и параметров
    x0 = np.array([m.initial_value for m in measurements])
    tolerance = np.array([m.tolerance for m in measurements])
    min_values = np.array([m.min_value for m in measurements])
    max_values = np.array([m.max_value for m in measurements])

    # Подготовка ограничений для оптимизации
    scipy_constraints = []
    num_constraints = len(constraints)
    num_flows = len(measurements)
    Aeq = np.zeros((num_constraints, num_flows))

    # Формирование матрицы ограничений и функций ограничений
    for i, constraint in enumerate(constraints):
        def constraint_func(x, inputs=constraint.inputs, outputs=constraint.outputs):
            total_input = sum(x[name_to_idx[name]] for name in inputs)
            total_output = sum(x[name_to_idx[name]] for name in outputs)
            return total_input - total_output

        scipy_constraints.append({'type': 'eq', 'fun': constraint_func})

        # Заполнение матрицы ограничений
        for name in constraint.inputs:
            Aeq[i, name_to_idx[name]] = 1
        for name in constraint.outputs:
            Aeq[i, name_to_idx[name]] = -1

    # Установка граничных условий
    bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

    # Целевая функция для минимизации
    def objective(x):
        return np.sum(((x - x0) / tolerance) ** 2)  # Взвешенная сумма квадратов отклонений

    # Выполнение оптимизации
    result = minimize(objective, x0, constraints=scipy_constraints, bounds=bounds)
    reconciled = result.x

    # Проверка соблюдения предельных значений
    within_limits = [
        (min_val <= val <= max_val)
        for val, min_val, max_val in zip(reconciled, min_values, max_values)
    ]

    # Расчет параметров для глобального теста
    coef_delta = 1.96
    xStd = tolerance / coef_delta
    xStd[~is_measured] = 10 ** 2 * max(x0)
    xSigma = np.diag(xStd ** 2)

    # Расчет статистики глобального теста
    r = Aeq @ x0
    V = Aeq @ xSigma @ Aeq.T
    GT_original = r.T @ pinv(V) @ r

    alpha = 0.05  # Уровень значимости
    degree_of_freedom = Aeq.shape[0]
    GT_limit = chi2.ppf(1 - alpha, degree_of_freedom)

    # Нормировка результата теста
    GT_normalized = GT_original / GT_limit if GT_limit != 0 else 0
    is_passed = GT_normalized <= 1.0

    # Формирование итоговых результатов
    success = result.success and all(within_limits) and is_passed
    message = "Оптимизация успешна" if success else "Оптимизация не успешна"

    return reconciled, within_limits, success, message, {
        'gt_normalized': float(GT_normalized),
        'is_passed': bool(is_passed)
    }






@app.post("/reconcile", response_model=ReconciliationResponse)
async def reconcile(request: ReconciliationRequest):
    """Конечная точка API для выполнения согласования"""
    try:
        # Вызов основной логики
        reconciled, within_limits, success, message, global_test = reconcile_flows(
            request.measurements,
            request.constraints
        )

        # Формирование ответа
        response_flows = []
        for i, m in enumerate(request.measurements):
            response_flows.append(ReconciledFlow(
                name=m.name,
                reconciled_value=reconciled[i],
                within_limits=within_limits[i]
            ))

        return ReconciliationResponse(
            flows=response_flows,
            success=success,
            message=message,
            global_test=GlobalTestResult(
                gt_normalized=global_test['gt_normalized'],
                is_passed=global_test['is_passed']
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

