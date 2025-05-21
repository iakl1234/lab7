import pytest
from fastapi.testclient import TestClient
from lab5 import app, FlowMeasurement, BalanceConstraint, ReconciliationRequest, reconcile_flows
import numpy as np

client = TestClient(app)


@pytest.fixture
def simple_data():
    # Фикстура с базовым примером данных для тестирования: 1 входной поток и 2 выходных
    measurements = [
        FlowMeasurement(name="F1", initial_value=100.0, tolerance=5.0),
        FlowMeasurement(name="F2", initial_value=50.0, tolerance=5.0),
        FlowMeasurement(name="F3", initial_value=50.0, tolerance=5.0)
    ]
    constraints = [
        BalanceConstraint(inputs=["F1"], outputs=["F2", "F3"])  # F1 = F2 + F3
    ]
    return ReconciliationRequest(measurements=measurements, constraints=constraints)


def test_basic_reconciliation(simple_data):
    # Тест проверяет базовый сценарий согласования потоков
    reconciled, within_limits, success, message, global_test = reconcile_flows(
        simple_data.measurements,
        simple_data.constraints
    )


    assert success == True


    assert np.isclose(reconciled[0], reconciled[1] + reconciled[2])


    assert all(within_limits)


def test_global_test_pass():
    # Тест проверяет работу глобального теста
    measurements = [
        FlowMeasurement(name="A", initial_value=100, tolerance=1.0),
        FlowMeasurement(name="B", initial_value=100, tolerance=1.0)
    ]

    constraints = [BalanceConstraint(inputs=["A"], outputs=["B"])]

    _, _, _, _, gt = reconcile_flows(measurements, constraints)

    assert gt['is_passed'] == False


def test_validation_error():
    # Тест проверяет обработку невалидных данных
    invalid_data = {"measurements": [{}], "constraints": []}  # Неполные данные
    response = client.post("/reconcile", json=invalid_data)
    assert response.status_code == 422


def test_unmeasured_flows():
    # Тест проверяет обработку ненаблюдаемых потоков
    measurements = [
        FlowMeasurement(name="M1", initial_value=10.0, tolerance=1.0, is_measured=True),
        FlowMeasurement(name="U1", initial_value=20.0, tolerance=10.0, is_measured=False)  # Ненаблюдаемый
    ]
    # Ограничение M1 = U1
    constraints = [BalanceConstraint(inputs=["M1"], outputs=["U1"])]

    reconciled, _, _, _, _ = reconcile_flows(measurements, constraints)
    # Проверяем что значения согласованы согласно ограничению
    assert np.isclose(reconciled[0], reconciled[1])


def test_bounds_enforcement():
    # Тест проверяет соблюдение минимальных и максимальных значений
    measurements = [
        FlowMeasurement(name="X", initial_value=150.0, tolerance=10.0,
                        min_value=100.0, max_value=200.0),
    ]
    constraints = []  # Нет ограничений, только границы

    reconciled, within_limits, _, _, _ = reconcile_flows(measurements, constraints)
    # Проверка что результат в допустимом диапазоне
    assert 100 <= reconciled[0] <= 200
    assert within_limits[0] == True  # Проверка флага within_limits


@pytest.mark.parametrize("initial, expected", [
    (90, 100),  # Тест случая когда начальное значение ниже минимума
    (210, 200),  # Тест случая когда начальное значение выше максимума
])
def test_boundary_conditions(initial, expected):
    # Параметризованный тест граничных условий
    measurements = [
        FlowMeasurement(name="Y", initial_value=initial, tolerance=10.0,
                        min_value=100.0, max_value=200.0),
    ]
    constraints = []

    reconciled, _, _, _, _ = reconcile_flows(measurements, constraints)
    # Проверяем что результат "прижат" к границе
    assert np.isclose(reconciled[0], expected)