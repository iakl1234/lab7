import unittest
import numpy as np
from scipy.optimize import minimize


def reconcile_balance(x0, tolerance):
    # Целевая функция для минимизации отклонений
    def objective(x):
        return np.sum(((x - x0) / tolerance) ** 2)

    # Ограничения для баланса потоков
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - x[1] - x[2]},  # x1 - x2 - x3 = 0
        {'type': 'eq', 'fun': lambda x: x[2] - x[3] - x[4]},  # x3 - x4 - x5 = 0
        {'type': 'eq', 'fun': lambda x: x[4] - x[5] - x[6]}   # x5 - x6 - x7 = 0
    ]

    # Выполнение оптимизации
    result = minimize(objective, x0, constraints=constraints)

    if result.success:
        reconciled = result.x
        norm_dev = (reconciled - x0) / tolerance
        return reconciled, norm_dev
    else:
        raise ValueError("Оптимизация не удалась: " + result.message)


def reconcile_balance_2(x0, tolerance):
    # Целевая функция для минимизации отклонений
    def objective(x):
        return np.sum(((x - x0) / tolerance) ** 2)

    # Ограничения для баланса потоков
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - x[1] - x[2]- x[7]},  # x1 - x2 - x3 - x8 = 0
        {'type': 'eq', 'fun': lambda x: x[2] - x[3] - x[4]},  # x3 - x4 - x5 = 0
        {'type': 'eq', 'fun': lambda x: x[4] - x[5] - x[6] }  # x5 - x6 - x7 = 0
    ]

    # Выполнение оптимизации
    result = minimize(objective, x0, constraints=constraints)

    if result.success:
        reconciled = result.x
        norm_dev = (reconciled - x0) / tolerance
        return reconciled, norm_dev
    else:
        raise ValueError("Оптимизация не удалась: " + result.message)


def reconcile_balance_3(x0, tolerance):
    # Целевая функция для минимизации отклонений
    def objective(x):
        return np.sum(((x - x0) / tolerance) ** 2)

    # Ограничения для баланса потоков
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - x[1] - x[2]},  # x1 - x2 - x3 = 0
        {'type': 'eq', 'fun': lambda x: x[2] - x[3] - x[4]},  # x3 - x4 - x5 = 0
        {'type': 'eq', 'fun': lambda x: x[4] - x[5] - x[6]},  # x5 - x6 - x7 = 0
        {'type': 'eq', 'fun': lambda x: x[0] - 10 * x[1]}    # x1 = 10 * x2
    ]

    # Выполнение оптимизации
    result = minimize(objective, x0, constraints=constraints)

    if result.success:
        reconciled = result.x
        norm_dev = (reconciled - x0) / tolerance
        return reconciled, norm_dev
    else:
        raise ValueError("Оптимизация не удалась: " + result.message)


class TestBalanceReconciliation(unittest.TestCase):
    def setUp(self):
        self.x0 = np.array([10.0054919341489, 3.03265795024749, 6.83122010827837,
                            1.98478460320379, 5.09293357450987, 4.05721328676762,
                            0.991215230484718])
        self.tolerance = np.array([0.200109838682978, 0.1213063180099, 0.683122010827837,
                                   0.0396956920640758, 0.101858671490197, 0.0811442657353524,
                                   0.0198243046096944])
        self.expected_reconciled = np.array([10.056, 3.014, 7.041, 1.982, 5.059,
                                             4.067, 0.992])

    def test_reconciliation(self):
        reconciled, norm_dev = reconcile_balance(self.x0, self.tolerance)

        print("\n=== TestBalanceReconciliation ===")
        print("Ожидаемые значения:")
        print(self.expected_reconciled)
        print("Полученные значения:")
        print(np.round(reconciled, 3))
        print("Нормализованные отклонения:")
        print(np.round(norm_dev, 3))

        # Проверка скорректированных потоков с точностью до 3 знаков после запятой
        np.testing.assert_array_almost_equal(reconciled, self.expected_reconciled, decimal=3)


class TestBalanceReconciliation2(unittest.TestCase):
    def setUp(self):
        self.x0 = np.array([10.0054919341489, 3.03265795024749, 6.83122010827837,
                            1.98478460320379, 5.09293357450987, 4.05721328676762,
                            0.991215230484718, 6.66666])
        self.tolerance = np.array([0.200109838682978, 0.1213063180099, 0.683122010827837,
                                   0.0396956920640758, 0.101858671490197, 0.0811442657353524,
                                   0.0198243046096944, 0.666666])
        self.expected_reconciled = np.array([10.540, 2.836, 6.973, 1.963, 5.009,
                                             4.020, 0.989,0.731])

    def test_reconciliation(self):
        reconciled, norm_dev = reconcile_balance_2(self.x0, self.tolerance)

        print("\n=== TestBalanceReconciliation2 ===")
        print("Ожидаемые значения:")
        print(self.expected_reconciled)
        print("Полученные значения:")
        print(np.round(reconciled, 3))
        print("Нормализованные отклонения:")
        print(np.round(norm_dev, 3))

        # Проверка скорректированных потоков с точностью до 3 знаков после запятой
        np.testing.assert_array_almost_equal(reconciled, self.expected_reconciled, decimal=3)


class TestBalanceReconciliation3(unittest.TestCase):
    def setUp(self):
        self.x0 = np.array([10.0054919341489, 3.03265795024749, 6.83122010827837,
                            1.98478460320379, 5.09293357450987, 4.05721328676762,
                            0.991215230484718])
        self.tolerance = np.array([0.200109838682978, 0.1213063180099, 0.683122010827837,
                                   0.0396956920640758, 0.101858671490197, 0.0811442657353524,
                                   0.0198243046096944])
        # Ожидаемые значения будут вычислены автоматически при оптимизации

    def test_reconciliation(self):
        reconciled, norm_dev = reconcile_balance_3(self.x0, self.tolerance)

        print("\n=== TestBalanceReconciliation3 ===")
        print("Исходные значения:")
        print(self.x0)
        print("Полученные значения:")
        print(np.round(reconciled, 3))
        print("Нормализованные отклонения:")
        print(np.round(norm_dev, 3))

        # Проверка, что x1 = 10 * x2
        self.assertAlmostEqual(reconciled[0], 10 * reconciled[1], delta=1e-6)

        # Проверка балансовых уравнений
        self.assertAlmostEqual(reconciled[0] - reconciled[1] - reconciled[2], 0, delta=1e-6)
        self.assertAlmostEqual(reconciled[2] - reconciled[3] - reconciled[4], 0, delta=1e-6)
        self.assertAlmostEqual(reconciled[4] - reconciled[5] - reconciled[6], 0, delta=1e-6)


if __name__ == '__main__':
    unittest.main()