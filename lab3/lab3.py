import cvxpy as cp

# min_value = float('inf')
# best_x = None
# best_y = None
#
# for x in range(0, 55):
#     max_y1 = (200 - 3 * x) // 3
#     max_y2 = 380 - 7 * x
#     max_y = min(max_y1, max_y2)
#     max_y = max(max_y, 0)
#     for y in range(0, max_y + 1):
#         if 3 * x + 3 * y <= 200 and 7 * x + y <= 380:
#             current_value = 0.001 * (x**2 + y**2) - 7 * x - 4 * y
#             if current_value < min_value:
#                 min_value = current_value
#                 best_x = x
#                 best_y = y
#
# print(f"Минимум достигается при x = {best_x}, y = {best_y}")
# print(f"Значение функции: {min_value:.2f}")





import cvxpy as cp
x = cp.Variable(integer=True)
y = cp.Variable(integer=True)

objective = cp.Minimize(0.001 * x**2 + 0.001 * y**2 - 7 * x - 4 * y)
constraints = [
    3 * x + 3 * y <= 200,
    7 * x + y <= 380,
    x >= 0,
    y >= 0
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCIP)

print(f"Минимум достигается при x = {x.value}, y = {y.value}")
print(f"Значение функции = {problem.value}")