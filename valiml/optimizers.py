import numpy as np


# LEVENBERG - MARQUARDT
def lm(start_x, function, max_iters=10, alpha=1e9, max_iters_no_change=30, callback=None):
    n_input_dimensions = start_x.shape[0]
    y, jacobian = function(start_x, jacobian=True)
    n_output_dimensions = y.shape[0]
    
    A = np.zeros((n_output_dimensions + n_input_dimensions, n_input_dimensions))
    solution = np.zeros((n_output_dimensions + n_input_dimensions,))
    
    x = start_x
    n_iter = 1
    last_change_iter = 1
    while n_iter <= max_iters:
        A[:n_output_dimensions, :] = jacobian
        A[n_output_dimensions:, :] = np.sqrt(alpha) * np.identity(n_input_dimensions)
        solution[:n_output_dimensions] = -y
    
        lstsq_solution = np.linalg.lstsq(A, solution, rcond=None)[0]
        next_x = x + lstsq_solution
        next_y, new_jacobian = function(next_x, jacobian=True)
        
        if n_iter - last_change_iter > max_iters_no_change:
            break

        if next_y < y:
            x = next_x
            y = next_y
            alpha *= 0.8
            jacobian = new_jacobian
            last_change_iter = n_iter
        else:
            alpha *= 2.0

        if callback is not None:
            callback(n_iter, x, y, alpha)
            
        n_iter += 1
    
    return x, y


def gd_online(X, start_x, cost_function, max_iters=1000, step_size=0.1, max_iters_no_change=40, callback=None):
    y, jacobian = cost_function(start_x, 0, jacobian=True)

    is_step_size_function = callable(step_size)
    n_pass_iter = 1
    n_iter = 1
    x = start_x
    iter_step_size = step_size
    while n_pass_iter <= max_iters:
        for idx in range(X.shape[0]):
            iter_step_size = step_size(n_iter) if is_step_size_function else iter_step_size
            if len(jacobian.shape) == 1:
                next_x = x - iter_step_size * jacobian
            else:
                next_x = x - iter_step_size * jacobian.mean(axis=1)

            next_y, new_jacobian = cost_function(next_x, idx, jacobian=True)

            x = next_x
            y = next_y
            jacobian = new_jacobian

            if callback is not None:
                callback(n_iter, x, y, iter_step_size)

            n_iter += 1

        n_pass_iter += 1
    return x, y


def gd(start_x, cost_function, max_iters=1000, step_size=0.1, max_iters_no_change=40, callback=None):
    y, jacobian = cost_function(start_x, jacobian=True)

    is_step_size_function = callable(step_size)
    last_change_iter = 1
    n_iter = 1
    x = start_x
    iter_step_size = step_size
    while n_iter <= max_iters:
        iter_step_size = step_size(n_iter) if is_step_size_function else iter_step_size
        if len(jacobian.shape) == 1:
            next_x = x - iter_step_size * jacobian
        else:
            next_x = x - iter_step_size * jacobian.mean(axis=1)

        next_y, new_jacobian = cost_function(next_x, jacobian=True)

        if n_iter - last_change_iter > max_iters_no_change:
            break

        if next_y < y:
            x = next_x
            y = next_y
            if not is_step_size_function:
                step_size *= 2
            jacobian = new_jacobian
            last_change_iter = n_iter
        else:
            if not is_step_size_function:
                iter_step_size *= 0.8

        if callback is not None:
            callback(n_iter, x, y, step_size)

        n_iter += 1

    return x, y
