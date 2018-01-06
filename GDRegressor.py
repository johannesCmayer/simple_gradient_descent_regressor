import numpy as np


def step_gradient(b, m, points, learning_rate):
    #gradient descent
    pass


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_of_iterations):
    for i in range(num_of_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    # Hyperparameter
    learning_rate = 0.0001

    initial_b = 0
    initial_m = 0
    num_of_iterations = 1000

    gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_of_iterations)


if __name__ == 'main':
    run()
