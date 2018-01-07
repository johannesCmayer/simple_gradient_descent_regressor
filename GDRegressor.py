import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA


def calc_mserror(b, m, points):
    return 1

def step_gradient(b, m, points, learning_rate):
    # gradient descent
    mserror = calc_mserror(b, m, points)

    mserror_b_inc = calc_mserror(b + learning_rate, m, points)
    if mserror_b_inc < mserror:
        b += learning_rate
    else:
        b -= learning_rate

    mserror_m_inc = calc_mserror(b, m + learning_rate, points)
    if mserror_m_inc < mserror:
        m += learning_rate
    else:
        m -= learning_rate

    return b, m


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_of_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_of_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    # Hyperparameter
    learning_rate = 0.0001

    initial_b = 50
    initial_m = -0.5
    num_of_iterations = 1000

    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_of_iterations)
    draw_result(np.array(points), b, m)


def draw_result(data, b, m):
    x1, y1 = zip(data.T)
    plt.plot(x1, y1, 'ro', alpha=0.6)

    x2, y2 = zip([0, b], [100, 100*m])
    plt.plot(x2, y2, 'b-')

    data_in_local_line_space = data - [0, b]

    line_vec = np.array([100, 100*m])
    norm_line_vec = line_vec / LA.norm(line_vec)

    proj = []
    for vec in data:
        proj.append(np.dot(vec, norm_line_vec) * norm_line_vec)

    proj = np.array(proj)
    x3, y3 = zip(proj.T)
    print(x3,y3)
    plt.plot(x3, y3, 'bo')

    plt.grid()
    plt.show()


if __name__ == '__main__':
    run()
