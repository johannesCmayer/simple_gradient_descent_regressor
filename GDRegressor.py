import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import sys


def calc_mserror(b, m, data):
    line_start_point = np.array([0, b])
    line_end_point = np.array([100, 100 * m + b])

    line_end_point_local = line_end_point - line_start_point
    norm_line_vec_local = line_end_point_local / LA.norm(line_end_point_local)

    data_in_local_line_space = data - line_start_point

    proj_local = []
    for vec in data_in_local_line_space:
        a = (np.dot(vec, norm_line_vec_local) * norm_line_vec_local)
        proj_local.append(a)
    proj_local = np.array(proj_local)

    proj = proj_local + line_start_point

    errors = []
    for x3, y3 in zip(data, proj):
        errors.append(np.power(x3 - y3, 2))

    errors = np.array(errors)
    return errors.mean()


def step_gradient(b, m, points, learning_rate):
    # gradient descent
    mserror = calc_mserror(b, m, points)

    mserror_b_inc = calc_mserror(b + learning_rate * mserror, m, points)
    if mserror_b_inc < mserror:
        b += learning_rate
    else:
        b -= learning_rate

    mserror_m_inc = calc_mserror(b, m + learning_rate * mserror, points)
    if mserror_m_inc < mserror:
        m += learning_rate
    else:
        m -= learning_rate

    return b, m


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_of_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_of_iterations):
        print_progress(i, num_of_iterations)
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return b, m


class GradientDescender:
    def __init__(self, initial_b=0, initial_m=0, learning_rate=0.001, num_of_iterations=2000):
        self.initial_b = initial_b
        self.initial_m = initial_m
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.b = 0
        self.m = 0
        self.predicted = None

    def fit(self, data):
        self.b, self.m = gradient_descent_runner(data, self.initial_b, self.initial_m, self.learning_rate, self.num_of_iterations)

    def predict(self, X):
        self.predicted = np.array([X, np.array(X) * self.m + self.b])
        return self.predicted

    def draw_result(self, data, draw_predicted=True):
        x1, y1 = zip(data.T)
        plt.plot(x1, y1, 'ro', alpha=0.6)

        line_start_point = np.array([0, self.b])
        line_end_point = np.array([100, 100 * self.m + self.b])

        x2, y2 = zip(line_start_point, line_end_point)
        plt.plot(x2, y2, 'b-')

        line_end_point_local = line_end_point - line_start_point
        norm_line_vec_local = line_end_point_local / LA.norm(line_end_point_local)

        data_in_local_line_space = data - line_start_point

        projected_local = []
        for vec in data_in_local_line_space:
            a = (np.dot(vec, norm_line_vec_local) * norm_line_vec_local)
            projected_local.append(a)
        projected_local = np.array(projected_local)

        proj = projected_local + line_start_point

        for x3, y3 in zip(data, proj):
            x3, y3 = zip(x3, y3)
            plt.plot(x3, y3, 'b-', alpha=0.1)

        if draw_predicted and self.predicted is not None:
            x4, y4 = zip(self.predicted)
            plt.plot(x4, y4, 'go', alpha=1)

        plt.axis('equal')
        plt.grid()
        plt.show()



def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    grad_des = GradientDescender(num_of_iterations=2000)
    grad_des.fit(points)
    print(grad_des.predict(20))
    grad_des.draw_result(points, draw_predicted=True)


if __name__ == '__main__':
    run()
