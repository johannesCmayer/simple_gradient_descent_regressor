import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy.linalg as LA
import time
from Util import ProgressIndicator, truncate


def calc_ms_error_simple(data, b, m):
    total_error = 0

    for x, y in zip(data.T[0], data.T[1]):
        total_error += (y - (m * x + b)) ** 2
    return total_error / len(data)


def calc_ms_error_projected(data, b, m):
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
    for x, y in zip(data, proj):
        errors.append(np.power(x - y, 2))

    errors = np.array(errors)
    return errors.mean()


def naive_stepper(b, m, points, learning_rate):
    ms_error = calc_ms_error_projected(points, b, m)
    ms_error_m_inc = calc_ms_error_projected(points, b, m + learning_rate * ms_error)

    for i in range(20):
        if ms_error_m_inc < ms_error:
            m += learning_rate
        else:
            m -= learning_rate

    for i in range(20):
        ms_error = calc_ms_error_projected(points, b, m)
        ms_error_b_inc = calc_ms_error_projected(points, b + learning_rate * ms_error, m)
        if ms_error_b_inc < ms_error:
            b += learning_rate
        else:
            b -= learning_rate

    return b, m


def step_gradient(b_current, m_current, data, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = len(data)
    for x, y in zip(data.T[0], data.T[1]):
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m


def gradient_descent_runner(data, initial_b, initial_m, learning_rate, num_of_iterations):
    b = initial_b
    m = initial_m

    all_b = []
    all_m = []

    progress_indicator = ProgressIndicator(1, num_of_iterations)
    for i in range(num_of_iterations):
        progress_indicator.advance_iter(i, num_of_iterations)
        b, m = naive_stepper(b, m, np.array(data), learning_rate)
        all_b.append(b)
        all_m.append(m)
    progress_indicator.print_total_execution_time()
    return all_b, all_m


class GradientDescender:
    def __init__(self, initial_b=0, initial_m=0, learning_rate=0.0001, num_of_iterations=1000):
        self.b = initial_b
        self.m = initial_m
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.predicted = None
        self.ln_b_data = []
        self.ln_m_data = []

    def reset(self):
        self.b = self.initial_b
        self.m = self.initial_m
        self.predicted = None

    def fit_continuous(self, data):
        self.ln_b_data, self.ln_m_data = gradient_descent_runner(
            data, self.b, self.m, self.learning_rate, self.num_of_iterations)
        self.b = self.ln_b_data[-1]
        self.m = self.ln_m_data[-1]


    def predict(self, X):
        self.predicted = np.array([X, np.array(X) * self.m + self.b])
        return self.predicted

    def get_error(self, data, rmse=True):
        mse = calc_ms_error_simple(data, self.m, self.b)
        if rmse:
            return mse ** 0.5
        else:
            return mse

    def draw_result(self, data, draw_predicted=True):
        fig, ax = plt.subplots()

        x1, y1 = zip(data.T)
        ax.plot(x1, y1, 'ro', alpha=0.6)

        ln, = plt.plot([], [], 'r-', animated=True)

        def init():
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 140)
            return ln,

        speed = 40
        def update(frame):

            line_start_point = np.array([0, self.ln_b_data[int(frame*speed)]])
            line_end_point = np.array([100, 100 * self.ln_m_data[int(frame*speed)] + self.ln_b_data[int(frame*speed)]])
            x2, y2 = zip(line_start_point, line_end_point)
            ln.set_data(x2, y2)
            return ln,

        ani = anim.FuncAnimation(fig, update, frames=int(len(self.ln_b_data)/speed),
                            init_func=init, blit=True, repeat=False)

        plt.grid()
        plt.show()

        def draw_error(line_end_point, line_start_point):
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

def run():
    start_time = time.time()
    points = np.genfromtxt('data.csv', delimiter=',')

    grad_desender = GradientDescender(num_of_iterations=5000, learning_rate=0.0001)
    grad_desender.fit_continuous(points)

    grad_desender.predict(20)
    print('Final value b: {}\nFinal value m: {}'.format(grad_desender.b, grad_desender.m))
    print('Final RMSE: {0}'.format(truncate(grad_desender.get_error(points), 2)))
    grad_desender.draw_result(points)

def test_anim():

    def data_gen(t=0):
        cnt = 0
        while cnt < 1000:
            cnt += 1
            t += 0.1
            yield t, np.sin(2 * np.pi * t) * np.exp(-t / 10.)

    def init():
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(0, 80)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        return line,

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.grid()
    xdata, ydata = [], []

    def run(data):
        # update the data
        t, y = data
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = ax.get_xlim()

        if t >= xmax:
            ax.set_xlim(xmin, 2 * xmax)
            ax.figure.canvas.draw()
        line.set_data(xdata, ydata)

        return line,

    no_gc_ani_ref = anim.FuncAnimation(fig, run, data_gen, blit=True, interval=0.0001,
                                  repeat=True, init_func=init)
    plt.show()

if __name__ == '__main__':
    #test_anim()
    run()
