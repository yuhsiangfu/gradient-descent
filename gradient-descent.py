"""
Exercise: Gradient Descent

@auth: Yu-Hsiang Fu
@date: 2018/08/07
"""
# --------------------------------------------------------------------------------
# 1.Import packages
# --------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import sys


# --------------------------------------------------------------------------------
# 2.Const variables
# --------------------------------------------------------------------------------
# plot variable
PLOT_X_SIZE = 5
PLOT_Y_SIZE = 5
PLOT_DPI = 300
PLOT_FORMAT = 'png'


# --------------------------------------------------------------------------------
# 3.Define function
# --------------------------------------------------------------------------------
def gradient_descent(x_data, y_data, bias, weight, rate_learning, num_iteration):
    bias_history = [bias]
    weight_history = [weight]
    bias_rate = 0.0
    weight_rate = 0.0

    # DO gradient-descent
    for t in range(num_iteration):
        bias_grad = 0.0
        weight_grad = 0.0

        for i in range(len(x_data)):
            # dL/db = 2 * y - (b + w * x) * -1
            bias_grad += 2.0 * (y_data[i] - bias - (weight * x_data[i])) * -1.0

            # dL/dw = 2 * y - (b + w * x) * -x
            weight_grad += 2.0 * (y_data[i] - bias - (weight * x_data[i])) * -x_data[i]

        # Adagrad
        bias_rate += pow(bias_grad, 2)
        weight_rate += pow(weight_grad, 2)

        # update bias and weight
        bias += -(rate_learning / np.sqrt(bias_rate)) * bias_grad
        weight += -(rate_learning / np.sqrt(weight_rate)) * weight_grad

        # store updated-results
        bias_history.append(bias)
        weight_history.append(weight)

    # final results
    print("final results")
    print("b:", bias_history[-1])
    print("w:", weight_history[-1])

    return (bias_history, weight_history)


def draw_contour_plot(x_data, y_data, theta_history):
    # plot-xy-range
    x_left, x_right = -80, 140
    y_left, y_right = -20, 20

    # plot-grid
    num_level = 20
    x = np.linspace(x_left, x_right, num_level)
    y = np.linspace(y_left, y_right, num_level)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            bias = x[i]
            weight = y[j]

            for k in range(len(x_data)):
                Z[j][i] += pow(y_data[k] - (bias + weight * x_data[k]), 2)

            Z[j][i] /= len(x_data)

    # --------------------------------------------------
    # unpack data
    bias_history, weight_history = theta_history

    # create a figure
    fig, ax = plt.subplots(figsize=(PLOT_X_SIZE, PLOT_Y_SIZE), facecolor="w")

    # draw plot
    ax.plot(bias_history, weight_history, "o-", ms=7, lw=1.2, mew=1.2, markevery=2, fillstyle="none", c="b")

    # --------------------------------------------------
    # plot setting
    ax.contourf(X, Y, Z, num_level, alpha=0.5, cmap=plt.get_cmap("jet"))
    ax.grid(color="k", linestyle="dotted", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(r"$b$", fontdict={"fontsize": 12})
    ax.set_ylabel(r"$w$", fontdict={"fontsize": 12})
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_left, y_right)
    # ax.set_xticks(())
    # ax.set_yticks(())
    ax.tick_params(axis="both", direction="in", which="major", labelsize=8)

    # save the figure
    image_path = "gradient-descent.{0}".format(PLOT_FORMAT)
    plt.tight_layout()
    plt.savefig(image_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight', pad_inches=0.05)
    plt.close()


# --------------------------------------------------------------------------------
# 4.Main function
# --------------------------------------------------------------------------------
def main_function():
    # data: y = x^2
    x_data = list(np.random.uniform(-10, 10, 100))
    y_data = [pow(x, 2) for x in x_data]

    # --------------------------------------------------
    # linear-regression model: y = b + (w * x), parameters: theda = (b, w)
    bias_initial = -70   # or np.random.uniform(-80.0, 140.0)
    weight_initial = 18  # or np.random.uniform(-20, 20)

    # train model by using gradient descent, theta_history = (bias_history, weight_history)
    rate_learning = 1
    num_iteration = 10000
    theta_history = gradient_descent(x_data, y_data, bias_initial, weight_initial, rate_learning, num_iteration)

    # --------------------------------------------------
    # draw a contour-plot of parameters' changes
    draw_contour_plot(x_data, y_data, theta_history)


if __name__ == '__main__':
    main_function()
