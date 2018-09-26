from itertools import product
from sklearn.gaussian_process import GaussianProcess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as sk_kern
import numpy as np
import matplotlib.pyplot as plt


def true_func(x):
    """
    正しい関数

    :param np.array x:
    :return: 関数値 y
    :rtype: np.array
    """
    y = x + np.sin(5 * x)
    return y

def plot_result(x_test, mean, std):
    plt.plot(x_test[:, 0], mean, color="C0", label="predict mean")
    plt.fill_between(x_test[:, 0], mean + std, mean - std, color="C0", alpha=.3,label= "1 sigma confidence")
    plt.plot(x_train, y_train, "o",label= "training data")

#練習出力
np.random.seed(1)
x_train = np.random.normal(0, 1., 20)
y_train = true_func(x_train) + np.random.normal(loc=0, scale=.1, size=x_train.shape)
xx = np.linspace(-3, 3, 200)
plt.scatter(x_train, y_train, label="Data")
plt.plot(xx, true_func(xx), "--", color="C0", label="True Function")
plt.legend()
plt.title("traning data")
plt.show()


kernel = sk_kern.RBF(1.0, (1e-3, 1e3)) + sk_kern.ConstantKernel(1.0, (1e-3, 1e3)) + sk_kern.WhiteKernel()
gp = GaussianProcessRegressor(normalize_y=True,
                               kernel=kernel,
                               optimizer="fmin_l_bfgs_b",
                               alpha=1e-10,
                               n_restarts_optimizer=3)
#reshape-1で行ベクトルに1で列数1にする
# X は (n_samples, n_features) の shape に変形する必要がある
gp.fit(x_train.reshape(-1, 1), y_train)

# パラメータ学習後のカーネルは self.kernel_ に保存される
gp.kernel_ # < RBF(length_scale=0.374) + 0.0316**2 + WhiteKernel(noise_level=0.00785)

# 予測は平均値と、オプションで 分散、共分散 を得ることが出来る
x_test = np.linspace(-3., 3., 200).reshape(-1, 1)
pred_mean, pred_std= gp.predict(x_test, return_std=True)

print(pred_std)

plot_result(x_test, pred_mean, pred_std)
plt.title("Prediction by Scikit-learn")
plt.legend()
plt.show()
# gp.fit(x, y)
