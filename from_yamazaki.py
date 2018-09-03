from itertools import product
#from sklearn.gaussian_process import GaussianProcess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import kernels as sk_kern

import GPy.kern as gp_kern
import GPy

class BayesianOptimizer:
    x_list = None
    y_list = None
    yielding_index = None
    k_band = 5
    verbose = False

    def __init__(self, params):
        self.params = params
        self.keys = []
        self.values = []
        for k, v in sorted(self.params.items()):
            self.keys.append(k)
            self.values.append(v)
    #propertyデコれーたで装飾された関数は読み取りしかできなくなる
    @property
    #max_iterを決める
    def n_pattern(self):
        return len(list(product(*self.values))) #2500

    def output(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def supply_next_param(self, max_iter=None):
        self.x_list = []
        self.y_list = []
        #list() = []
        #product 入力イテラブルのデカルト積です
        #2つのリストの各要素のすべての積の要素
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        #*は引数をすべてタプルとする
        all_parameters = list(product(*self.values))  # [(0.01, [0, 0], 0, [10, 10]), (0.01, [0, 0], 0, [15, 15]), ...
        print(all_parameters)
        index_space = [list(range(len(v))) for v in self.values]  # [[0,1,2,3,..][0,1,2....]]
        print(index_space)
        all_index_list = list(product(*index_space))  # (0,0),(0,1),(0,2)
        print(all_index_list)
        #print(self.x_list, self.y_list, all_parameters, all_index_list)

        # examine 2 random points initially
        idx = list(range(len(all_index_list))) # 50*50 = 2500
        #要素をシャッフルする
        np.random.shuffle(idx) #[242,424,22,290...]
        searched_index_list = []
        searched_index_list.append(index)
        for index in idx[:2]:
            param = self.to_param(all_parameters[index])
            #self.yielding_index = none =>
            self.yielding_index = all_index_list[index]
            yield param

        # Bayesian Optimization
        max_iter = int(min(max_iter or max(np.sqrt(self.n_pattern)*4, 20), self.n_pattern))  # 最大探索回数を適当に計算。
        for iteration in range(max_iter):
            k = 1 - np.exp(-iteration / max_iter * 3) * self.k_band  # kの値を徐々に減らして 探索重視 → 活用重視 にしてみる
            gp = self.create_gp_and_fit(np.array(self.x_list), np.array(self.y_list))

            mean_array, mse_array = gp.predict(all_index_list, return_std=True)
            next_index, acq_array = self.acquisition(mean_array, mse_array, k, excludes=searched_index_list)

            #print(mean_array, mse_array)
            #print(k)

            self.output("--- Most Expected Predictions")
            for acq, ps in sorted(zip(acq_array, all_parameters), reverse=True)[:3]:
                self.output("%.2f: %s" % (acq, list(zip(self.keys, ps))))
            self.output("--- Past Best Results")
            for acq, vs, ps in self.best_results(3):
                self.output("%.2f: %s" % (acq, list(zip(self.keys, vs))))

            if next_index in searched_index_list:
                break
            searched_index_list.append(next_index)
            self.yielding_index = all_index_list[next_index]
            yield self.to_param(all_parameters[next_index])

    @staticmethod #親クラスのメソッドに依存する
    def create_gp_and_fit(x, y, max_try=100):
        # この辺怪しい
        theta0 = 0.1
        for i in range(max_try+1):
            try:
                # gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=theta_u)
                #kernel = ConstantKernel() * RBF() + WhiteKernel()
                #kernel = sk_kern.RBF(length_scale=.5)
                #kernel = sk_kern.ExpSineSquared() * sk_kern.RBF()
                #kernel = sk_kern.ConstantKernel()
                #kernel = sk_kern.WhiteKernel(noise_level=3.)
                #kernel = GPy.kern.PeriodicExponential(lengthscale=.1, variance=3) * GPy.kern.Matern32(1)
                #kernel = GPy.kern.RatQuad(1, lengthscale=.5, variance=.3)
                #kernel = GPy.kern.White(input_dim=1)
                #kernel = C(10.0, (1e-8, 1e5)) * RBF([5,5], (1e-2, 1e2))
                #kernel=RBF()
                #kernel=GPy.kern.Matern52(2)
                #gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True, n_restarts_optimizer=3)
                #gp = GaussianProcessRegressor(alpha=0.001, normalize_y=True)
                #normalize_y = 予測変数の平均を 0 になるように正規化します
                #n_restarts_optimizer = カーネルのハイパーパラメータを最適化する回数です
                gp = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=3)
                gp.fit(x, y)
                print(gp)
                return gp
            except Exception as e:
                theta0 *= 10
                if i == max_try:
                    print(theta0)
                    raise e

    def to_param(self, row):
        return dict(zip(self.keys, row))

    def report(self, score):
        self.x_list.append(self.yielding_index)
        self.y_list.append(score)

    def best_results(self, n=5):
        index_list = []
        param_list = []
        for xs in self.x_list:
            values = [self.values[i][x] for i, x in enumerate(xs)]
            index_list.append(values)
            param_list.append(self.to_param(values))
        return sorted(zip(self.y_list, index_list, param_list), reverse=True)[:n]

    @staticmethod
    def acquisition(mean_array, mse_array, k, excludes=None):
        excludes = excludes or []
        values = mean_array + np.sqrt(mse_array) * k
        for_argmax = np.copy(values)
        #print(excludes, values, for_argmax)
        for ex in excludes:
            for_argmax[ex] = -np.Inf
        return np.argmax(for_argmax), values


def unknown_function(ps):
    return value_space[ps[0], ps[1]]

def plot_heatmap(value_space, checked_points):
    sns.heatmap(value_space, cmap='rainbow', vmin=np.min(value_space)*1.5, vmax=np.max(value_space)*1.5)

    cp = np.array(checked_points)
    #rows = value_space.shape[0] - cp[:, 0] - 1
    space_size = value_space.shape
    rows = space_size[0]-(value_space.shape[0]-cp[:, 0])
    cols = cp[:, 1]
    plt.plot(cols, rows, 'ro', markersize=5)
    plt.plot(cols[-1], rows[-1], 'o', markersize=10)

params = {
    "col": list(range(50)),
    "row": list(range(50)),
}

import sys
import numpy as np
bo = BayesianOptimizer(params)

all_parameters = list(product(*bo.values))  # [(0.01, [0, 0], 0, [10, 10]), (0.01, [0, 0], 0, [15, 15]), ...
print(all_parameters)



print(len(list(product(*bo.values))))
