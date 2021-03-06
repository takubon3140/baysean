from itertools import product
from sklearn.gaussian_process import GaussianProcess

class BaysianOptimize:
  verbose = False
  k_band = 5
  x_list = None
  y_list = None

  def __init__(self, params):
      self.params = params
      self.keys = []
      self.values = []
      for k, v in sorted(self.params.items()):
            self.keys.append(k)
            self.values.append(v)

　def supply_next_param(self, max_iter=None):
        self.x_list = []
        self.y_list = []
        all_parameters = list(product(*self.values))  # [(0.01, [0, 0], 0, [10, 10]), (0.01, [0, 0], 0, [15, 15]), ...
        index_space = [list(range(len(v))) for v in self.values]  # [[0], [0, 1, 2], [0], [0, 1, 2]]
        all_index_list = list(product(*index_space))  # [(0, 0, 0, 0), (0, 0, 0, 1), ...

        # examine 2 random points initially
        idx = list(range(len(all_index_list)))
        np.random.shuffle(idx)
        searched_index_list = []
        for index in idx[:2]:
            param = self.to_param(all_parameters[index])
            self.yielding_index = all_index_list[index]
            searched_index_list.append(index)
            yield param

    　　# Bayesian Optimization
        max_iter = int(min(max_iter or max(np.sqrt(self.n_pattern)*4, 20), self.n_pattern))  # 最大探索回数を適当に計算。
        for iteration in range(max_iter):
            k = 1 + np.exp(-iteration / max_iter * 3) * self.k_band  # kの値を徐々に減らして 探索重視 → 活用重視 にしてみる
            gp = self.create_gp_and_fit(np.array(self.x_list), np.array(self.y_list))

            mean_array, mse_array = gp.predict(all_index_list, eval_MSE=True)
            next_index, acq_array = self.acquisition(mean_array, mse_array, k, excludes=searched_index_list)

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

  def create_gp_and_fit(x, y, max_try=100):
        # この辺怪しい
        theta0 = 0.1
        for i in range(max_try+1):
            try:
                # gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=theta_u)
                gp = GaussianProcess(theta0=theta0)
                gp.fit(x, y)
                return gp

            except Exception as e:
                theta0 *= 10
                if i == max_try:
                    print(theta0)
                    raise e

  def acquisition(mean_array, mse_array, k, excludes=None):
        excludes = excludes or []
        values = mean_array + np.sqrt(mse_array) * k
        for_argmax = np.copy(values)
        for ex in excludes:
            for_argmax[ex] = -np.Inf
        #配列の最大値を返す
        return np.argmax(for_argmax), values



def main():
    params = {
    "col": list(range(50)),
    "row": list(range(50)),
    }
    print(params)
    bo = BaysianOptimize(params)

    checked_points = []

    for i, param in enumerate(bo.supply_next_param()):  # param is dict
      x = [param['row'], param['col']]
      y = unknown_function(x)
      bo.report(y)

      checked_points.append(x)


if __name__ == '__main__':
  main()
