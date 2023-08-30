from xi_method.ximp import *

if __name__ == '__main__':
    Y = np.random.choice(np.array([1, 2, 3]), size=1000)
    X = np.random.normal(5, 5, size=25 * 1000)  # df_np[:, 1:11]
    X = X.reshape((1000, 25))
    #         start_time = time.time()
    xi = XIClassifier(m=10)
    p = xi.explain(X=X, y=Y, replicates=1, separation_measurement='Kullback-leibler')
    print(p.get('Kullback-leibler').explanation)
