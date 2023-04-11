import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

## ====================================== Plotting function =========================================== ##
def show_plot(y, img_name):
    y_values = [round(val, 4) for val in y]
    x = np.arange(len(y_values))  # the label locations
    width = 0.35  # the width of the bars
    labels = [str(n+1) for n in x]
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, y_values, width)
    #rects2 = ax.bar(x + width / 2, elu_array, width, label='elu activation fn')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg Log-Likelihood')
    ax.set_xlabel('GMM model order')
    ax.set_title('Avg Log-Likelihood in K-fold cross validation for '+img_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    #autolabel(rects2)
    fig.tight_layout()

## ===================== Reading images ========================= ##
plane = plt.imread('pics/plane.jpg')
bird = plt.imread('pics/bird.jpg')
'''
plt.figure()
plt.imshow(plane)
plt.figure()
plt.imshow(bird)
plt.show()
'''
## ======================= Preprocessing ======================= ##
plane_data = []
bird_data = []
for i in range(0, plane.shape[0]):
    for j in range(0, plane.shape[1]):
        pl_sample = [i/320.0, j/480.0, plane[i][j][0]/255.0, plane[i][j][1]/255.0, plane[i][j][2]/255.0]
        bd_sample = [i/320.0, j/480.0, bird[i][j][0]/255.0, bird[i][j][1]/255.0, bird[i][j][2]/255.0]
        plane_data.append(pl_sample)
        bird_data.append(bd_sample)
plane_data = np.array(plane_data)
bird_data = np.array(bird_data)
#plane_data = normalize(plane_data, axis=0)
#bird_data = normalize(bird_data, axis=0)
#print(len(bird_data), bird_data[0:10])

## ====================== Fitting GMM of order 2 ================== ##
gmm_plane = GaussianMixture(n_components=2, covariance_type='full')
y_plane = gmm_plane.fit_predict(plane_data).reshape(321, 481)
plt.figure()
plt.imshow(y_plane)

gmm_bird = GaussianMixture(n_components=2, covariance_type='full')
y_bird = gmm_bird.fit_predict(bird_data).reshape(321, 481)
plt.figure()
plt.imshow(y_bird)
#print(gmm_plane.weights_)
#print(gmm_bird.weights_)

## ======================= K-fold for plane ============================ ##
kf = KFold(n_splits=10)
ll_plane = []
for m in range(1, 8):
    p = []
    for train_index, test_index in kf.split(plane_data):
        x_train, x_test = plane_data[train_index], plane_data[test_index]
        gmm_plane = GaussianMixture(n_components=m, covariance_type='full')
        gmm_plane.fit(x_train)
        p.append(gmm_plane.score(x_test))
    print('Plane: M: ', m, ' ll:', np.mean(p))
    ll_plane.append(round(np.mean(p), 4))

best_ll_plane = max(ll_plane)
best_m_plane = ll_plane.index(best_ll_plane) + 1
print('Selected Model Order for plane: ', best_m_plane, ' with avg log-liklihood ', best_ll_plane)
show_plot(ll_plane, 'airplane')


## ======================= K-fold for bird ============================ ##
kf = KFold(n_splits=10 )
ll_bird = []
for m in range(1, 8):
    p = []
    for train_index, test_index in kf.split(bird_data):
        x_train, x_test = bird_data[train_index], bird_data[test_index]
        gmm_bird = GaussianMixture(n_components=m, covariance_type='full')
        gmm_bird.fit(x_train)
        p.append(gmm_bird.score(x_test))
    print('Bird: M: ', m, ' ll:', np.mean(p))
    ll_bird.append(round(np.mean(p), 4))

best_ll_bird = max(ll_bird)
best_m_bird = ll_bird.index(best_ll_bird) + 1
print('Selected Model Order for bird: ', best_m_bird, ' with avg log-liklihood ', best_ll_bird)
show_plot(ll_bird, 'bird')

## ===================== Clustring  using selected model order ============= ##
best_gmm_plane = GaussianMixture(n_components=best_m_plane, covariance_type='full')
y_plane = best_gmm_plane.fit_predict(plane_data).reshape(321, 481)
plt.figure()
plt.imshow(y_plane)

best_gmm_bird = GaussianMixture(n_components=best_m_bird, covariance_type='full')
y_bird = best_gmm_bird.fit_predict(bird_data).reshape(321, 481)
plt.figure()
plt.imshow(y_bird)

## ============================================================== ##
plt.show()
