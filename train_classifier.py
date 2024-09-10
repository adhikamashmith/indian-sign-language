import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# 1. Inspect Data Dimensions
print("Shape of input data array:", data.shape)

# 2. Visualize Data Samples (if applicable)
# You can use matplotlib or other plotting libraries to visualize data samples

# 3. Explore Data Statistics
print("Mean of input data:", np.mean(data))
print("Standard deviation of input data:", np.std(data))
# You can calculate other statistics as needed

# 4. Print Data Samples
# Print a few samples of input data along with their corresponding labels
num_samples_to_print = 5
for i in range(num_samples_to_print):
    print("Sample {}: Data = {}, Label = {}".format(i+1, data[i], labels[i]))








x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
