import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary  # Make sure this import is correct
from players import aaron_judge, jose_altuve, david_ortiz  # Make sure player data imports are correct

fig, ax = plt.subplots()

# Explore the data
# print(aaron_judge.columns)
# print(aaron_judge.description.unique())
# print(aaron_judge.type.unique())

# Map 'type' to numerical values if necessary
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 2})

# Drop rows with NaN values in 'plate_x', 'plate_z', and 'type'
aaron_judge = aaron_judge.dropna(subset=['plate_x', 'plate_z', 'type'])

# Scatter plot of plate_x vs plate_z colored by 'type'
plt.scatter(x=aaron_judge['plate_x'], y=aaron_judge['plate_z'], c=aaron_judge['type'], cmap=plt.cm.coolwarm, alpha=0.25)
plt.show()

# Split data into training and validation sets
training_set, validation_set = train_test_split(aaron_judge, random_state=1)

# Train SVM classifier with default parameters
classifier = SVC(kernel='rbf')
aaron = training_set[['plate_x', 'plate_z']]
classifier.fit(aaron, training_set['type'])

# Visualize decision boundary
draw_boundary(ax, classifier)
plt.show()

# Evaluate accuracy on the validation set
accuracy = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
print(f"Accuracy of SVM (default parameters): {accuracy}")

# Train SVM classifier with adjusted parameters (gamma = 100, C = 100)
classifier_2 = SVC(kernel='rbf', gamma=100, C=100)
aaron2 = training_set[['plate_x', 'plate_z']]
classifier_2.fit(aaron2, training_set['type'])

# Visualize decision boundary with adjusted parameters
draw_boundary(ax, classifier_2)
plt.show()

# Evaluate accuracy on the validation set
accuracy_2 = classifier_2.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
print(f"Accuracy of SVM (gamma=100, C=100): {accuracy_2}")
