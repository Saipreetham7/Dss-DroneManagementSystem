from flask import Flask, render_template, request
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import pickle

# Load the dataset
data_set = pd.read_csv('C:/Users/suchi/Downloads/drone_sys/final_sample.csv')

# Split the dataset into features and target variable
features = data_set.drop('Drone Model', axis=1)
target = data_set['Drone Model']

# Feature scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Load the trained model from the pickle file
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
    
# Calculate the accuracy of the model
accuracy_percentage = model.score(X_test, y_test) * 100    

app = Flask(__name__,template_folder="C:/Users/suchi/Downloads/drone_sys/template1")
client = MongoClient('mongodb://localhost:27017/')




@app.route('/', methods=[ 'GET','POST'])
def home():
    '''result_data = {}
    if request.method == 'POST':
        user_requirements = {}
        for criteria in features.columns:
            value = float(request.form.get(criteria, 0))
            user_requirements[criteria] = value

        # Scale user requirements
        user_requirements_scaled = scaler.transform(pd.DataFrame([user_requirements]))

        # Calculate the Euclidean Distance between user requirements and drone features for each drone model
        drone_scores = {}
        for idx, drone_features in enumerate(scaled_features):
            dist = distance.euclidean(user_requirements_scaled[0], drone_features)
            drone_scores[idx] = dist

        # Select the drone model with the minimum distance as the best choice
        best_drone_model_idx = min(drone_scores, key=drone_scores.get)
        best_drone_model = data_set["Drone Model"][best_drone_model_idx]

        #Store the result in the MongoDB database
        db = client['drone_management_system']
        collection = db['drone_collection_system']
        result_data = {
            'best_drone': best_drone_model,
            'score': drone_scores[best_drone_model_idx],
            'accuracy': accuracy_percentage
        }
        collection.insert_one(result_data)

        return render_template('result.html', best_drone=best_drone_model, score=drone_scores[best_drone_model_idx], accuracy=accuracy_percentage)'''

    return render_template('index.html',features=features)


@app.route('/result', methods=['GET','POST'])
def result():
        user_requirements = {}
        for criteria in features.columns:
            value = float(request.form.get(criteria, 0))
            user_requirements[criteria] = value

        # Scale user requirements
        user_requirements_scaled = scaler.transform(pd.DataFrame([user_requirements]))

        # Calculate the Euclidean Distance between user requirements and drone features for each drone model
        drone_scores = {}
        for idx, drone_features in enumerate(scaled_features):
            dist = distance.euclidean(user_requirements_scaled[0], drone_features)
            drone_scores[idx] = dist

        # Select the drone model with the minimum distance as the best choice
        best_drone_model_idx = min(drone_scores, key=drone_scores.get)
        best_drone_model = data_set["Drone Model"][best_drone_model_idx]

        #Store the result in the MongoDB database
        db = client['drone_management_system']
        collection = db['drone_collection_system']
        result_data = {
            'best_drone': best_drone_model,
            'score': drone_scores[best_drone_model_idx],
            'accuracy': accuracy_percentage
        }
        collection.insert_one(result_data)

        return render_template('result.html',best_drone=best_drone_model, score=drone_scores[best_drone_model_idx], accuracy=accuracy_percentage)
    
'''db = client['drone_management_system']
    collection = db['drone_collection_system']
    result_data = collection.find_one()
    if result_data:
        return render_template('result.html', best_drone=result_data['best_drone'], score=result_data['score'],
                               accuracy=result_data['accuracy'])
    else:
        return "No result data available."
    #result_data = collection.find_one()
    #return render_template('result.html', best_drone=result_data['best_drone'], score=result_data['score'], accuracy=result_data['accuracy'])
    #return render_template('result.html', best_drone=best_drone_model, score=result_data['score'], accuracy=result_data['accuracy'])'''


if __name__ == '__main__':
    app.run(debug='True')
