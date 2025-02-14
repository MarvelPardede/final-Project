# final-Project
<!DOCTYPE html>
<marvel>
<head>
    <title>Predicting Exercise Execution Using Wearable Sensor Data</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2 { color: #2C3E50; }
        .section { margin-bottom: 20px; }
        .code { background-color: #f4f4f4; padding: 10px; border-left: 5px solid #3498db; font-family: monospace; }
    </style>
</head>
<body>
    <h1>Predicting Exercise Execution Using Wearable Sensor Data</h1>
    <p><strong>Author:</strong> [Your Name]</p>
    <p><strong>Date:</strong> [Date]</p>

    <div class="section">
        <h2>Introduction</h2>
        <p>The goal of this project is to predict the manner in which exercises are performed based on accelerometer data collected from wearable sensors. This analysis aims to build a machine learning model that classifies execution style based on the dataset.</p>
    </div>

    <div class="section">
        <h2>Data Preprocessing</h2>
        <p>The dataset contains sensor readings from various body locations. Before training the model, preprocessing steps are required:</p>
        <ul>
            <li>Handling missing values</li>
            <li>Feature selection to remove redundant data</li>
            <li>Scaling and normalization</li>
        </ul>
        <div class="code">
            # Example preprocessing in Python<br>
            import pandas as pd<br>
            df = pd.read_csv("pml-training.csv")<br>
            df.dropna(axis=1, inplace=True)  # Remove columns with missing values
        </div>
    </div>

    <div class="section">
        <h2>Model Training & Validation</h2>
        <p>A Random Forest classifier was used for training due to its robustness with high-dimensional data.</p>
        <div class="code">
            from sklearn.ensemble import RandomForestClassifier<br>
            from sklearn.model_selection import train_test_split<br>
            <br>
            X_train, X_test, y_train, y_test = train_test_split(df.drop('classe', axis=1), df['classe'], test_size=0.2)<br>
            model = RandomForestClassifier(n_estimators=100)<br>
            model.fit(X_train, y_train)<br>
        </div>
    </div>

    <div class="section">
        <h2>Predictions & Evaluation</h2>
        <p>The trained model was tested on a separate dataset, and accuracy was computed.</p>
        <div class="code">
            from sklearn.metrics import accuracy_score<br>
            predictions = model.predict(X_test)<br>
            accuracy = accuracy_score(y_test, predictions)<br>
            print("Model Accuracy:", accuracy)<br>
        </div>
    </div>

    <div class="section">
        <h2>Conclusion</h2>
        <p>The Random Forest model demonstrated strong predictive power, successfully classifying exercise execution. Future improvements can include hyperparameter tuning and ensemble methods.</p>
    </div>
</body>
</html>
