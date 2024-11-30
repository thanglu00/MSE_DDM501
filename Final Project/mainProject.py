from flask import Flask, request, jsonify, render_template_string
import mlflow.pyfunc
import pandas as pd

# Tạo ứng dụng Flask
app = Flask(__name__)

# Load mô hình tốt nhất từ MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/Best_Classifier_Model/Production")

# Trang chủ với form HTML để người dùng nhập dữ liệu
@app.route('/', methods=['GET'])
def index():
    html = """
    <!doctype html>
    <title>Predict with ML Model</title>
    <h2>Enter Data for Prediction</h2>
    <form action="/predict" method="post">
      <label>Feature 1: <input type="text" name="feature1"></label><br>
      <label>Feature 2: <input type="text" name="feature2"></label><br>
      <label>Feature 3: <input type="text" name="feature3"></label><br>
      <label>Feature 3: <input type="text" name="feature4"></label><br>
      <label>Feature 3: <input type="text" name="feature5"></label><br>
      <label>Feature 3: <input type="text" name="feature6"></label><br>
      <label>Feature 3: <input type="text" name="feature7"></label><br>
      <label>Feature 3: <input type="text" name="feature8"></label><br>
      <label>Feature 3: <input type="text" name="feature9"></label><br>
      <label>Feature 3: <input type="text" name="feature10"></label><br>
      <label>Feature 3: <input type="text" name="feature11"></label><br>
      <label>Feature 3: <input type="text" name="feature12"></label><br>
      <label>Feature 3: <input type="text" name="feature13"></label><br>
      <label>Feature 3: <input type="text" name="feature14"></label><br>
      <label>Feature 3: <input type="text" name="feature15"></label><br>
      <label>Feature 3: <input type="text" name="feature16"></label><br>
      <label>Feature 3: <input type="text" name="feature17"></label><br>
      <label>Feature 3: <input type="text" name="feature18"></label><br>
      <label>Feature 3: <input type="text" name="feature19"></label><br>
      <label>Feature 3: <input type="text" name="feature20"></label><br>
      <input type="submit" value="Predict">
    </form>
    """
    return render_template_string(html)

# Endpoint predict sử dụng POST để xử lý form hoặc JSON
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            # Dữ liệu được gửi dưới dạng JSON
            input_data = request.get_json()
            X_input = pd.DataFrame(input_data)
        else:
            # Dữ liệu được gửi từ form HTML
            features = [float(request.form[f'feature{i}']) for i in range(1, 21)]
            X_input = pd.DataFrame([features])

        # Dự đoán
        predictions = model.predict(X_input)

        return jsonify({"prediction": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})
    
# Chạy ứng dụng
if __name__ == '__main__':
    app.run(port=5001, debug=True)
