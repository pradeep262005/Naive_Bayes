from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('spam_model.pkl', 'rb') as f:
    cv, model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = ""
    if request.method == 'POST':
        message = request.form['message']
        message_vec = cv.transform([message])
        pred = model.predict(message_vec)
        prediction = "Spam ❌" if pred[0] == 1 else "Not Spam ✅"
    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
