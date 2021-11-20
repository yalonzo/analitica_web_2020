from flask import Flask, render_template, request
import preprocess

## create an instance of the app
app = Flask(__name__)

## start URL page with decorator
@app.route("/")
def home():
    #return "<h1>Hola Mundo</h1>"
    return render_template("home.html")

## move to predict
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["text"]
        print(text)
        pred = preprocess.svm_classifier(text)
        print(pred)

        if pred[0] == 1:
            result = "LEADERSHIP"
        elif pred[0] == 0:
            result = "NO LEADERSHIP"

        print(result)

    return render_template("home.html", 
                            text_original = text,
                            pred = result)

if __name__ == "__main__":
    app.run(debug=True, port=8060)