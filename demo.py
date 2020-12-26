from genre_classifier import GenreClassifier
from codecs import open
import time
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

print("Prepare classifier")
start_time = time.time()
classifier = GenreClassifier()
print("Classifier is ready")
print(time.time() - start_time, "sec")


@app.route("/", methods=["POST", "GET"])
def index_page(text="", length = 0, prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        logfile = open("ydf_demo_logs.txt", "a", "utf-8")
        print(text)
        print("<response>", file=logfile)
        print(text, file=logfile)
        prediction_message = classifier.get_prediction_message(text)
        print(prediction_message, file=logfile)
        print("</response>", file=logfile)
        length = len(prediction_message)
        print(length)
        logfile.close()
    return render_template('base.html', text=text, len=length, prediction_message=prediction_message)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)