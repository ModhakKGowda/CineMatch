from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    user_id = None

    if request.method == "POST":
        user_id = request.form.get("user_id")

    return render_template(
        "index.html",
        user_id=user_id
    )


if __name__ == "__main__":
    app.run(debug=True)
