from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CineMatch</title>
    </head>

    <body>
        <h1>🎬 CineMatch</h1>

        <h2>AI Movie Recommendation System</h2>

        <p>
            Welcome to CineMatch!
        </p>

        <p>
            Our recommendation engine will suggest movies
            based on user ratings and viewing preferences.
        </p>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True)
