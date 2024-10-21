from flask import Flask, render_template

# webserver gateway interface
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('layout.html')

if __name__ =="__main__":
    app.run(debug=True)