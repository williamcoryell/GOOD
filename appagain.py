from flask import Flask, render_template, request # type: ignore
import os
from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd

def get_urls(topic):
    googlenews=GoogleNews()
    googlenews.search(topic)
    result=googlenews.result()
    df=pd.DataFrame(result)
    return [df["link"][i][0:df["link"][i].rfind("&ved=")] for i in range(len(df["link"]))]

def get_article_content(url):
    article = Article(url)
    print(url)
    print(article)
    print(article.text)
    return article.text

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html", og=True)

@app.route('/', methods=['POST'])
def text():
    topic = request.form['topic']
    urls = get_urls(topic)
    return render_template("index.html", topic=topic, go=True, urls=urls)

@app.route('/urlInfo', methods=['POST'])
def topic():
    topic = request.form['topicChose']
    texter = get_article_content(topic)
    return render_template("index.html", specificTopic=topic, topicText = texter)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
