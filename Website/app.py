from flask import Flask, render_template, request # type: ignore
import os
from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
from gnews import GNews
import time


def get_urls(topic):
    global quickDict
    timer = time.time()
    googlenews=GoogleNews()
    googlenews.search(topic)
    lstOfDf = []
    for i in range(2):
        googlenews.get_page(i)
        result=googlenews.result()
        lstOfDf.append(pd.DataFrame(result))
    retLst = []
    quickDict = {}
    for df in lstOfDf:
        for i in range(len(df["link"])):
            url = df["link"][i][0:df["link"][i].find("&ved=")]
            article = Article(url)
            try:
                article.download()
                article.parse()
                quickDict[url] = article.text
                retLst.append(url)
            except:
                print("didnt work")
    return retLst

def get_article_content(url):
    return quickDict[url]

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

@app.route('/sent', methods=['POST'])
def sentiment():
    return render_template("sent.html")
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)