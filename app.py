from flask import Flask, render_template, request, jsonify  # type: ignore
import os
from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
from gnews import GNews
import time
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from twikit import Client
import random
import main as tspmo
global id2label
global label2id
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}


USERNAME = 'mickey76148'
PASSWORD = 'twitterScrapingTool'


topicCHOSEN = ""
URLS = []
quickDict  = {}

def get_urls_news(topic):
    global quickDict
    start_time = time.time()
    googlenews = GoogleNews()
    googlenews.search(topic)
    lstOfdf = []
    for i in range(4):
        result = googlenews.page_at(i)
        lstOfdf.append(pd.DataFrame(result))
    retLst = []
    quickDict = {}
    for df in lstOfdf:
        for i in range(len(df["link"])):
            url = df["link"][i][:df["link"][i].rfind("&ved=")] + "/"
            if url is None or url in retLst:
                continue
            article = Article(url)
            try:
                article.download()
                article.parse()
                quickDict[url] = article.text
                if url is not None:
                    retLst.append(url)
                    print(url)
            except:
                article
    print(time.time() - start_time)
    return retLst

async def get_urls_twitter(topic):
    global quickDict
    quickDict = {}
    client = Client('en-US')
    await client.login(
        auth_info_1=USERNAME,
        password=PASSWORD
    )
    retLst = []
    tweets = await client.search_tweet(topic, 'Latest')
    for tweet in tweets:
        quickDict[tweet.user.name] = tweet.text
        retLst.append(tweet.user.name)
    return retLst

def get_article_content(url):
    return quickDict[url]

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html", og=True)

@app.route('/', methods=['POST'])
def text():
    global topicCHOSEN
    global URLS
    topic = request.form['topic']
    source = request.form['source']
    topicCHOSEN = topic

    if source == "news":
        urls = get_urls_news(topic)
    elif source == "twitter":
        urls = asyncio.run(get_urls_twitter(topic))
    else:
        urls = []

    URLS = urls
    return render_template("index.html", topic=topic, go=True, urls=urls, source=source)

@app.route('/get_urls', methods=['POST'])
def fetch_urls():
    global quickDict
    global URLS
    topic = request.json['topic']
    source = request.json['source']
    
    if source == "news":
        urls = get_urls_news(topic)
    elif source == "twitter":
        urls = asyncio.run(get_urls_twitter(topic))
    else:
        urls = []

    URLS = urls
    return jsonify({"urls": urls})  # Send URLs as JSON

@app.route('/urlInfo', methods=['POST'])
def topic():
    topic = request.form['topicChose']
    texter = get_article_content(topic)
    return render_template("index.html", specificTopic=topic, topicText=texter)

@app.route('/graph', methods=['POST'])
def graph():
    global URLS   
    print(URLS)
    return render_template('graph.html', urls = URLS, specificTopic=topicCHOSEN)

@app.route('/sent', methods=['POST'])
def sentiment():
    global URLS, topicCHOSEN, quickDict
    urls = URLS
    topic = topicCHOSEN
    Model = tspmo.DeBERTa("checkpoint.config")
    random.shuffle(urls)
    for i in urls:
        artCont = quickDict[i]
        para = artCont.split("\n")
        score = 0
        addString = ""
        for j in para:
            retStr = j
            if addString:
                retStr = addString + " " + retStr

            if len(retStr) < len(topic):
                continue
            if len(retStr) < 500:
                addString += retStr

            if topic.lower() not in retStr.lower():
                continue

            guess = Model.predict(retStr, topic)["totalScore"]
            score += guess
        score /= len(para)
        print(i)
        print(Model.predict(artCont, topic))
        exit()
    return render_template("sent.html", urls = urls)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
    

