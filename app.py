from flask import Flask, render_template, request, jsonify  # type: ignore
import os
from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
import time
import asyncio
from twikit import Client
import main as tspmo
import threading
global id2label
global label2id
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}
global Model
global sentDict
Model = tspmo.DeBERTa("checkpoint.config")


USERNAME = 'thingforsys3'
PASSWORD = 'hellomyunglee'


topicCHOSEN = ""
URLS = []
quickDict  = {}
titleDict = {}
sentDict = {}

def get_urls_news(topic, page):
    global quickDict
    global titleDict
    start_time = time.time()
    googlenews = GoogleNews()
    googlenews.search(topic)
    lstOfdf = []
    result = googlenews.page_at(page)
    lstOfdf.append(pd.DataFrame(result))
    retLst = []
    retLst2 = []
    for df in lstOfdf:
        for i in range(len(df["link"])):
            url = df["link"][i][:df["link"][i].rfind("&ved=")] + "/"
            if url is None or url in retLst or "nytime" in url or "msn" in url:
                continue
            article = Article(url)
            try:
                article.download()
                article.parse()
                quickDict[url] = article.text
                titleDict[url] = article.title
                if url is not None and article.is_media_news:
                    retLst.append(url)
                    retLst2.append([article.title, article.publish_date])
            except:
                article
    print(time.time() - start_time)
    return retLst, retLst2

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

@app.route('/articles', methods=['POST'])
def text():
    global topicCHOSEN
    global URLS
    topic = request.form['topic']
    source = request.form['source']
    topicCHOSEN = topic
    
    return render_template("articles.html", topic=topic, source=source, titles=titleDict, newsBool = source=='News/Long Text Model')

@app.route('/get_urls/<page>/<topic>/<source>', methods=['GET'])
def fetch_urls(page, topic, source):
    print("get_URLS")
    global quickDict
    global URLS
    if source == "news":
        urls, titles = get_urls_news(topic, int(page))
        for i in urls:
            if len(quickDict[i]) < 200:
                urls.remove(i)
            else:
                URLS.append(i)
        # URLS = urls
        return jsonify({"urls": urls, "titles": titles})  # Send URLs as JSON
    if source == "twitter":
        urls = asyncio.run(get_urls_twitter(topic))
        print(urls)
        for i in urls:
            if i not in URLS:
                URLS.append(i)
            else:
                urls.remove(i)
        return jsonify({"urls": urls})
@app.route("/startSent")
def startSent():
    threader = threading.Thread(target=startSentThread)
    threader.start()
    return jsonify({'pluh': "pluh"})

def startSentThread():
    global sentDict
    for urlM in URLS:
        textFor = quickDict[urlM]
        sentDict[urlM] = sentOneArt(textFor, topicCHOSEN)

@app.route('/urlInfo', methods=['POST'])
def topic():
    topic = request.form['topicChose']
    texter = get_article_content(topic)
    sent= sentOneArt(texter, topic)
    return render_template("urlinfo.html", specificTopic=topic, topicText=texter, sent=sent)

@app.route('/graph', methods=['POST'])
def graph():
    global URLS   
    # print(URLS)
    return render_template('graph.html', urls = URLS, specificTopic=topicCHOSEN)

def sentOneArt(text, topic):
    artCont = text
    # print(artCont)
    print(text[:50] + "...")
    para = artCont.split("\n")
    score = 0
    addString = ""
    scoringPara = 0
    
    for j in para:
        # print(addString)
        # print("-------")
        retStr = j
        # print(f"addString: {addString}")
        # print(f"retStr: {retStr}")

        if len(addString + retStr) < 150:
            addString += " " + retStr
            continue

        if addString:
            retStr = addString + " " + retStr

        # print(len(retStr))

        addString = ""
        guess = Model.predict(retStr, topic)["totalScore"]
        # print(len(retStr))
        # print(retStr)
        # print(guess)
        if abs(guess) > 0.13:
            score += guess
            scoringPara += 1
    if addString:
        retStr = addString + " " + retStr
        guess = Model.predict(retStr, topic)["totalScore"]
        # print(len(retStr))
        # print(retStr)
        # print(guess)
        if abs(guess) > 0.13:
            score += guess
            scoringPara += 1
    if not scoringPara:
        return 0
    score /= scoringPara
    return score

@app.route("/sentOnWeb/<urlM>", methods=['GET'])
def sentOnWeb(urlM):
    print(urlM)
    print(URLS[int(urlM)])
    if URLS[int(urlM)] in sentDict:
        print(sentDict[URLS[int(urlM)]])
        return jsonify({"pluh": sentDict[URLS[int(urlM)]]})
    else:
        return jsonify({'pluh': "not yet"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)