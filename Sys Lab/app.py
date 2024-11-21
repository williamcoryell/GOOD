from flask import Flask, render_template, request # type: ignore
import os
from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
from gnews import GNews
import time
import numpy as np
import matplotlib.pyplot as plt

updateGraph = 0
topicCHOSEN = ""

def get_urls(topic):
    global quickDict
    start_time = time.time()
    googlenews=GoogleNews()
    googlenews.search(topic)
    lstOfdf = []
    for i in range(2):
        googlenews.get_page(i)
        result=googlenews.result()
        lstOfdf.append(pd.DataFrame(result))
    retLst = []
    quickDict = {}
    for df in lstOfdf:
        for i in range(len(df["link"])):
            url = df["link"][i][:df["link"][i].rfind("&ved=")] + "/"
            article = Article(url)
            try:
                article.download()
                article.parse()
                quickDict[url] = article.text
                if url is not None:
                    retLst.append(url)
            except:
                print("not working url")
    print(time.time()-start_time)
    return retLst

def get_article_content(url):
    return quickDict[url]

#def get_plot(sentData):
def get_plot(): 
    x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
    y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
    plt.scatter(x, y, color = 'hotpink')
     
    plt.xlabel('X label') 
    plt.ylabel('Y label') 
    return plt 

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html", og=True)

@app.route('/', methods=['POST'])
def text():
    global topicCHOSEN
    topic = request.form['topic']
    topicCHOSEN = topic
    urls = get_urls(topic)
    return render_template("index.html", topic=topic, go=True, urls=urls)

@app.route('/urlInfo', methods=['POST'])
def topic():
    topic = request.form['topicChose']
    texter = get_article_content(topic)
    return render_template("index.html", specificTopic=topic, topicText = texter)

@app.route('/graph', methods=['POST'])
def graph():
    global updateGraph, topicCHOSEN
    if updateGraph < 1:
        plot = get_plot() 
        # Save the figure in the static directory  
        plot.savefig(os.path.join('static', 'images', 'plot.png')) 
        updateGraph += 1
    topic = topicCHOSEN
    topicCHOSEN = ""
    for word in topic.split():
        topicCHOSEN = topicCHOSEN + word.title() + " "
    return render_template('graph.html', specificTopic=topicCHOSEN) 

@app.route('/sent', methods=['POST'])
def sentiment():
    return render_template("sent.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=open)