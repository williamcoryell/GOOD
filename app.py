from flask import Flask, render_template, request # type: ignore
from bs4 import BeautifulSoup
import requests
import os
import base64
from gnews import GNews
from newspaper import Article


def get_urls(topic):
    # for url in soup.find_all('a',  attrs={'href': re.compile("^https://")}): 
    # print(url.get('href'))   
    news = GNews()

    rawUrls = news.get_news(topic)

    urls = []
    
    for rawUrl in rawUrls:
        b64s = rawUrl['url'][rawUrl['url'].find("articles/")+9:rawUrl['url'].find("?")]
        print(b64s)
        urls.append(rawUrl['url'])
        
    return urls

def get_article_content(url):
    # headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    # response = requests.get(url, headers=headers)
    # return response.text
    # if response.status_code == 200:
    #     soup = BeautifulSoup(response.content, 'html.parser')
    #     article_content = soup.find_all('p')
    #     article_text = ' '.join([para.get_text() for para in article_content])
    #     return article_text
    # else:
    #     return f"Failed to retrieve the article. Status code: {response.status_code}"
    # article = Article(url, language="en")
    # article = Article(url)
    # article.download()
    # article.parse()
    # print(article.text)
    # return article.text
    article = GNews().get_full_article(url)
    return article
    

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
