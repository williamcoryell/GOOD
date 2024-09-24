from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from bs4 import BeautifulSoup
import requests

model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def get_urls(topic):
    # for url in soup.find_all('a',  attrs={'href': re.compile("^https://")}): 
    # print(url.get('href'))   

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    url = 'https://www.google.com/search?q=' + topic + '&ie=utf-8&oe=utf-8&num=10'
    html = requests.get(url, headers=headers)
    
    urls = []

    if html.status_code == 200:
        soup = BeautifulSoup(html.text, "html.parser") 
        allData = soup.find_all("div",{"class":"g"})
        
        for data in allData:
            url = data.find('a').get('href')
            if url is not None:
                if url.find('https') != -1 and url.find('http') == 0 and url.find('aclk') == -1:
                    urls.append(url)
        
    return urls

def get_article_content(url):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        article_content = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in article_content])
        return article_text
    else:
        return f"Failed to retrieve the article. Status code: {response.status_code}"


topic = 'birds' #input("Topic: ")

urls = get_urls(topic)

for url in urls:
    article_text = get_article_content(url)
    #test but need to split up article_text into chunks
    print(classifier(article_text[:100], text_pair=topic))

   