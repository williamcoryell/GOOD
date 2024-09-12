import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_article_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        article_content = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in article_content])
        return article_text
    else:
        return f"Failed to retrieve the article. Status code: {response.status_code}"

# Example usage:
df = pd.read_csv('Sentiment_dataset.csv')
for ind in df.index:
    try:
        url = df['url'][ind]
        article_text = get_article_content(url)
        data = article_text.split()
        print(data) 
    except:
        print("sorry")