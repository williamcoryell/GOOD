# import os
# os.system("pip install gnews")
from gnews import GNews
news = GNews()

print(news.get_news("pakistan"))