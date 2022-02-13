#[2. example of loading blog's page _crawling Async]
import requests
import asyncio
from bs4 import BeautifulSoup
import time

s=time.time()
results=[]

#--part A--
async def getpage(url):
    req=await loop.run_in_executor(None, requests.get, url)
    html=req.text
    soup=await loop.run_in_executor(None, BeautifulSoup, html, 'lxml')
    return soup
#--part B--
async def main():
    urls = ["https://wp.me/p9x2W1-x",
            "https://wp.me/p9x2W1-w",
            "https://wp.me/p9x2W1-t",
            "https://wp.me/p9x2W1-q",
            "https://wp.me/p9x2W1-p",
            "https://wp.me/p9x2W1-j",
            "https://wp.me/p9x2W1-h"]#크롤링 리스트들

    fts=[asyncio.ensure_future(getpage(u)) for u in urls]#asyncio의 future오브젝트를 만들어 fts에 넣고, 내용은 urls의 page들이다.
    r=await asyncio.gather(*fts)
    global results
    results=r
#--part C--
loop=asyncio.get_event_loop()#루프 생성
loop.run_until_complete(main())#main종료시까지 루프 작동
loop.close
e=time.time()

print("{0:.2f} seconds are spent".format(e-s))
