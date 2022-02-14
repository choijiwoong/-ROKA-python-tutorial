#[2. example of loading blog's page _crawling Async]
import requests
import asyncio
from bs4 import BeautifulSoup
import time

s=time.time()
results=[]

#--part A--
async def getpage(url):#실제 페이지를 불러오는 코루틴
    req=await loop.run_in_executor(None, requests.get, url)#awit은 코루틴을 실행하는 것이며, 실행한 코루틴은 loop.run_in_executor()로
    #코루틴만 실행되는 파이썬환경에서 일반 함수를 코루틴처럼 실행시켜주는 역활을 한다. 즉 그것만 다른거지 위는 requests.get(url)이다. 첫인자는 executor라고 한다.
    html=req.text
    soup=await loop.run_in_executor(None, BeautifulSoup, html, 'lxml')#마찬가지로 BeautifulSoup(html('lxml'))을 코루틴화하여 실행한다
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
    r=await asyncio.gather(*fts)#future(계획서)를 실행하고, 결과를 r에 담는다. C++로 따지면 asyncio.gather(*fts)로 futures를 gather하고 결과 promise를 담는다.(
    global results#글로벌 변수를 선언하고
    results=r#결과를 담는다.
#--part C--
loop=asyncio.get_event_loop()#루프 생성
loop.run_until_complete(main())#main종료시까지 루프 작동
loop.close
e=time.time()

print("{0:.2f} seconds are spent".format(e-s))
