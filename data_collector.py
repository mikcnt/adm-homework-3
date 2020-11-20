from bs4 import BeautifulSoup
import requests
import asyncio
import aiohttp
import aiofiles
import nest_asyncio
import os
nest_asyncio.apply()

loop = asyncio.get_event_loop()

def list_path(page_number):
    return "./best_books_pages/{}.html".format(page_number)

async def fetch_best(session: aiohttp.ClientSession, page_number):
    url = 'https://www.goodreads.com/list/show/1.Best_Books_Ever?page={}'.format(page_number)
    path = list_path(page_number)
    if (os.path.exists(path)):
        return
    response = await session.get(url)
    data = await response.read()
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)

async def fetch_all_best():
    async with aiohttp.ClientSession() as session:
        step = 50
        for index in range(1, 300, step):
            page_numbers = range(index, index + step)
            
            futures = [fetch_best(session, page_number) for page_number in page_numbers]
            await asyncio.gather(*futures)


def book_links(page_number):
    html = BeautifulSoup(open(list_path(page_number)), features="lxml")
    links = html.find_all("a", {'class':'bookTitle'})
    return ['https://www.goodreads.com{}'.format(link['href']) for link in links]

def extract_links():
    with open('book_links.txt', 'w') as f:
        for page_number in range(1, 301):
            links = book_links(page_number)
            f.write('\n'.join(links) + '\n')
        
def books_folder(page_number):
    return 'books/page_{}'.format(page_number)

def book_path(page_number, url):
    name = url.split('/')[-1].rstrip()
    return '{}/{}.html'.format(books_folder(page_number), name)

async def fetch_html(session: aiohttp.ClientSession, page_number, url):
    path = book_path(page_number, url)
    if (os.path.exists(path)):
        return
    print('ao')
    response = await session.get(url)
    data = await response.read()
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)
        
async def fetch_all_html():
    with open('book_links.txt', 'r') as txt:
        async with aiohttp.ClientSession() as session:
            for page_number in range(1, 301):
                print('Page number', page_number)
                os.makedirs(books_folder(page_number), exist_ok=True)
                urls = [txt.readline() for _ in range (100)]
                futures = [fetch_html(session, page_number, url) for url in urls]
                await asyncio.gather(*futures)


def download_books(bests=False, links=False, books=False):
    if bests:
        loop.run_until_complete(fetch_all_best())
    if links:
        extract_links()
    if books:
        loop.run_until_complete(fetch_all_html())
        

download_books()