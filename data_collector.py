from bs4 import BeautifulSoup
import requests
import asyncio
import aiohttp
import aiofiles
import nest_asyncio
import os
import shutil
from pathlib import Path

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

def book_name(page_number, book_n):
    return '{}/{}.html'.format(books_folder(page_number), (page_number - 1) * 100 + book_n + 1)

async def fetch_html(session: aiohttp.ClientSession, page_number, book_n, url):
    path = book_name(page_number, book_n)
    if (os.path.exists(path)):
        return
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
                urls = [txt.readline() for _ in range(100)]
                futures = []
                for book_n, url in enumerate(urls):
                    futures.append(fetch_html(session, page_number, book_n, url))
                await asyncio.gather(*futures)
                
            
def rm_fails():
    rootdir = './books'
    for subdir, _, files in os.walk(rootdir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if Path(file_path).stat().st_size < 5000:
                os.remove(file_path)

def create_dirs():
    main_path = './books'
    best_book_path = './best_books_pages/'
    Path(main_path).mkdir(parents=True, exist_ok=True)
    Path(best_book_path).mkdir(parents=True, exist_ok=True)

    for i in range(1, 301):
        page_path = main_path + '/page_{}'.format(i)
        Path(page_path).mkdir(parents=True, exist_ok=True)


def download_books(dirs=False, bests=False, links=False, books=False, fails=False):
    if dirs:
        create_dirs()
    if bests:
        loop.run_until_complete(fetch_all_best())
    if links:
        extract_links()
    if books:
        loop.run_until_complete(fetch_all_html())
    if fails:
        rm_fails()