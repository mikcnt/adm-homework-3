from bs4 import BeautifulSoup
import csv
import os

def html_parser(book_path, url):
    ttr = []
    if not os.path.exists(book_path):
        return
    html = BeautifulSoup(open(book_path), features='lxml')
    
    bookTitle = html.find('h1', id = 'bookTitle')
    bookTitle = bookTitle.contents[0].strip() if bookTitle else ''
    
    ttr.append(bookTitle)
    
    bookSeries = html.find('h2', id = 'bookSeries')
    if bookSeries:
        bookSeries= bookSeries.find('a')
    bookSeries = bookSeries.contents[0].strip()[1:-1] if bookSeries else ''
    
    ttr.append(bookSeries)
    
    bookAuthors = html.find('span', itemprop = 'name')
    bookAuthors = bookAuthors.contents[0].strip() if bookAuthors else ''
    
    ttr.append(bookAuthors)
    
    ratingValue = html.find('span', itemprop = 'ratingValue')
    ratingValue = ratingValue.contents[0].strip() if ratingValue else ''
    
    ttr.append(ratingValue)
    
    ratingCount = html.find('meta', itemprop='ratingCount')
    ratingCount = ratingCount['content'] if ratingCount else ''

    ttr.append(ratingCount)

    reviewCount = html.find('meta', itemprop='reviewCount')
    reviewCount = reviewCount['content'] if reviewCount else ''
    
    ttr.append(reviewCount)
    
    plot = html.find('div', {'id': 'description'})
    
    if plot:
        plot = plot.get_text().split('\n')
        if len(plot) == 5:
            plot = plot[2]
        else:
            plot = plot[1]
    else:
        plot = ''
    
    ttr.append(plot)
    
    NumberofPages = html.find('span', itemprop='numberOfPages')
    NumberofPages = NumberofPages.contents[0].split()[0] if NumberofPages else ''

    ttr.append(NumberofPages)

    PublishingDate = html.find_all('div', {'class':'row'})
    if PublishingDate and len(PublishingDate) > 1:
        PublishingDate = ' '.join(PublishingDate[1].contents[0].split()[1:4]).replace('by', '').rstrip()
    else:
        PublishingDate = ''
        
    ttr.append(PublishingDate)
    
    Characters = html.find_all('a', href=True)
    
    if Characters:
        Characters = ' '.join([' '.join(el.contents) for el in Characters if el['href'].startswith('/characters/')])
    
    ttr.append(Characters)
    
    Setting = html.find_all('a', href=True)
    
    if Setting:
        Setting = ' '.join([' '.join(el.contents) for el in Setting if el['href'].startswith('/places/')])
        
    ttr.append(Setting)
    
    ttr.append(url)
    
    return ttr



header = [
    'bookTitle', 'bookSeries', 'bookAuthors',
    'ratingValue', 'ratingCount', 'reviewCount',
    'plot', 'numberOfPages', 'PublishingDate',
    'Characters', 'Setting', 'url'
    ]

with open('parsed_books.tsv', 'w') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(header)
    with open('book_links.txt', 'r') as txt:
        for i, url in enumerate(txt):
            book_n = i + 1
            print('Book number:', book_n)
            page_n = (book_n - 1) // 100 + 1
            path = 'books/page_{}/{}.html'.format(page_n, book_n)
            row = html_parser(path, url)
            if row:
                tsv_writer.writerow(row)