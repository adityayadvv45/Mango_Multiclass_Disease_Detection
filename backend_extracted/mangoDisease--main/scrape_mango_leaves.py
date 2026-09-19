import os
from bing_image_downloader import downloader
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, BaiduImageCrawler

# The user wants to download at least 2000 images in the current folder.
# They requested mango leaves with multiple diseases like Anthracnose, Bacterial Canker, etc.

output_dir = "/Users/abhinavsahu/Downloads/projects/mangoDisease"
os.makedirs(output_dir, exist_ok=True)

queries = [
    "mango leaf Anthracnose disease",
    "mango leaf Bacterial Canker",
    "mango leaf Powdery Mildew",
    "mango leaf Cutting Weevil",
    "mango leaf Die Back",
    "mango leaf Sooty Mould",
    "mango leaf Gall Midge",
    "mango leaf multiple diseases",
    "diseased mango leaves",
    "mango leaf spots",
    "infected mango leaves"
]

images_per_query = 200 # 11 queries * 200 images = 2200 images

def scrape_with_icrawler(query, max_num):
    print(f"Scraping with GoogleImageCrawler for: {query}")
    query_dir = os.path.join(output_dir, query.replace(" ", "_"))
    os.makedirs(query_dir, exist_ok=True)
    google_crawler = GoogleImageCrawler(storage={'root_dir': query_dir})
    google_crawler.crawl(keyword=query, max_num=max_num)

    print(f"Scraping with BingImageCrawler for: {query}")
    bing_crawler = BingImageCrawler(storage={'root_dir': query_dir})
    bing_crawler.crawl(keyword=query, max_num=max_num)

for query in queries:
    scrape_with_icrawler(query, images_per_query)

print("Scraping completed.")
