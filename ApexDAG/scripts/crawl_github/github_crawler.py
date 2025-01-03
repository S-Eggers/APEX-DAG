import requests
import time
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

class GitHubCrawler:
    def __init__(self):
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("app.log", mode='a')
            ]
        )
        token = os.getenv('GITHUB_TOKEN')
        
        self.logger = logging.getLogger(__name__)
        self.GITHUB_TOKEN = token
        self.HEADERS = {
            "Authorization": f"token {self.GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
    def process_response(self, response):
        '''
        To be used in the while loop during crawling for subclasses.
        Returns None if the rate limit is reached.
        '''
        rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", time.time()))
        
        if response.status_code == 403 and rate_limit_remaining == 0:
            reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rate_limit_reset))
            seconds_to_reset = rate_limit_reset - time.time() + 1
            if seconds_to_reset > 0:
                self.logger.warning(f"Rate limit reached. Reset at {reset_time}. Waiting... {seconds_to_reset:.3f} seconds.")  
            else:
                seconds_to_reset = 10
                self.logger.warning(f"Clock misalligmnent on rate reset time, waiting {seconds_to_reset} seconds.")
            time.sleep(seconds_to_reset) 
            return None
            

        if response.status_code != 200:
            self.logger.error(f"Error: {response.status_code}, {response.json()}")
            raise Exception(f"Error: {response.status_code}, {response.json()}")

        data = response.json()
        return data
    