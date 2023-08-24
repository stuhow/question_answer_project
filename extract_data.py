# script to extract

import os
import requests
import re

url = "https://www.gov.uk/api/content/foreign-travel-advice"
base_url = "https://www.gov.uk/api/content"


# functions to clean text from api
def remove_text_between_angle_brackets(input_string):
    pattern = r'<.*?>'  # This pattern matches everything between angle brackets
    result = re.sub(pattern, '', input_string)
    return result

def remove_newlines(text):
    text = text.replace('\n', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('   ', ' ')
    return text


# A function to extract the list of individual country api urls
def url_list(url):
    response = requests.get(url).json()
    list_of_urls = [i["base_path"] for i in response["links"]["children"]]
    return list_of_urls


# a function to extract the text from each country on the fco government advice page and save them into individual text files
def api_crawl():
    # get a list of the api urls
    api_url_list = url_list(url)

    # Create a directory to store the text files
    if not os.path.exists("raw_data/"):
            os.mkdir("raw_data/")

    if not os.path.exists("raw_data/"):
            os.mkdir("raw_data/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

        # iterate through url list to get all the text for every country
    for country in api_url_list:

        print(country[23:])


        # Save text from the url to a country.txt file
        with open('raw_data/'+ country[23:] + ".txt", "w", encoding="UTF-8") as f:

            # create api url
            country_url = base_url + country

            # make request to api
            response = requests.get(country_url).json()

            # extract and clean text for the country
            text = ''
            for part in response["details"]["parts"]:
                part = remove_text_between_angle_brackets(part['body'])
                part = remove_newlines(part)
                text+=part

            # write the text to the file in the text directory
            f.write(text)

if __name__ == "__main__":
    api_crawl()
