"""
Basic and simple prototype of GAPRS designed to
take user query and return list of relevant
items in an answer set through the OpenAlex API.
No front-end or back-end, completely CLI-based.
Viewing Results: http://jsonprettyprint.net/
"""
# Importing necesary and relevant modules
import requests
from pprint import pprint

# Defining variables
user_query = None # User's search query to be included in URL
url = None # Complete URL to be sent to OpenAlex
response = None # Response from request sent to OpenAlex
reponse_json = None # JSON representation of response sent by OpenAlex
# Presenting user with application welcome message
print("Welcome to GAPRS: Graph-based Academic Paper Recommender System!")
print("*This is a data-retrieval recommender prototype of GAPRS*")
# Prompt user to enter search query
user_query = input("Please enter the title of your thesis topic or your research question or keywords: ")
url = f"https://api.openalex.org/works?search={user_query}&select=id,display_name"
response = requests.get(url)
response_json = response.json()
pprint(response_json)