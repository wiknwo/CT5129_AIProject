"""
Basic and simple prototype of GAPRS designed to
take user query and return list of relevant
items in an answer set through the OpenAlex API.
No front-end or back-end, completely CLI-based.
Viewing Results: http://jsonprettyprint.net/
"""
# Importing necesary and relevant modules
import requests
import networkx as nx
import matplotlib.pyplot as plt
import time
import datetime
import math
import itertools
import os
from pprint import pprint

# Defining functions
def distribute_centre_nodes_evenly(n_centre_nodes):
    """Evenly distribute centre nodes (egos) around a circle"""
    angle_increment = 2 * math.pi / n_centre_nodes
    angle = 0 # Angle from centre of circle
    x, y = None, None # Centre node's co-ordinates
    centre_nodes_positions = [] # List containing co-ordinates of each centre node
    for centre_node_index in range(n_centre_nodes):
        x = math.cos(angle)
        y = math.sin(angle)
        centre_node_position = (x, y)
        centre_nodes_positions.append(centre_node_position)
        angle += angle_increment
    return centre_nodes_positions

def save_hybrid_citation_network(hcn, datetimestamp, dirpath, iteration):
    """Save the hybrid citation network in a TXT file in edgelist format"""
    filename = f"{dirpath}/hcn_{datetimestamp}_L{iteration}_edgelist.txt"
    nx.write_edgelist(hcn, filename, delimiter=", ", data=["weight"])

def plot_hybrid_citation_network(hcn, ego_labels, iteration, datetimestamp, time_taken, dirpath, user_query):
    """Plots the hybrid citation network and saves it as an image"""
    # Set network display settings
    node_positions = nx.spring_layout(hcn, weight=None)
    edge_weights_labels = {e: f"{hcn.edges[e]['weight']:.4f}" for e in hcn.edges}
    node_colors = ["red" if node_label in ego_labels else "blue" for node_label in hcn.nodes.keys()]
    # Draw network and save as image
    plt.figure(figsize=(10, 10), dpi=100)
    plt.suptitle(f"Hybrid Citation Network (L{iteration}) | Nodes: Academic Papers | Edge Weights: avg(NCCC, NBCC)", verticalalignment="top", fontsize="large", fontweight="bold")
    plt.title(f"User Query: {user_query} | Time Elapsed: {time_taken:.2f} seconds", fontdict={"fontsize": 10}, y=-0.01)
    plt.axis("off")
    nx.draw(hcn, node_color=node_colors, pos=node_positions, font_size=5, with_labels=True)
    nx.draw_networkx_edge_labels(hcn, pos=node_positions, edge_labels=edge_weights_labels, font_size=5)
    plt.savefig(f"{dirpath}/hcn_{datetimestamp}_L{iteration}.png")
    plt.clf()

def create_alter_object(alter_oa_id_url, api_info):
    """Create an alter object from the api information fetched from OpenAlex"""
    url = f"https://api.openalex.org/works/{alter_oa_id_url}?select={','.join(api_info)}"
    response = requests.get(url)
    response_json = response.json()
    alter_object = response_json
    url = f"https://api.openalex.org/works/{alter_oa_id_url}?select=authorships"
    response = requests.get(url)
    response_json = response.json()
    if response_json["authorships"]:
        alter_object["first_author_name"] = response_json["authorships"][0]["author"]["display_name"]
        alter_object["first_author_id"] = response_json["authorships"][0]["author"]["id"]
        alter_object["network_label"] = alter_object["first_author_name"] + " " + str(alter_object["publication_year"])
    else:
        print(f"Anonymous ID: {alter_object['id']}")
        alter_object["network_label"] = "Anonymous " + str(alter_object["publication_year"])
    return alter_object

def calculate_edgeweight_between_ego_and_alter(ego_references, alter_references):
    """Calculates the edge weight between ego and one of its alters in hybrid citation network"""
    # Compute normalised bibliographic coupling between alters u and v
    bibliographic_coupling_jaccard_coefficient = -1
    bibliographic_coupling_intersection = set.intersection(ego_references, alter_references)
    bibliographic_coupling_union = set.union(ego_references, alter_references)
    if not bibliographic_coupling_union:
        bibliographic_coupling_jaccard_coefficient = 0
    else:
        bibliographic_coupling_jaccard_coefficient = len(bibliographic_coupling_intersection) / len(bibliographic_coupling_union)
    normalized_bcc = bibliographic_coupling_jaccard_coefficient
    # Compute normalised co-citation count between alters u and v
    normalized_ccc = 0 # Normalied co-citation count between egos and alters is always zero so no need to calculate it
    # Calculate edge weight between ego and alter
    edge_weight = (normalized_ccc + normalized_bcc) / 2
    return edge_weight

def calculate_egdeweight_between_alters(u_object, v_object, alters_less_uv_objects):
    """Calculates edge weight between alter objects u and v"""
    # Create sets of referenced works for alters u and v
    u_object_references = set(u_object["referenced_works"])
    v_object_references = set(v_object["referenced_works"])
    # Compute normalised bibliographic coupling between alters u and v
    bibliographic_coupling_jaccard_coefficient = -1
    bibliographic_coupling_intersection = set.intersection(u_object_references, v_object_references)
    bibliographic_coupling_union = set.union(u_object_references, v_object_references)
    if not bibliographic_coupling_union:
        bibliographic_coupling_jaccard_coefficient = 0
    else:
        bibliographic_coupling_jaccard_coefficient = len(bibliographic_coupling_intersection) / len(bibliographic_coupling_union)
    normalized_bcc = bibliographic_coupling_jaccard_coefficient
    # Compute normalised co-citation count between alters u and v
    citation_count_u, citation_count_v, cocitation_count_uv = 0, 0, 0
    for w_object in alters_less_uv_objects:
        w_object_references = set(w_object["referenced_works"])
        if u_object["id"] in w_object_references and v_object["id"] in w_object_references:
            cocitation_count_uv += 1
        elif u_object["id"] in w_object_references:
            citation_count_u += 1
        elif v_object["id"] in w_object_references:
            citation_count_v += 1
    if citation_count_u == 0 and citation_count_v == 0:
        normalized_ccc = 0
    else:
        normalized_ccc = cocitation_count_uv / (citation_count_u + citation_count_v)
    # Calculate edge weight between alters u and v
    edge_weight = (normalized_ccc + normalized_bcc) / 2
    return edge_weight

def calculate_alter_with_highest_centrality_measure(egocentric_subnetworks, alters_objects):
    """Calculates alter node with highest centrality measure in egocentric subnetwork"""
    new_ego_objects = []
    for ego_index, egocentric_subnetwork in enumerate(egocentric_subnetworks):
        ego_nets_dd = sorted(egocentric_subnetwork.degree(weight="weight"), key=lambda t: t[1], reverse=True)
        new_ego_label = ego_nets_dd[0][0]
        new_ego_object = None
        for alter_object in alters_objects[ego_index]:
            if alter_object["network_label"] == new_ego_label:
                new_ego_object = alter_object
                new_ego_objects.append(new_ego_object)
                break
    return new_ego_objects

def collate_alters_objects(alters_objects, ego_objects):
    """Collates each alter paper of ego paper into a list of objects"""
    for ego_object_index, ego_object in enumerate(ego_objects):
        # Iterating through every reference in reference section of ego paper
        for alter_oa_id_url in ego_object['referenced_works']:
            # Creating alter object for each reference in ego paper's references section
            alters_objects[ego_object_index].append(create_alter_object(alter_oa_id_url, alters_info))

def assemble_hybrid_citation_network(hcn, ego_objects, alters_objects):
    """Connects hybrid citation egocentric network by adding edges between egos and their respective alters"""
    # Calculate edge weights between egos and their respective alters
    for ego_object_index, ego_object in enumerate(ego_objects):
        for alter_object in alters_objects[ego_object_index]:
            # Create sets of referenced works between ego and alter
            ego_references = set(ego_object["referenced_works"])
            alter_references = set(alter_object["referenced_works"])
            # Calculate edge weight between ego and alter
            edge_weight = calculate_edgeweight_between_ego_and_alter(ego_references, alter_references)
            # Add edges between egos and their respective alters
            hcn.add_edge(ego_object["network_label"], alter_object["network_label"], weight=edge_weight)

def create_hybrid_citation_subnetworks(hcn, hcsn, ego_objects, alters_objects):
    """Creates hybrid citation 1.5 degree egocentric subnetworks with egos excluded"""
    for ego_object_index, ego_object in enumerate(ego_objects):
        hcsn.append(nx.ego_graph(hcn, ego_object["network_label"], center=False))
        # for alter_object in alters_objects[ego_object_index]:
        alter_object_combinations = itertools.combinations(alters_objects[ego_object_index], 2)
        for u_object, v_object in alter_object_combinations:
            # Compute normalised co-citation count between alters u and v
            alters_less_uv_objects = alters_objects[ego_object_index].copy()
            alters_less_uv_objects.remove(u_object)
            alters_less_uv_objects.remove(v_object)
            # Calculate edge weight between alters u and v
            edge_weight = calculate_egdeweight_between_alters(u_object, v_object, alters_less_uv_objects)
            # Add edge between alters u and v in 1.5 degree egocentric subnetwork
            hcsn[ego_object_index].add_edge(u_object["network_label"], v_object["network_label"], weight=edge_weight)

def display_time_elapsed(time_taken, iteration):
    """Print time elapsed for current iteration based on given start and end times"""
    print(f"== LEVEL {iteration} COMPLETE ==")
    print(f"Time Elapsed L{iteration}: {time_taken:.2f} seconds")

# Defining user variables
user_choice = None # User's choice from list of options
user_query = None # User's search query to be included in URL

# Defining admin variables
url = None # Complete URL to be sent to OpenAlex
item_info = ["id", "display_name", "publication_year"] # Information to display for each recommendation
response = None # Response from request sent to OpenAlex
reponse_json = None # JSON representation of response sent by OpenAlex
recommendations = [] # List of recommendations
selected_recommendations = [] # List of initial recommendations selected by user
selected_recommendations_ranks = set() # Set of ranks of selected recommendations
alters = None # List to hold JSON objects representing each alter in egocentric network
alters_info = ["id", "display_name", "publication_year", "referenced_works"] # Information to display for each alter
ego_labels = set() # Set to hold network labels for designated egos of egocentric networks
egos = [] # List to hold JSON objects representing each ego in egocentric networks
hybrid_citation_network = nx.Graph() # Hybrid citation network where edge weights equal avg(NCCC, NBCC)
hybrid_citation_subnetworks = None # Hybrid citation subnetworks comprised of 1.5 degree egocentric subnets with egos exlcuded
hybrid_citation_network_egos_only = None # Hybrid citation network where edge weights equal avg(NCCC, NBCC) and only egos are displayed
start_time = None # Monitor time taken for program to run
end_time = None # Monitor time taken for program to run
time_elapsed = None # end_time - start_time
todays_datetime_string = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') # Differentiate between files
foldername = f"hcn_{todays_datetime_string}"
folderpath = os.path.join(os.getcwd(), foldername)
time_step = 0 # Monitors the iteration in the evolution of the hybrid citation network we are on
# education_levels = {1: "Undergraduate", 2: "Masters"}

# Presenting user with application welcome message and informing user about how to use GAPRS
print("== WELCOME MESSAGE ==")
print("Welcome to GAPRS: Graph-based Academic Paper Recommender System!")
print("== INSTRUCTIONS ==")
print("> Your search for academic papers is like a funnel; wide at the beginning and increasingly narrow towards the end.")
print("> GAPRS can take your thesis topic, keywords, research question or research problem as a starting point and give you recommendations based on them.")
print("> As you come to understand your thesis topic, you may narrow your search by looking for specific papers, authors, etc. and GAPRS can retrieve them.")

# Prompt user to enter search query
print("== INITIAL USER INPUT ==")
user_query = input("> Please enter what you would like to search for: ")
url = f"https://api.openalex.org/works?search={user_query}&select={','.join(item_info)}&per-page=5"
response = requests.get(url)
response_json = response.json()

# Collect recommendations for printing
recommendations = response_json["results"]
for index, recommendation in enumerate(recommendations):
    # Add rank to each recommendation item
    recommendation["rank"] = index + 1

# Display list of initial recommendations to user
print("== INITIAL RECOMMENDATIONS ==")
pprint(recommendations)

# Ask user to select recommendations from list
print("== RECOMMENDATION SELECTION ==")
print("> Please select the most relevant recommendations from the list above.")
print("> Please type 'D' when you are done.")
while True:
    user_choice = input("Please type rank of recommendation to select it: ")
    if user_choice.upper() == 'D':
        print("> Thank you for completing your selection.")
        break
    elif  1 <= int(user_choice) <= len(recommendations) and int(user_choice) not in selected_recommendations_ranks:
        selected_recommendations_ranks.add(int(user_choice))
    else:
        print(f"> Please select a valid rank [1-{len(recommendations)}].")

# Reset user choice
user_choice = None

# Start counting time after all user input is collected before L0 calculation
start_time = time.time()

# Sort selected recommendations in descending order of rank
for selected_recommendation_rank in selected_recommendations_ranks:
    selected_recommendations.append(recommendations[selected_recommendation_rank - 1])
selected_recommendations = sorted(selected_recommendations, key=lambda rec: rec["rank"])

# Retrieve references for selected recommendations
for selected_recommendation in selected_recommendations:
    oa_id = selected_recommendation['id']
    print("Ego OpenAlex ID:", oa_id)
    url = f"https://api.openalex.org/works/{oa_id}?select=referenced_works,authorships"
    response = requests.get(url)
    response_json = response.json()
    selected_recommendation["referenced_works"] = response_json["referenced_works"]
    # if response_json["authorships"]:
    if response_json["authorships"]:
        selected_recommendation["first_author_name"] = response_json["authorships"][0]["author"]["display_name"]
        selected_recommendation["first_author_id"] = response_json["authorships"][0]["author"]["id"]
    selected_recommendation["network_label"] = selected_recommendation["first_author_name"] + " " + str(selected_recommendation["publication_year"])
    ego_labels.add(selected_recommendation["network_label"])
    del selected_recommendation["rank"]
    egos.append(selected_recommendation)

# Display selected recommendations again with new key added
print("== SELECTED RECOMMENDATIONS WITH REFERENCES ==")
pprint(selected_recommendations)
selected_recommendations.clear()

# STEP 1: Assembling hybrid weighted egocentric citation network
# Fetch alter information for alters of new egos
alters = [[] for i in range(len(egos))]
collate_alters_objects(alters, egos)
# Set the egos currently being considered to be copy of egos
ego_objects_snapshot = egos.copy() # Holds the egos of the egocentric networks currently being considered, i.e., L_I
# Connect ego and alters in hcn
assemble_hybrid_citation_network(hybrid_citation_network, egos, alters)
# Note time again now that L_I is calculated
end_time = time.time()
time_elapsed = end_time - start_time
display_time_elapsed(time_elapsed, time_step)
# Create directory where image files representing network plots will be stored
os.mkdir(folderpath)
# Plot and save the network for the current time step
plot_hybrid_citation_network(hybrid_citation_network, ego_labels, time_step, todays_datetime_string, time_elapsed, folderpath, user_query)
# Move to next iteration 
time_step += 1
# =====================================================================================

# Keep polling user for input until they want program to terminate
while True:
    # STEP 2: Creating hybrid weighted 1.5 degree egocentric subnetworks with egos excluded
    # Reset time counting before L_I+1 calculation
    start_time = time.time()
    # Create 1.5 degree egocentric network for each ego
    ego_subnets_snapshot = [] # Holds the egocentric subnetworks currently being considered, i.e., L_I
    create_hybrid_citation_subnetworks(hybrid_citation_network, ego_subnets_snapshot, ego_objects_snapshot, alters)
    # Combine each 1.5 degree egocentric subnetwork with ego excluded into one network
    hybrid_citation_subnetworks = nx.compose_all(ego_subnets_snapshot)
    # Note time again now that L_I+1 is calculated
    end_time = time.time()
    time_elapsed = end_time - start_time
    display_time_elapsed(time_elapsed, time_step)
    # Plot and save the network for the current time step
    plot_hybrid_citation_network(hybrid_citation_subnetworks, ego_labels, time_step, todays_datetime_string, time_elapsed, folderpath, user_query)
    # =====================================================================================
    # STEP 3: Expand hybrid weighted 1.5 degree egocentric subnetworks with egos excluded
    # Reset time counting before L_I+2 calculation
    time_step += 1
    start_time = time.time()
    # Calculating alter node with highest centrality measure in each egocentric subnetork
    ego_objects_snapshot = calculate_alter_with_highest_centrality_measure(ego_subnets_snapshot, alters)
    # Add new ego objects to list of egos
    egos.extend(ego_objects_snapshot)
    # Add new ego labels to set of ego labels
    ego_labels.update([ego_object["network_label"] for ego_object in ego_objects_snapshot])
    # Fetch alter information for alters of new egos
    alters = [[] for i in range(len(ego_objects_snapshot))]
    collate_alters_objects(alters, ego_objects_snapshot)
    # Connect ego and alters in hcn
    assemble_hybrid_citation_network(hybrid_citation_network, ego_objects_snapshot, alters)
    # Note time again now that L_I+2 is calculated
    end_time = time.time()
    time_elapsed = end_time - start_time
    display_time_elapsed(time_elapsed, time_step)
    # Plot and save the network for the current time step
    plot_hybrid_citation_network(hybrid_citation_network, ego_labels, time_step, todays_datetime_string, time_elapsed, folderpath, user_query)
    # =====================================================================================
    # Ask user if they would like to see another iteration
    user_choice = input("Would like to see another iteration (Y/N): ")
    user_choice = user_choice.upper()
    if user_choice == "N":
        break
    else:
        time_step += 1
# Create one last image of egos only and connections between them
start_time = time.time()
time_step += 1
hybrid_citation_network_egos_only = hybrid_citation_network.subgraph(ego_labels)
end_time = time.time()
time_elapsed = end_time - start_time
save_hybrid_citation_network(hybrid_citation_network_egos_only, todays_datetime_string, folderpath, time_step)
plot_hybrid_citation_network(hybrid_citation_network_egos_only, ego_labels, time_step, todays_datetime_string, time_elapsed, folderpath, user_query)

# Print recommendations to console
print("== FINAL RECOMMENDATIONS (RED NODES) ==")
pprint(egos)