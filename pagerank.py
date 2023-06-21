import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Create a dictionary to store the probability distribution
    probability_distribution = {}

    # If the page has no outgoing links, then transition_model should return a probability distribution that chooses randomly among all pages with equal probability.
    if len(corpus[page]) == 0:
        for page_ in corpus:
            probability_distribution[page_] = 1 / len(corpus)
        return probability_distribution
    
    # With probability damping_factor, the random surfer should randomly choose one of the links from page with equal probability.
    for corpus_page in corpus:
        probability_distribution[corpus_page] = (1 - damping_factor) / len(corpus)
    for link in corpus[page]:
        probability_distribution[link] += damping_factor / len(corpus[page])
    for corpus_page in corpus:
        probability_distribution[corpus_page] = round(probability_distribution[corpus_page], 4)
    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Create a dictionary to store the number of times each page is sampled
    page_visits = {}
    for page in corpus:
        page_visits[page] = 0
    
    # Choose a page at random to start
    current_page = random.choice(list(corpus.keys()))
    page_visits[current_page] = 1

    # Generate n samples
    for i in range(n - 1):
        current_page_transition_model = transition_model(corpus, current_page, damping_factor)
        new_page = random.choices(list(current_page_transition_model.keys()), weights=list(current_page_transition_model.values()))[0]
        current_page = new_page
        page_visits[current_page] += 1

    # Calculate the proportion of samples for each page
    page_rank = {}
    for page in page_visits:
        page_rank[page] = page_visits[page] / n
    
    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialize the page rank dictionary
    page_rank = {}

    # Assign 1 / N to each page
    for page in corpus:
        page_rank[page] = 1 / len(corpus)

    # Repeat until no PageRank value changes by more than 0.001 between the current rank values and the new rank values
    convergence_achieved = False
    while not convergence_achieved:
        convergence_achieved = True
        # For each page, calculate the new rank value based on the current rank values
        for current_page in page_rank:
            # For all pages linked to the current page, calculate the sum of their page_rank values, devided by the number of links on the linked page
            sum_page_rank = 0
            for linked_page in corpus:
                if current_page in corpus[linked_page] and current_page != linked_page:
                    sum_page_rank += page_rank[linked_page] / len(corpus[linked_page])
            # Calculate the new rank value for the current page
            new_rank_value = (1 - damping_factor) / len(corpus) + damping_factor * sum_page_rank
            # Calculate the difference between the new rank value and the current rank value
            difference = abs(new_rank_value - page_rank[current_page])
            # Update the current rank value with the new rank value
            page_rank[current_page] = new_rank_value
            # If the difference between the new rank value and the current rank value is greater than 0.001, then set a convergence variable to False
            if difference > 0.001:
                convergence_achieved = False

    # Return the page rank dictionary
    return page_rank


if __name__ == "__main__":
    main()
