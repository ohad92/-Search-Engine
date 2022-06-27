from flask import Flask, request, jsonify
import pickle
import numpy as np
import math
from collections import Counter
import json
import pandas as pd

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


## reading the inverted_index file
with open('index.pkl', 'rb') as f:
    data = pickle.load(f)

with open('title_index.pkl', 'rb') as f:
    title_index = pickle.load(f)

### code from Assignment 4 ###

words, pls = zip(*data.posting_lists_iter())
# print ((words))
# print ((pls))

## calculating term_total
term_total_dict = {}
for i in range(len(words)):
  sum = 0
  for j in pls[i]:
    sum+=j[1]
  term_total_dict[words[i]] = sum


###############################################
def generate_query_tfidf_vector(query_to_search,index):   
    epsilon = .0000001
    total_vocab_size = len(term_total_dict)
    Q = np.zeros((total_vocab_size))
    term_vector = list(term_total_dict.keys())    
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in term_total_dict.keys(): #avoid terms that do not appear in the index.               
            tf = counter[token]/len(query_to_search) # term frequency divded by the length of the query
            df = index.df[token]            
            idf = math.log((len(index.DL))/(df+epsilon),10) #smoothing
            
            try:
                ind = term_vector.index(token)
                Q[ind] = tf*idf                    
            except:
                pass
    return Q

###############################################
def generate_document_tfidf_matrix(query_to_search,index,words,pls):
    total_vocab_size = len(term_total_dict)
    candidates_scores = get_candidate_documents_and_scores(query_to_search,index,words,pls) #We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)
    
    D.index = unique_candidates
    D.columns = term_total_dict.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key    
        D.loc[doc_id][term] = tfidf

    return D
     
###############################################3
def cosine_similarity(D,Q):
    dictionary = {}
    for index,row in D.iterrows():
      dictionary[index]=(row.dot(Q))/(math.sqrt(row.dot(row))*math.sqrt(Q.dot(Q)))
    return dictionary

###############################################3
def get_top_n(sim_dict,N=3): 
    return sorted([(doc_id,round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]

###############################################

def get_candidate_documents_and_scores(query_to_search,index,words,pls):
    candidates = {}
    N = len(index.DL)        
    for term in np.unique(query_to_search):        
        if term in words:            
            list_of_doc = pls[words.index(term)]                        
            normlized_tfidf = [(doc_id,(freq/index.DL[doc_id])*math.log(N/index.df[term],10)) for doc_id, freq in list_of_doc]
                        
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf               
        
    return candidates
















@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    print(query)
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query = query.lower().split()
    # print ("data: ",data.df)
    query_idf_vector=generate_query_tfidf_vector(query,data)
    doc_idf=generate_document_tfidf_matrix(query,data,words,pls)
    calculate_cosSim=cosine_similarity(doc_idf,query_idf_vector)
    dictionary_=get_top_n(calculate_cosSim,100)
    #print (query_idf_vector.sum())
    #print ("doc_idf: ",doc_idf)
    #print ("calculate_cosSim: ",calculate_cosSim)
    #print ("dictionary_: ", dictionary_)

    for i in dictionary_: #i is tuple (wiki_id,score)
      tup=(i[0],"get_title_of_wikiidPage") # need to get the title by the wiki_id
      res.append(tup)
    print (res)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for word in query:
        res += title_index.word_to_titles[word]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
