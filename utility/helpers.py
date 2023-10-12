import os
import json
import sqlite3
import random
import threading
import sys
import pandas as pd
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt

current_directory = os.getcwd()
load_dotenv('resources/conf.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TEMPERTURE = float(os.getenv('TEMPERTURE'))
MODEL = os.getenv('MODEL')
MAX_TOKENS = int(os.getenv('MAX_TOKENS'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
CHUNK_OVERLAP  = int(os.getenv('CHUNK_OVERLAP'))

def inner_vector_product(vector_1, vector_2):
    """
    This function calculates the inner vector product of two vectors.
    """
    import numpy as np
    a = np.array(vector_1)
    b = np.array(vector_2)

    return np.dot(a, b)

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data

class DbOps:
    def __init__(self, db_name) -> None:
        """
        Initializes a new instance of the DbOps class.
        Parameters:
        -----------
        db_name : str
            The name of the SQLite database to connect to.
        """
        # create a connection to the database
        self.db_name = db_name
        conn = sqlite3.connect(self.db_name)
        conn.close()

    def create_table(self, table_name):
        """
        Creates a new table in the database.
        Parameters:
        -----------
        table_name : str
            The name of the table to create.
        columns : str
            The columns of the table to create.
        """
        conn = sqlite3.connect(self.db_name)

        # check if the table exists, if not create it
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            conn.execute(f"CREATE TABLE {table_name} (question TEXT, tags TEXT);")
            conn.close() 
    
    def add_question(self, table, question, tags='##'):
        """
        Adds a new question to the 'behavioral_question' table in the database.
        Parameters:
        -----------
        question : str
            The question to add to the database.
        tags : str, optional
            The tags associated with the question. Default is '-'.
        """
        conn = sqlite3.connect(self.db_name)
        # insert a new question into the table
        conn.execute(f"INSERT INTO {table} (question, tags) VALUES ('{question}','{tags}')")
        # commit the changes
        conn.commit()
        # close the connection
        conn.close()
    
    def get_random_element(self, table):
        """
        Returns a random question from the 'behavioral_question' table in the database.
        Returns:
        --------
        str
            A random question from the 'behavioral_question' table in the database.
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        random_index = random.randint(0, count-1)
        cursor.execute(f"SELECT question FROM {table} LIMIT 1 OFFSET {random_index}")
        question = cursor.fetchone()[0]
        conn.close()        
        return question
   
    def populate_db(self, table, path_to_file):
       with open(path_to_file, 'r') as f:
            lines = f.readlines()
       
       for line in lines:
            question = line.strip().lower()
            self.add_question(table, question, '##')

def get_random_interview_question(table_name):
        """
        This function returns a random interview question from the database.
        """
        db = DbOps('resources/interview_questions.db')

        db.create_table('coding_interview_questions')
        db.create_table('leadership_interview_questions')
        db.create_table('ml_system_design_interview_questions')
        
        if table_name == 'coding_interview_questions':
            db.populate_db('coding_interview_questions', 'resources/sample_coding_questions.txt')
        
        if table_name == 'leadership_interview_questions':
            db.populate_db('leadership_interview_questions', 'resources/sample_behavioral_questions.txt')
        
        if table_name == 'ml_system_design_interview_questions':
            db.populate_db('ml_system_design_interview_questions', 'resources/sample_ML_System_Design_questions.txt')
        
        interview_question = db.get_random_element(table_name)
        return interview_question 

def get_transcript_from_youtube_video(video_id, file_path):
    """
    Gets the transcript from a YouTube video.
    Parameters:
    -----------
    video_id : str
        The ID of the YouTube video.
    Returns:
    --------
    str
        The transcript of the YouTube video.
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    res = ''
    for txt in transcript:
        res += ' ' + txt['text']

    with open(file_path, 'w', encoding='utf-8') as interview:
        interview.write(res) 
    return res

def summarize_response(txt):
    # Instantiate the LLM model
    llm = ChatOpenAI(temperature = TEMPERTURE, 
                 model = MODEL,
                 max_tokens = 2000,
                 openai_api_key = OPENAI_API_KEY)
    
    if len(txt) > int(CHUNK_SIZE) :
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = CHUNK_SIZE,
        chunk_overlap  = CHUNK_OVERLAP,
        length_function = len,
        add_start_index = True)
        
        texts = text_splitter.split_text(txt)
        # Create multiple documents
        docs = [Document(page_content=t) for t in texts]
        # Text summarization
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        return chain.run(docs)
    else:
        return llm(txt)

def calculate_similarity_to_leadership_principles(leadership_principles, 
                                                  summarized_response, 
                                                  random_story):
    """
    This function calculates the similarity of the summarized response to the leadership principles.
    """
    # Define embedding
    embedding = OpenAIEmbeddings(client=None)

    leadership_principles_embedding = {}
    # Embedding leadership principles
    for key, value in leadership_principles.items():
        leadership_principles_embedding[key] = embedding.embed_query(value)

    # Embedding summarized text, interviewee response
    summarized_response_vector = embedding.embed_query(summarized_response)
    # Embedding random story as a baseline
    random_story_vector = embedding.embed_query(random_story)
    # Normalize consider random story
    baseline_weigth = inner_vector_product(summarized_response_vector, random_story_vector)

    candidate_response_similarity_to_leadership_principles = {}

    sum = 0.0
    for key, value in leadership_principles_embedding.items():
        candidate_response_similarity_to_leadership_principles[key] = abs(inner_vector_product(summarized_response_vector, value) - baseline_weigth)
        sum += candidate_response_similarity_to_leadership_principles[key]

    for key, value in candidate_response_similarity_to_leadership_principles.items():
        candidate_response_similarity_to_leadership_principles[key] = (value/sum)*100

    sorted_candidate_response_similarity_to_leadership_principles = sorted(candidate_response_similarity_to_leadership_principles.items(), 
                                                                key=lambda x:x[1], 
                                                                reverse=True)
    return sorted_candidate_response_similarity_to_leadership_principles

def plot_horizontal_bar_chart(data_dict):
    """
    This function takes in a list of tuples of keys and values and plots them as a horizontal bar chart using matplotlib.
    """
    keys = []
    values = []

    for item in data_dict:
        keys.append(item[0])
        values.append(item[1])
        
    l=[]
    for i in range(0, len(keys)+1):
        l.append(tuple(np.random.choice(range(0, 2), size=3)))

    # Set the size of the figure
    plt.figure(figsize=(8, 8))

    # Create the horizontal bar chart with the extracted keys and values
    plt.barh(keys[::-1], values[::-1], color=l, edgecolor='black', linewidth=1)

    # Set the title and axis labels
    plt.title('Similarity of the Interviewee\'s Response to the Leadership Principles')
    plt.xlabel('Percentage')
    plt.ylabel('Leadership Principles')

    # Show the horizontal bar chart
    plt.show()

def generate_pandas_df_from_dict(tuple_list):
    """
        This function takes in a dictionary of keys and values and converts it into a pandas dataframe.
    """
    keys = []
    values = []

    for item in tuple_list:
        keys.append(item[0])
        values.append(item[1])

    return pd.DataFrame({"Leadership Principles":keys, "Percentage":values})

class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return  # could alternatively raise an exception, depends on the use case
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  # properly handle the exception

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result 