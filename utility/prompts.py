b_interview_question_prompt = ''' We asked a candidate an interview question of {interview_question_asked}.
Here is the summarized response: {candidate_response}

Question: How do you evaluate the candidate's response?

Instructions:

1) The story should be related to engineering, product, or marketing experience.
2) The story should be clear and easy to understand.
3) The story should be relevant to the question.

Please use the following template for your response:

Your general ranking should be only one of these choices:

1) Weak, 
2) Average, 
3) Good,
4) Excellent

With the Explanation of your ranking. Create a response like the following format:

Ranking: Weak; 
AI-Explanation: your explanation
'''

b_check_principles_promt ='''
We asked a candidate an interview question. Here is the sumerized response: {candidate_response}

Context: we have different leadership principles that we want to evaluate the candidate response against.
Here is the list of leadership principles represented as a json string. Key is the leadership principle and value 
is the definition of the leadership principle.
{leadership_principles}

Question : Is candidate response relevant to this leadership principles?.

Instructions:
1) Validate your answer by highlighting the related text extracted from the candidate response.
2) Use the format of principle name and your answer.
3) Use all the leadership principles in your answer.

For example, if you think the candidate response is relevant to the leadership principle "Customer Obsession", then your answer 
should look like "Customer Obsession": your answer. If it is not relevant to leadership principles, then write 'AI-could not get 
specific evidence in answer'. 

Create a response like the follwoing format:

1- Principle Name;
AI-Explanation: your answer;
----------------------------------------------------------------------------------------------------
2- Principle Name;
AI-Explanation: your answer;
----------------------------------------------------------------------------------------------------
...
'''

b_ai_answer_promt ='''
Consider yourself as a candidate for software engineering manageral role. 
We ask you question : {interview_question_asked}

Please answer the mentioned question based on the following instructions:

1) Make sure you provide a high level hypothetical answer to the question first.
2) Support your answer with a acceptable story related to the question.

'''

c_interview_question_prompt = ''' We asked a candidate an coding interview question of {interview_question_asked}.
Here is the summarized response: {candidate_response}

Question: How do you evaluate the candidate's response?

Instructions:

1) It should contains python code.
2) How the python code is written. Does it follow the best practices?
3) Does the code solve the problem?
4) Does code have good level of comments and explaination?

Please use the following template for your response:

Your general ranking should be only one of these choices:

1) Weak, 
2) Average, 
3) Good,
4) Excellent

With the Explanation of your ranking. Create a response like the following format:

Ranking: Weak; 
AI-Explanation: your explanation
'''

c_ai_answer_promt ='''
Consider yourself as a candidate for software engineering role. 
We ask you a coding question : {interview_question_asked}

Please answer the mentioned coding question based on the following instructions:

1) Write a python code with a simple test that works.
2) Explain your code.
3) Your Response should be less than 4000 words.

Create a response like the following format:

Code: your code
AI-Explanation: your explanation
'''

ms_interview_question_prompt = ''' We asked a candidate an machine learning System Design question of {interview_question_asked}.
Here is the summarized response: {candidate_response}

Question: How do you evaluate the candidate's response?

Instructions:

1) It should contains important blocks of the machine learning system components.
2) It should contains important machine learning models that are used in the system.
3) It should talk about important offline and online metrics.
4) It should talk about important features use in models.

Please use the following template for your response:

Your general ranking should be only one of these choices:

1) Weak, 
2) Average, 
3) Good,
4) Excellent

With the Explanation of your ranking. Create a response like the following format:

Ranking: Weak; 
AI-Explanation: your explanation
'''

ms_ai_answer_promt ='''
Consider yourself as a candidate for machine learning engineering role. 
We ask you the following machine learning system design question : {interview_question_asked}

Please answer the mentioned question based on the following instructions:

1) It should contains important blocks of the machine learning system components.
2) It should contains important machine learning models that are used in the system.
3) It should talk about important offline and online metrics.
4) It should talk about important features use in machine learning models.

Create a response like the following format:

Important Machine Learning Components: your explanation
Important online and offline metrics for system evaluation: your explanation
Important Machine Learning Models: your explanation
Important Features: your explanation
'''