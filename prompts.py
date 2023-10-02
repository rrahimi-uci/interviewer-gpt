interview_question_prompt = ''' We asked a candidate an interview question of {interview_question_asked}. 
Here is the summarized response: {candidate_response} 

Question : How do you evaluate the candidate's response?

Instructions:

1) The story should be specific about an industrial project in the domain of 
   engineering, product or marketing.
2) The story should be clear and easy to understand.
3) The story should be relevant to the question.

Please use the following template for your response :

Your general ranking should be only one of these choices:

1) weak, 
2) average, 
3) strong

With the Explanation of your ranking. Create response like the follwoing format:

"ranking": "week"; 
"ai-explanation": "your explanation"
'''

check_principles_promt ='''
We asked a candidate an interview question. Here is the sumerized response: {candidate_response}

Context: we have different leadership principles that we want to evaluate the candidate response against.
Here is the list of leadership principles represented as a json string. Key is the leadership principle and value 
is the definition of the leadership principle.
{leadership_principles}

Question : Is candidate response relevant to this leadership principles?.

Instructions:
1) Validate your answer by highlighting the related text extracted from the candidate response.
2) Your Response should be less than 200 words.
3) Use the format of principle name and your answer. 

For example, if you think the candidate response is relevant to the leadership principle "Customer Obsession", then your answer 
should look like "Customer Obsession": your answer. If it is not relevant to leadership principles, then write 'AI-could not get 
specific evidence in answer'. 

Create beautiful response like the follwoing format:

1- "principle_name";
"ai-explanation": "your answer";
----------------------------------------------------------------------------------------------------
2- "principle_name";
"ai-explanation": "your answer";
----------------------------------------------------------------------------------------------------
...
'''

ai_answer_promt ='''
Consider yourself as a candidate for software engineering manageral role. 
We ask you question : {interview_question_asked}

Please answer the mentioned question based on the following instructions:

1) Make sure you provide a high level hypothetical answer to the question first.
2) Support your answer with a acceptable story related to the question.

limit your answer to less than 500 words.
'''