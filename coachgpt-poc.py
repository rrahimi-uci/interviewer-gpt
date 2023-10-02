from helpers import *
from prompts import *
import gradio as gr

# Open the random story response. It is used to define the baseline of similarity 
# between the candidate response and random story.
with open('resources/random_story.txt', 'r') as f:
    random_story = f.read()

# Load the leadership principles from the json file. This is used to evaluate the candidate response on how
# it is relevant to the leadership principles.
leadership_principles = read_json_file('resources/leadership_principles.json')

question = ''

def generate_random_question():
            question = get_random_interview_question()
            return question

llm = OpenAI(temperature = TEMPERTURE, 
             model = MODEL, 
             max_tokens = MAX_TOKENS, 
             openai_api_key = OPENAI_API_KEY)

def evaluate_by_ai_interviewer(candidate_response_str):
            
            # Summarize the candidate response to the question
            summarized_response_str = summarize_response(candidate_response_str)
            
            # Create a prompt template to use in langchain
            interview_question_prompt_template = PromptTemplate(
                input_variables =["interview_question_asked","candidate_response"],
                template = interview_question_prompt
            )
            interview_question_query = interview_question_prompt_template.format(
                    candidate_response = summarized_response_str,
                    interview_question_asked = question
                )

            check_leadership_principle_prompt_template = PromptTemplate(
                input_variables =["candidate_response", "leadership_principles"],
                template = check_principles_promt
            )

            # Create a prompt template to use in langchain for checking how it has leadership principles in high level
            check_leadership_principle_prompt = check_leadership_principle_prompt_template.format(
                candidate_response = summarized_response_str,
                leadership_principles = json.dumps(leadership_principles))

            sorted_tuple_list = calculate_similarity_to_leadership_principles(leadership_principles, summarized_response_str, random_story)
    
            data =  generate_pandas_df_from_dict(sorted_tuple_list) 
            
            return { ai_evaluation:llm(interview_question_query),
                    ai_detailed_evaluation:llm(check_leadership_principle_prompt),
                    ai_similarity_analysis:data}

with gr.Blocks() as coach_gpt_gradio_ui:
    gr.Markdown(
    """
    # 🎤Welcome to the AI Behavioral Interviewer for Software Engineers, Managers and Product engineers
    
    ## 📝 Instructions :

    1) Click on the button "Generate Random Question" to generate a random question.
    2) Click on the button "Evaluate By AI Interviewer" to evaluate the candidate response.
    
    ## 📊 AI Analysis Inerpretation :
    
    The evaluation result will be displayed in the text box "General Evaluation" and "Details Considering Different 
    Leadership Principles". The decomposed response to leadership principles will be displayed in the image part using cosine 
    similarity for more insigths. It gives you a sense of how the candidate response is related/ranked to different leadership 
    principles. 
    
    """)
    with gr.Column():
        btn_random_question = gr.Button("Generate Random Question")
        random_question = gr.Textbox(label="Behavioral Interview Question", )
        candidate_response_str = gr.Textbox(label="Candidate Response", lines=20)
        evaluate_by_ai = gr.Button("Evaluate By AI Interviewer")
        ai_evaluation = gr.Textbox(label= 'General Evaluation', lines=10)
        ai_detailed_evaluation = gr.Textbox(label= 'Details Considering Different Leadership Principles', lines=20)
        ai_similarity_analysis= gr.BarPlot( x = "Leadership Principles",
                                            y = "Percentage",
                                            x_title = "Leadership Principles",
                                            y_title = "Percentage",
                                            title = "Similarity of the Interviewee's Response to the Leadership Principles",
                                            vertical = False,
                                            height= 600,
                                            width= 1000)

        btn_random_question.click(fn=generate_random_question, 
                                  outputs=[random_question], 
                                  api_name="generate_random_question")

        evaluate_by_ai.click(
                  fn=evaluate_by_ai_interviewer, 
                  inputs=[candidate_response_str], 
                  outputs = [ai_evaluation, ai_detailed_evaluation, ai_similarity_analysis], 
                  api_name="evaluate_by_ai_interviewer")

coach_gpt_gradio_ui.launch(share=True, 
                           width=600, 
                           height=600)