from helpers import *
from prompts import *
import gradio as gr
from transformers import pipeline

# Open the random story response. It is used to define the baseline of similarity 
# between the candidate response and random story.
with open('resources/random_story.txt', 'r') as f:
    random_story = f.read()

# Load the leadership principles from the json file. This is used to evaluate the candidate response on how
# it is relevant to the leadership principles.
leadership_principles = read_json_file('resources/leadership_principles.json')

def generate_random_question():
            question = get_random_interview_question()
            return question

llm = OpenAI(temperature = TEMPERTURE, 
             model = MODEL, 
             max_tokens = MAX_TOKENS, 
             openai_api_key = OPENAI_API_KEY)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", 
                 max_tokens = MAX_TOKENS, 
                 openai_api_key = OPENAI_API_KEY)

def evaluate_by_ai_interviewer(candidate_response_str, random_question):
            
            # Summarize the candidate response to the question
            summarized_response_str = summarize_response(candidate_response_str)
            
            # Create a prompt template to use in langchain
            interview_question_prompt_template = PromptTemplate(
                input_variables =["interview_question_asked","candidate_response"],
                template = interview_question_prompt
            )
            interview_question_query = interview_question_prompt_template.format(
                    candidate_response = summarized_response_str,
                    interview_question_asked = random_question
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

            #print(random_question) 
            data =  generate_pandas_df_from_dict(sorted_tuple_list)
            #print(ai_answer_promt.format(interview_question_asked = random_question)) 
            
            return { ai_evaluation:chat.predict(interview_question_query),
                    ai_detailed_evaluation:chat.predict(check_leadership_principle_prompt),
                    ai_similarity_analysis:data,
                    ai_answer:chat.predict(ai_answer_promt.format(interview_question_asked = random_question) )}

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", max_new_tokens = 1000)

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    res = ''
    splited_audio = np.array_split(y, 10)
    for audio_chunck in splited_audio:
        res += transcriber({"sampling_rate": sr, "raw": audio_chunck})["text"]
    
    return res


with gr.Blocks() as coach_gpt_gradio_ui:
    gr.Markdown(
    """
    # 🎤 Welcome to the AI Behavioral Interviewer for Software Engineer Leaders and Managers
    
    ## 📝 Instructions :

    1) Click on the button "**Generate Random Interview Question**" to generate a random interview question.
    2) You can record your answer and then we will do transcription OR Enter your response to the question. Try to use 
       less than **1000** words.
    3) Click on the button "**AI Evaluation of the Candidate Response**" to evaluate the your response.
    
    ## 📊 AI Analysis Inerpretation :
    
    The high-level evaluation result will be displayed in the text box "**General Evaluation**". It will rank the answer to:

        1) Weak, 
        2) Average, 
        3) Good,
        4) Excellent
    
    We also provide you "**Details Considering Different Leadership Principles**". These principle are based 
    on [**Amazon leadership principle**](https://www.amazon.jobs/content/en/our-workplace/leadership-principles 'Amazon leadership principle') which are accepted in software industry as a guidline.

    The decomposed response to leadership principles will be displayed in the chart part using cosine 
    similarity for more insigths. It gives you a sense of how your response is related/ranked to different 
    leadership principles. 
    
    Finally we provide the AI answer to the question to give you a sense of how the AI would answer the question 
    for your reference and guidline.
    """)
    
    with gr.Column():
        btn_random_question = gr.Button("Generate Random Interview Question")
        random_question = gr.Textbox(label="Behavioral Interview Question", )
        candidate_response_audio_input = gr.Audio(label="Record Your Response", 
                                                  type="numpy", 
                                                  source="microphone",
                                                  show_download_button=True)
        candidate_response_str = gr.Textbox(label="Candidate Response", lines=10)
        evaluate_by_ai = gr.Button("AI Evaluation of the Candidate Response")
        ai_evaluation = gr.Textbox(label= 'High-Level Evaluation', lines=10)
        ai_detailed_evaluation = gr.Textbox(label= 'Details Considering Different Leadership Principles', lines=10)
        ai_similarity_analysis= gr.BarPlot( x = "Leadership Principles",
                                            y = "Percentage",
                                            x_title = "Leadership Principles",
                                            y_title = "Percentage",
                                            title = "Similarity of the Interviewee's Response to the Leadership Principles",
                                            vertical = False,
                                            height= 600,
                                            width= 1000)
        ai_answer = gr.Textbox(label= 'AI answer to the question', lines=10)
        
        btn_random_question.click(fn=generate_random_question, 
                                  outputs=[random_question], 
                                  api_name="generate_random_question")
        
        candidate_response_audio_input.stop_recording(fn=transcribe, 
                                            inputs=[candidate_response_audio_input],
                                            outputs=[candidate_response_str], 
                                            api_name="audio_transcribe")

        evaluate_by_ai.click(
                  fn=evaluate_by_ai_interviewer, 
                  inputs=[candidate_response_str, random_question], 
                  outputs = [ai_evaluation, ai_detailed_evaluation, ai_similarity_analysis, ai_answer], 
                  api_name="evaluate_by_ai_interviewer")

coach_gpt_gradio_ui.launch(share=True, width=600, height=600, debug=True)