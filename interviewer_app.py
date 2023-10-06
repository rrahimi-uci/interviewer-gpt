from utility.helpers import *
from utility.prompts import *
import gradio as gr
from transformers import pipeline

# Open the random story response. It is used to define the baseline of similarity 
# between the candidate response and random story.
with open('resources/random_story.txt', 'r') as f:
    random_story = f.read()

# Load the leadership principles from the json file. This is used to evaluate the candidate response on how
# it is relevant to the leadership principles.
leadership_principles = read_json_file('resources/leadership_principles.json')

def generate_random_question(choices):
            if choices == "Coding Question":
                table = "coding_interview_questions"
                question = get_random_interview_question(table)
                return question
            elif choices == "ML System Design Question":
                table = "ml_system_design_interview_questions"
                question = get_random_interview_question(table)
                return question
            else:
                table = "leadership_interview_questions"    
                question = get_random_interview_question(table)
            return question

chat = ChatOpenAI(
                temperature = TEMPERTURE,
                 model_name="gpt-3.5-turbo", 
                 max_tokens = MAX_TOKENS, 
                 openai_api_key = OPENAI_API_KEY)

def evaluate_by_ai_interviewer(question_choices, candidate_response_str, random_question):
            
            # Summarize the candidate response to the question
            summarized_response_str = summarize_response(candidate_response_str)

            if question_choices == "Leadership and Behavioural Question":
                
                # Create a prompt template to use in langchain
                b_interview_question_prompt_template = PromptTemplate(
                    input_variables =["interview_question_asked","candidate_response"],
                    template = b_interview_question_prompt
                )
                b_interview_question_query = b_interview_question_prompt_template.format(
                        candidate_response = summarized_response_str,
                        interview_question_asked = random_question
                    )

                check_leadership_principle_prompt_template = PromptTemplate(
                    input_variables =["candidate_response", "leadership_principles"],
                    template = b_check_principles_promt
                )

                # Create a prompt template to use in langchain for checking how it has leadership principles in high level
                check_leadership_principle_prompt = check_leadership_principle_prompt_template.format(
                    candidate_response = summarized_response_str,
                    leadership_principles = json.dumps(leadership_principles))
                
                # Do multi-threading to speed up the process
                ai_evaluation_thread = ReturnValueThread(target=chat.predict, args=(b_interview_question_query,))
                ai_detailed_evaluation_thread = ReturnValueThread(target=chat.predict, args=(check_leadership_principle_prompt,))
                ai_answer_thread = ReturnValueThread(target=chat.predict, args=(b_ai_answer_promt.format(interview_question_asked = random_question),)) 
                sorted_tuple_list_thread = ReturnValueThread(target=calculate_similarity_to_leadership_principles, args=(leadership_principles, summarized_response_str, random_story))

                ai_evaluation_thread.start()
                ai_detailed_evaluation_thread.start()
                ai_answer_thread.start()
                sorted_tuple_list_thread.start()

                ai_evaluation_thread_res = ai_evaluation_thread.join()
                ai_detailed_evaluation_thread_res = ai_detailed_evaluation_thread.join()
                ai_answer_thread_res = ai_answer_thread.join()
                data = generate_pandas_df_from_dict(sorted_tuple_list_thread.join())
            
                return { ai_evaluation:ai_evaluation_thread_res,
                        ai_detailed_evaluation:ai_detailed_evaluation_thread_res,
                        ai_similarity_analysis:data,
                        ai_answer:ai_answer_thread_res}
            
            elif question_choices == "Coding Question":
                 # Create a prompt template to use in langchain
                interview_question_prompt_template = PromptTemplate(
                    input_variables =["interview_question_asked","candidate_response"],
                    template = c_interview_question_prompt
                )
                interview_question_query = interview_question_prompt_template.format(
                        candidate_response = summarized_response_str,
                        interview_question_asked = random_question
                )
                # Do multi-threading to speed up the process
                ai_evaluation_thread = ReturnValueThread(target=chat.predict, args=(interview_question_query,))
                ai_answer_thread = ReturnValueThread(target=chat.predict, args=(c_ai_answer_promt.format(interview_question_asked = random_question),)) 

                ai_evaluation_thread.start()
                ai_answer_thread.start()

                ai_evaluation_thread_res = ai_evaluation_thread.join()
                ai_answer_thread_res = ai_answer_thread.join()
            
                return { ai_evaluation:ai_evaluation_thread_res,
                        ai_detailed_evaluation:' ',
                        ai_similarity_analysis:None,
                        ai_answer:ai_answer_thread_res}
            else:
                # Create a prompt template to use in langchain
                interview_question_prompt_template = PromptTemplate(
                    input_variables =["interview_question_asked","candidate_response"],
                    template = ms_interview_question_prompt
                )
                interview_question_query = interview_question_prompt_template.format(
                        candidate_response = summarized_response_str,
                        interview_question_asked = random_question
                )
                # Do multi-threading to speed up the process
                ai_evaluation_thread = ReturnValueThread(target=chat.predict, args=(interview_question_query,))
                ai_answer_thread = ReturnValueThread(target=chat.predict, args=(ms_ai_answer_promt.format(interview_question_asked = random_question),)) 

                ai_evaluation_thread.start()
                ai_answer_thread.start()

                ai_evaluation_thread_res = ai_evaluation_thread.join()
                ai_answer_thread_res = ai_answer_thread.join()
            
                return { ai_evaluation:ai_evaluation_thread_res,
                        ai_detailed_evaluation:' ',
                        ai_similarity_analysis:None,
                        ai_answer:ai_answer_thread_res}
                  
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

def change_choice(choice): 
    if choice == "Coding Question" or choice == "ML System Design Question":
                return [gr.update(visible=False), gr.update(visible=False)]
    else:
                return [gr.update(visible=True), gr.update(visible=True)]


with gr.Blocks() as coach_gpt_gradio_ui:
    gr.Markdown(
    """
    # 🎤 Welcome to the 🧘🏻‍♂️ **Guru**, your AI Interviewer for ML Engineering Leaders and Manageral roles!
    
    ## 🤔 How it Works :

    1) Choose the type of question you want to be asked.
    2) Click on the button "**Generate Random Interview Question**" to generate a random interview question.
    3) If you choose leadership question you can record your answer and then we will do transcription OR Enter your response to the question. Try to use 
       less than **2000** words.
    4) Click on the button "**AI Evaluation of the Candidate Response**" to evaluate the your response.
    5) For ledership question, we also provide you "**Details Considering Different Leadership Principles**". These principle are based 
    on [**Amazon leadership principle**](https://www.amazon.jobs/content/en/our-workplace/leadership-principles 'Amazon leadership principle') 
    which are accepted in software industry as a guidline.The decomposed response to leadership principles will be displayed in the chart part using cosine 
    similarity for more insigths. It gives you a sense of how your response is related/ranked to different 
    leadership principles. 
    6) Once you are done with the question and answers you can clear the board and start over.

    ## 📊 AI Analysis Inerpretation :
    
    The high-level evaluation result will be displayed in the text box "**General Evaluation**". It will rank the answer to:

        1) Weak, 
        2) Average, 
        3) Good,
        4) Excellent
    
    We also provide the AI answer to the question to give you a sense of how the AI would answer the question for your 
    reference and learning.
    """)
    
    with gr.Column():
        question_choices = gr.Radio(
        ["Leadership and Behavioural Question", "ML System Design Question", "Coding Question"], 
        label = "🧘🏻‍♂️ What kind of question do you want me to ask?",
        value = "Leadership and Behavioural Question")
         
        btn_random_question = gr.Button("🎲 Generate Me Random Interview Question")
        random_question = gr.Textbox(label="❓Interview Question", )
        
        with gr.Column(visible=True) as audio_visibility:
            candidate_response_audio_input = gr.Audio(label="Record Your Response", 
                                                  type="numpy", 
                                                  source="microphone",
                                                  show_download_button=True,
                                                  interactive=True,)
        candidate_response_str = gr.Textbox(label="📝 Your Response", lines=20)
        
        evaluate_by_ai = gr.Button("🧘🏻‍♂️ Guru Evaluation of the Response")
        ai_evaluation = gr.Textbox(label= '🔍 High-Level Evaluation', lines=20)
        
        with gr.Column(visible=True) as behavioral_evaluation_visibility:
            ai_detailed_evaluation = gr.Textbox(label= '📑 Details Considering Amazon Leadership Principles', 
                                            lines=20, 
                                            interactive=True)
            ai_similarity_analysis= gr.BarPlot( x = "Leadership Principles",
                                            y = "Percentage",
                                            x_title = "Leadership Principles",
                                            y_title = "Percentage",
                                            title = "Similarity of the Interviewee's Response to the Leadership Principles",
                                            vertical = False,
                                            height= 600,
                                            width= 600,
                                            interactive=True)
        ai_answer = gr.Textbox(label= '🧘🏻‍♂️ Guru answer to the question', lines=20)
        btn_clear_board = gr.ClearButton(value="🧹 Clear Board", 
                                         components=[random_question, 
                                                     candidate_response_str, 
                                                     ai_evaluation, 
                                                     ai_detailed_evaluation, 
                                                     ai_similarity_analysis, 
                                                     ai_answer])
        
        question_choices.change(fn=change_choice, 
                                inputs=[question_choices],
                                outputs=[audio_visibility, behavioral_evaluation_visibility])
        
        btn_random_question.click(fn=generate_random_question, 
                                  inputs=[question_choices],
                                  outputs=[random_question], 
                                  api_name="generate_random_question")
        
        candidate_response_audio_input.stop_recording(fn=transcribe, 
                                            inputs=[candidate_response_audio_input],
                                            outputs=[candidate_response_str], 
                                            api_name="audio_transcribe")

        evaluate_by_ai.click(
                  fn=evaluate_by_ai_interviewer, 
                  inputs=[question_choices, candidate_response_str, random_question], 
                  outputs = [ai_evaluation, ai_detailed_evaluation, ai_similarity_analysis, ai_answer], 
                  api_name="evaluate_by_ai_interviewer")

coach_gpt_gradio_ui.launch(share=True, width=500, height=700, debug=True)