# coach-gpt

This repository contains a collection of notebooks and associated code components designed for the GPT-powered application, specifically tailored to serve as a Behavioral Interview Assistant.

## Repository Structure

The repository is structured as follows:

- `outputs/`: contains general evaluation and detail evaluation when you are using test.py for review.
- `resources/`: It contains imporatnt files related to the application.
    -'conf.env/': contains important configuration. You need to update your openAI key to work.
    -'interview_questions.db': db that contains question. It is SQLite db
    -'interview_youtube.txt': sample interview results that we used in test.py and notebook
    -'leadership_principles.json': It contains Amazon leadership principles in json format. 
    -'random_story.txt': it is a random story which is used as the basedline for comparing similarity to leadership principles.
    -'sample_behavioral_questions.txt': You can add more question to this file in the newline. It will be automatically loaded to question db. 
- `coachgpt-poc.ipynb/`: This notebook has everything related to this app for testing.
- `coachgpt-poc.py/`: gradio app to run.
- `helpers.py/`: Helper functions and classes that are needed.
- `prompts.py/`: Related prompt for this project.
- `test.py/`: test function which is using a youtube mock interview for the question of 'Tell me about the time that you solved constumer problem.

## Getting Started

To get started with the application, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip3 install -r requirements.txt`.
3. Add your openAI key to .env file
4. Start the Flask web server by running `python3 coachgpt-poc.py`.
5. Open your web browser and navigate to ` http://127.0.0.1:7860` to use the application.





