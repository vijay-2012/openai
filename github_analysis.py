import os
import re
from glob import glob
import openai
from github import Github
import git
import shutil
from pathlib import Path
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
import streamlit as st
from notebook_converter import export_ipynb_to_py

st.set_page_config(page_title='Github Automated Analysis')

openai_api_key = st.text_input('Enter OPENAI API KEY', type='password')
print(type(openai_api_key))
user_url = st.text_input('Enter your Github URL')
user_id = user_url.split('/')[-1]

os.environ["OPENAI_API_KEY"] = openai_api_key

if st.button('Submit'):

    # Initialize OpenAI model
    llm = OpenAI(temperature=0.7, model_name='text-davinci-003', max_tokens=1024, max_retries=2)

    # Prompt engineering
    prompt_template = """
    Please evaluate the technical complexity of the following code snippets on a scale of 0 to 1, where 0 is not complex at all and 1 is extremely complex. Tell only the score between 0 and 1:

    {code}
    """

    if os.path.exists('temp_repo'):
        shutil.rmtree('temp_repo')

    # Function to preprocess code
    def preprocess_code(code):
        # Remove comments
        code = re.sub(r"#.*", "", code)
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"/\*[\s\S]*?\*/", "", code)

        # Remove strings
        code = re.sub(r"\".*?\"", "", code)
        code = re.sub(r"\'.*?\'", "", code)

        return code

    # Clone the repository
    def clone_repository(repo_url, repo_path):
        git.Repo.clone_from(repo_url, repo_path)

    # Remove unwanted files
    def remove_unwanted_files(repo_path):

        file_extensions = ["*.png", "*.svg", "*.sqlite3", "*.pyc", "*.ttf", "*.tflite", "*.jpg", "*.npy", "*.txt", "*.h5", "*.md", "*.xlsx", "*.csv", 'LICENSE', "*.jar", "*.jpeg", "*.ico"]

        unwanted_files = []
        for file_extension in file_extensions:
            unwanted_files.extend(Path(repo_path).rglob(file_extension))
        
        print(unwanted_files)
        for file_path in unwanted_files:
            try:
                os.remove(file_path)
            except FileNotFoundError as e:
                print(e)

    def notebook_convert(repo_path):

        notebook_files = []
        notebook_files.extend(Path(repo_path).rglob('*.ipynb'))  

        for notebook_file_path in notebook_files:

            notebook_name = os.path.basename(notebook_file_path).split('.')[0]
            notebook_path =  '/'.join(str(notebook_file_path).split('/')[:-1])

            py_filepath = notebook_path + '/' + notebook_name + '-exported.py'

            export_ipynb_to_py(notebook_file_path, py_filepath)

            os.remove(notebook_file_path)


    # Load repository documents
    def load_repository_documents(repo_path):
        loader = DirectoryLoader(repo_path, loader_cls=TextLoader)
        return loader.load()

    # Split documents into chunks
    def split_documents_into_chunks(documents):
        document_chunks = []
        splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
        for document in documents:
            for chunk in splitter.split_text(document.page_content):
                document_chunks.append(Document(page_content=chunk, metadata=document.metadata))
        return document_chunks

    # Calculate complexity score for a chunk
    def calculate_complexity_score(chunk):
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["code"])
        chain = LLMChain(llm=llm, prompt=PROMPT)
        result = chain({"code": chunk.page_content}, return_only_outputs=True)
        try:
            print(result['text'])
            return float(result['text'].strip())
        except:
            try:
                print(result['text'])
                score = re.search(r"\b(0(?:\.\d+)?|1(?:\.0)?)\b", result['text'].strip()).group()
                return float(score)
            except (ValueError, AttributeError):
                print(result['text'])
                return 0.3

    # Evaluate complexity scores for a repository
    def evaluate_complexity_scores(repo_url):
        repo_path = 'temp_repo'
        clone_repository(repo_url, repo_path)

        remove_unwanted_files(repo_path)

        notebook_convert(repo_path)

        documents = load_repository_documents(repo_path)
        preprocessed_documents = [Document(page_content=preprocess_code(doc.page_content), metadata=doc.metadata) for doc in documents]
        document_chunks = split_documents_into_chunks(preprocessed_documents)

        complexity_scores_list = [calculate_complexity_score(doc) for doc in document_chunks]

        shutil.rmtree(repo_path)
        if len(complexity_scores_list) != 0:
            average_complexity_score = sum(complexity_scores_list) / len(complexity_scores_list)
        else:
            average_complexity_score = 0
        return average_complexity_score

    # Get user repositories
    user_repositories = Github().get_user(user_id).get_repos()

    # Evaluate complexity scores for each repository
    complexity_scores = {}

    for repo in user_repositories:
        complexity_scores[repo.name] = evaluate_complexity_scores(repo.clone_url)

    # Find the repository with the highest complexity score
    most_complex_repo = max(complexity_scores, key=complexity_scores.get)

    print(f"The most technically complex repository is: {most_complex_repo} with a score of {complexity_scores[most_complex_repo]:.2f}")
    st.write(f"The most technically complex repository is: {most_complex_repo}.")
    st.write(f"The URL to the repo -> https://www.github.com/{user_id}/{most_complex_repo}")
    
    # Justification prompt engineering
    justification_prompt_template = """
    Given the code snippets from the repository '{repo_name}', the model has determined it to be the most technically complex. Please provide a detailed analysis justifying this complexity score.
    """

    # Get the justification
    PROMPT = PromptTemplate(template=justification_prompt_template, input_variables=["repo_name"])
    chain = LLMChain(llm=llm, prompt=PROMPT)
    justification = chain({"repo_name": most_complex_repo}, return_only_outputs=True)

    st.write("Justification:")
    st.markdown(justification["text"], unsafe_allow_html=True)
