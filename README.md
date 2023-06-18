# openai

# Github Automated Analysis

This Python code is a Streamlit application that performs automated analysis of Github repositories to evaluate the technical complexity of code snippets.

## Installation

To run this code, please follow these steps:

1. Clone the repository:

   ```bash
   git clone [repository URL]
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the OpenAI API key:

   - Obtain an API key from OpenAI. Visit the OpenAI website for instructions on obtaining an API key.

4. Run the Streamlit application:

   ```bash
   streamlit run github_automated_analysis.py
   ```

   The Streamlit application will open in your web browser.

## Usage

1. Enter your Github URL:

   - In the Streamlit application, you will see a text input field labeled "Enter OPENAI API KEY".
   - Enter your OPENAI API key
   - Then, provide the URL to your Github profile in the next text input field (e.g., `https://github.com/your-username`).
   - Click the "Submit" button to proceed.

2. Evaluation and Analysis:

   - The code will clone the provided Github repository into a temporary directory named `temp_repo`.
   - Unwanted files such as images, data files, and certain file extensions will be removed from the repository to focus on code snippets.
   - If any Jupyter notebooks (`*.ipynb` files) are found, they will be converted to Python files (`*.py`) using the `export_ipynb_to_py` script.
   - The code will evaluate the technical complexity of the code snippets using the OpenAI language model.
   - The complexity scores will be calculated and averaged for each code snippet.
   - The repository with the highest complexity score will be identified.

3. Results:

   - The most technically complex repository will be displayed in the Streamlit application, along with its complexity score.
   - Additionally, a detailed justification for the complexity score will be generated using the OpenAI language model and presented in the application.


## Contact

For any inquiries or support, please contact vijay99.mano@gmail.com.

