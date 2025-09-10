import logging
import os
import json
import psycopg2
from psycopg2 import sql
import azure.functions as func
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from crewai import Agent, Task, Crew, Process
from crewai import LLM
import tiktoken

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "questionsimilarity")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "AI@POC")

AZURE_API_KEY = os.getenv("AZURE_API_KEY", "420cf7e1e6434bc7bf60e8c180b4055f")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "https://excelsoftgpt4poc.openai.azure.com/")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")

os.environ["OPENAI_API_KEY"] = AZURE_API_KEY
os.environ["AZURE_API_BASE"] = AZURE_API_BASE
os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION

base_dir = os.path.dirname(os.path.abspath(__file__))
def calculate_tokens(text, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    return conn

class PDFRetrievalAgent(Agent):
    def __init__(self, retriever):
        super().__init__(
            role="PDF Retriever",
            backstory="I retrieve relevant information from educational materials stored as PDFs.",
            goal="Fetch the most relevant document excerpts based on a student's query.",
            verbose=True
        )
        self._retriever = retriever

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        query = task.description
        docs = self._retriever.invoke(query)
        
        
        retrieved_docs = []
        for doc in docs:
            retrieved_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return retrieved_docs

class EducationalLLMAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            role="Educational AI Evaluator/Examiner",
            backstory="Trained on academic datasets and pedagogical best practices, I specialize in evaluating student responses with precision. My purpose is to assess answers fairly while providing constructive feedback to enhance learning outcomes.",
            goal="Evaluate student responses based on defined criteria, provide scores, and deliver actionable feedback to improve understanding.",
            verbose=True
        )
        self.llm = llm

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        logging.info("EducationalLLMAgent: Executing task...")
        logging.info(f"Task Description: {task.description}")
        logging.info(f"Context: {task.context[0].description}") 
        context_str = "\n".join(task_item.description for task_item in task.context)
        # if not context or not isinstance(context, str) or len(context.strip()) == 0:
        #     return "This question isn't covered in your study materials."

        prompt = f"""   
        Current Document Context:
        {context_str}
        You are an Educational AI Teacher Evaluator/Examiner helping students with their specific study materials. 
        Query:
        {task.description}

        Don't give me SystemMessage and HumanMessage in your response.
        """

        answer = self.llm.call(prompt)
        return answer

def main(req: func.HttpRequest, res: func.Out[func.HttpResponse]) -> None:
    logging.info('Processing an ask-question request.')

    try:
        logging.info("Parsing request body...")
        logging.info(req)
        req_body = req.get_json()
        logging.info("Parsing request body1...")
        logging.info(req_body)
        question = req_body.get("question")
        url = req_body.get("url")

        if not question or not url:
            return func.HttpResponse("Missing 'question' or 'url' in request.", status_code=400)
        logging.info(f"Received url: {url}")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",           
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        db_path = os.path.join(base_dir, '..', 'dbs', url)
        db_path = os.path.abspath(db_path)

        logging.info(f"Loading FAISS from: {db_path}")

        if not os.path.exists(db_path):
            raise Exception(f"FAISS index path not found: {db_path}")
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        logging.info("Vector store loaded successfully.")
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
        )

        pdf_retrieval_agent = PDFRetrievalAgent(retriever)
        logging.info("PDF Retrieval Agent initialized.")
        llm = LLM(
            model="azure/gpt-4o",
            api_version=AZURE_API_VERSION
        )
        logging
        educational_agent = EducationalLLMAgent(llm)
        logging.info("Educational LLM Agent initialized.")
        retrieval_task = Task(
            description=question,
            expected_output="Relevant excerpts from the Source,Mark Schema Guidelines for evaluation.",
            agent=pdf_retrieval_agent
        )
        logging.info("Executing retrieval task...")
        retrieval_result = pdf_retrieval_agent.execute_task(retrieval_task)
        logging.info("Retrieval task completed.")
        logging.info(f"Retrieved {len(retrieval_result)} documents.")

        # Wrap each document into a Task object
        context_tasks = []
        for doc in retrieval_result:
            context_tasks.append(Task(
                description=doc["page_content"],
                expected_output="Relevant excerpt from document",
                agent=None  # No agent needed for context
            ))
        logging.info(f"Prepared {len(context_tasks)} context tasks for educational evaluation.")
        logging.info(f"Context Tasks: {context_tasks}")
        education_task = Task(
            description=question,
            expected_output="Evaluate the student response based on Source,Mark Schema Guidelines to give score and feedback.",
            agent=educational_agent,
            context=context_tasks
        )

        logging.info("Executing educational task...")
        logging.info("Executing educational task...")
        answer = educational_agent.execute_task(education_task)
        logging.info("Educational task completed.")
        logging.info(f"Generated answer: {answer}")
        #tokens_used_question = calculate_tokens(question)
        #tokens_used_answer = calculate_tokens(answer)
        #total_tokens_used = tokens_used_question + tokens_used_answer

        response_data = {
            "answer": answer,           
        }
        res.set(func.HttpResponse(json.dumps(response_data), status_code=200, mimetype="application/json"))
        # return func.HttpResponse(
        #     json.dumps(response_data),
        #     status_code=200,
        #     mimetype="application/json"
        # )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        res.set(func.HttpResponse(f"Error: {str(e)}", status_code=500))
        # return func.HttpResponse(
        #     json.dumps({"error": str(e)}),
        #     status_code=500,
        #     mimetype="application/json"
        # )
