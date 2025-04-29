from langdetect import detect
from googletrans import Translator
from transformers import pipeline
import os
import re
import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage, SystemMessage
from sklearn.metrics import cohen_kappa_score

# Initialize models
llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key="APIKEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def assign_trait_weights(question):
    """Uses Mistral AI to dynamically assign weightage to grading traits."""
    weightage_prompt = (
        "You are an expert evaluator. Analyze the given question and assign appropriate weightage (out of 100%) "
        "to the following grading traits:\n"
        "1. **Content Accuracy**: Importance of factual correctness and completeness.\n"
        "2. **Coherence**: Importance of logical flow and clarity.\n"
        "3. **Vocabulary**: Importance of precise wording and expression.\n"
        "4. **Grammar**: Importance of correct grammar and readability.\n\n"
        "### **STRICT OUTPUT FORMAT:**\n"
        "- Provide output **ONLY** in JSON format (without explanation).\n"
        "- Ensure the weights sum **exactly** to 100.\n"
        '- Example: { "content_accuracy": 40, "coherence": 30, "vocabulary": 20, "grammar": 10 }'
    )

    system_message = SystemMessage(content=weightage_prompt)
    user_message = HumanMessage(content=f"Question: {question}")
    response = llm([system_message, user_message])

    try:
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            weights = json.loads(match.group(0))
            total_weight = sum(weights.values())
            if total_weight != 100:
                weights = {k: round((v / total_weight) * 100, 2) for k, v in weights.items()}
            return weights
        raise ValueError("No valid JSON found in response.")
    except Exception as e:
        print(f"Error parsing weights: {e}")
        return {"content_accuracy": 40, "coherence": 30, "vocabulary": 20, "grammar": 10}

def calculate_qwk(rater1_scores, rater2_scores, bins=[0, 4, 7, 10]):
    """Calculates QWK between two sets of scores."""
    if len(rater1_scores) != len(rater2_scores):
        raise ValueError("Input arrays must have the same length")

    def bin_scores(scores):
        return np.digitize(scores, bins, right=True) - 1

    model_binned = bin_scores(np.array(rater1_scores))
    human_binned = bin_scores(np.array(rater2_scores))

    if len(np.unique(human_binned)) == 1 and len(np.unique(model_binned)) == 1:
        return float('nan')

    all_labels = np.unique(np.concatenate([model_binned, human_binned]))
    return cohen_kappa_score(human_binned, model_binned,
                           labels=all_labels,
                           weights='quadratic')

# [Previous PDF processing functions remain the same]
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def create_faiss_index(texts, embedding_model):
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Upload and Process PDF
def process_lecture_pdf(pdf_path):
    """Processes a PDF, extracts text, and builds a FAISS index."""
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text:
        raise ValueError("No text could be extracted from the PDF.")

    # Split text into smaller chunks
    chunk_size = 300
    texts = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]

    # Create a FAISS index
    index, embeddings = create_faiss_index(texts, embedding_model)
    return texts, index

def retrieve_context(question, texts, index, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(question_embedding), top_k)
    return [texts[i] for i in indices[0]]

def integrated_evaluation(question, student_answer, texts, index):
    """Integrated evaluation using RAG and dynamic weights."""

    # Get dynamic weights
    weights = assign_trait_weights(question)

    # Retrieve relevant contexts
    relevant_contexts = retrieve_context(question, texts, index, embedding_model)
    combined_context = " ".join(relevant_contexts)

    evaluation_prompt = f"""Using the following context from lecture notes, evaluate the student's answer:

Context:
{combined_context}

Evaluate based on these weighted criteria:
1. Content Accuracy ({weights['content_accuracy']}%): Factual correctness and completeness
2. Coherence ({weights['coherence']}%): Logical flow and clarity
3. Vocabulary ({weights['vocabulary']}%): Word choice and expression
4. Grammar ({weights['grammar']}%): Grammar and sentence structure

Provide:
1. Detailed feedback for each criterion
2. Score for each criterion (out of 10)
3. Final weighted score (out of 10)
4. End with exactly: 'Overall Score: X.X/10'"""

    system_message = SystemMessage(content=evaluation_prompt)
    user_message = HumanMessage(content=f"Question: {question}\nStudent Answer: {student_answer}")

    # Get primary evaluation
    response = llm([system_message, user_message])

    # Get second evaluation for QWK
    second_eval = llm([
        SystemMessage(content="Rate this answer numerically from 0-10 based on the question and context."),
        user_message
    ])

    # Extract scores and calculate QWK
    try:
        first_score = float(re.search(r"Overall Score:\s*(\d+\.?\d*)\/10", response.content).group(1))
        second_score = float(re.search(r"(\d+\.?\d*)", second_eval.content).group(1))
        qwk_score = calculate_qwk([first_score], [second_score])
    except:
        qwk_score = None

    return {
        'detailed_feedback': response.content,
        'qwk_score': qwk_score,
        'weights': weights
    }

# Add these new functions after the existing ones
def detect_language(text):
    """Detects the language of the given text."""
    try:
        return detect(text)
    except:
        return "en"  # default to English if detection fails

def translate_text(text, target_language="en"):
    """Translates text to the target language."""
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def detect_ai_content(text):
    """Detects if the text contains AI-generated content."""
    ai_detector = pipeline("text-classification", model="Hello-SimpleAI/chatgpt-detector-roberta")
    result = ai_detector(text)[0]
    return {
        'is_ai_generated': result['label'] == 'AI',
        'confidence': result['score']
    }

def multilingual_evaluation(question, answer, texts, index):
    """Handles multilingual evaluation by detecting and translating if needed."""
    # Detect languages
    question_lang = detect_language(question)
    answer_lang = detect_language(answer)

    translations = {}
    original_question = question
    original_answer = answer

    # Translate to English if needed (keeping original for feedback)
    if question_lang != "en":
        question = translate_text(question)
        translations['question'] = {
            'original_language': question_lang,
            'original_text': original_question,
            'translated_text': question
        }

    if answer_lang != "en":
        answer = translate_text(answer)
        translations['answer'] = {
            'original_language': answer_lang,
            'original_text': original_answer,
            'translated_text': answer
        }

    # Perform standard evaluation
    evaluation_result = integrated_evaluation(question, answer, texts, index)

    # Add AI detection
    ai_detection = detect_ai_content(original_answer)

    return {
        **evaluation_result,
        'translations': translations if translations else None,
        'ai_detection': ai_detection
    }

# Sample Question and Answer
# Question: Explain the concept of Policy Gradient methods in Reinforcement Learning and discuss how they differ from Q-learning approaches. Include their advantages and potential challenges in practical applications.

# Student's answer: Policy Gradient methods are a class of reinforcement learning algorithms that directly optimize the policy function instead of learning value
# functions. Unlike Q-learning which learns action-values, policy gradients learn the policy π(a|s) directly by following the gradient of expected return.
# The main advantage of policy gradients is their ability to handle continuous action spaces naturally. For example, in robotics control, where actions might be continuous
# joint movements, policy gradients can output smooth control signals. They also can learn stochastic policies, which is useful for exploration.
# The basic idea is to use gradient ascent to maximize the expected reward: J(θ) = E[R|π_θ] where θ represents the policy parameters. Using the REINFORCE algorithm,
# we can estimate this gradient using episode samples.  However, policy gradients face challenges like high variance in gradient estimates, which makes training unstable.
# This is usually addressed using baseline functions or actor-critic methods that combine both policy and value learning.
# Some practical applications include: - Robot manipulation tasks - Game playing agents - Resource allocation problems
# The training process can be slower than Q-learning methods and requires careful hyperparameter tuning.
# Additionally, policy gradients might converge to local optima rather than global ones.

# Modify the main function as follows:
if __name__ == "__main__":
    # Process PDF
    pdf_file = "RL_Notes.pdf"
    print("Processing PDF...")
    texts, faiss_index = process_lecture_pdf(pdf_file)
    print("PDF processed and FAISS index created.")

    # Get inputs
    question = input("Enter the question: ")
    student_answer = input("Enter the student's answer: ")

    # Generate evaluation with multilingual and AI detection support
    result = multilingual_evaluation(question, student_answer, texts, faiss_index)

    print("\n=== Evaluation Results ===")

    # Show translation info if applicable
    if result['translations']:
        print("\n=== Translation Information ===")
        if 'question' in result['translations']:
            trans = result['translations']['question']
            print(f"\nQuestion was translated from {trans['original_language']}:")
            print(f"Original: {trans['original_text']}")
            print(f"Translated: {trans['translated_text']}")

        if 'answer' in result['translations']:
            trans = result['translations']['answer']
            print(f"\nAnswer was translated from {trans['original_language']}:")
            print(f"Original: {trans['original_text']}")
            print(f"Translated: {trans['translated_text']}")

    # Show AI detection results
    print("\n=== AI Detection ===")
    ai_result = result['ai_detection']
    print(f"AI-generated content detected: {'Yes' if ai_result['is_ai_generated'] else 'No'}")
    print(f"Confidence: {ai_result['confidence']:.2%}")

    # Show original evaluation results
    print("\n=== Evaluation Feedback ===")
    print("\nDetailed Feedback:")
    print(result['detailed_feedback'])
    print(f"\nQWK Score: {result['qwk_score']}")
    print("\nWeights Used:")
    for criterion, weight in result['weights'].items():
        print(f"- {criterion}: {weight}%")
