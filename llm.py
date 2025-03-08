import pandas as pd
import ollama

qa_results_df = pd.read_csv('qa_pairs_results.csv',encoding='ISO-8859-1')

counter=0
def evaluate_llama(query, top_match_1, cosine_similarity_1, top_match_2, cosine_similarity_2, generated_response,ground_truth,th):
    global counter
    counter+=1
    print(f'Processing Row:- {counter}')
    prompt = f"""
    Question: {query}
    Top Match 1: {top_match_1}
    Cosine Similarity 1: {cosine_similarity_1}
    Top Match 2: {top_match_2}
    Cosine Similarity 2: {cosine_similarity_2}
    Generated Response: {generated_response}
    Ground Truth : {ground_truth}
    Instructions: Compare the generated response with the ground truth. If they match more than {th}%, output 1. Otherwise, output 0. Only return the output (1 or 0) with no explanation at all strictly only output.
    """


    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])


    evaluation_score = response['message']['content']
    return evaluation_score.strip()



def run_llm_as_judge(th):
    qa_results_df[f'Llama Evaluation_{th}%'] = qa_results_df.apply(
        lambda row: evaluate_llama(
            row['Question'], row['Top Match 1'], row['Cosine Similarity 1'],
            row['Top Match 2'], row['Cosine Similarity 2'], row['Generated Response'], row['Ground Truth'],th
        ), axis=1
    )

    qa_results_df.to_csv('qa_pairs_results.csv', index=False)

    return "Q-A pairs with Llama evaluation saved to 'qa_pairs_results.csv'!"
