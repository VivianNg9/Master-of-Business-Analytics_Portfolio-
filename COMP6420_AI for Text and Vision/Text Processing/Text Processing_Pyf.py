import pandas as pd 
import numpy as np
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag_sents
from collections import Counter 
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt') # For tokenization 
nltk.download('averaged_perceptron_tagger') #For POS tagging
nltk.download('universal_tagset') # For the Universal POS tags
import spacy 
import os
os.system("python -m spacy download en_core_web_sm")
import en_core_web_sm

# Task 1 (3 marks)
# Comparing POS distributions between questions and answers:
# - Similarities:
#   - Both have quite similar distributions for NOUN and '.'
#   - The distribution of DET (determiner) is also quite similar in both cases.
# 
# - Differences:
#   - VERB is higher in questions (0.1659) than in answers (0.1112).
#     This aligns with questions typically involving actions or inquiries.
#   - ADJ (adjective) is higher in answers (0.1204) than in questions (0.0892),
#     suggesting that answers may contain more descriptive information.
#   - ADV (adverb) is higher in answers (0.0245) than in questions (0.011),
#     indicating that answers may use adverbs for more context or modification.
#
# Overall, while there are similarities in certain tags, the differences 
# highlight the distinct functions of questions and answers.
def stats_pos(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract unique questions and concatenate them into a single string
    questions_text = ' '.join(df['question'].unique())
    # Concatenate all answer texts into a single string
    answers_text = " ".join(df['sentence text'])

    # Tokenize the concatenated text into sentences and then into words
    questions_sents = [word_tokenize(sent) for sent in sent_tokenize(questions_text)]
    answers_sents = [word_tokenize(sent) for sent in sent_tokenize(answers_text)]

    # Tag POS using NLTK's pos_tag_sents with the universal tag set
    questions_tags = [tag for sent in pos_tag_sents(questions_sents, tagset='universal') for _, tag in sent]
    answers_tags = [tag for sent in pos_tag_sents(answers_sents, tagset='universal') for _, tag in sent]

    # Calculate the total count of PoS tags for questions and answers
    total_question_tags = len(questions_tags)
    total_answer_tags = len(answers_tags)

    # Count the frequency of each PoS tag using Counter
    questions_freq = Counter(questions_tags)
    answers_freq = Counter(answers_tags)

    # Normalize and sort the frequencies for questions
    questions_normalized = [(tag, round(count / total_question_tags, 4)) 
                            for tag, count in sorted(questions_freq.items())]
    # Normalize and sort the frequencies for answers
    answers_normalized = [(tag, round(count / total_answer_tags, 4)) 
                          for tag, count in sorted(answers_freq.items())]

    return questions_normalized, answers_normalized


# Task 2 (3 marks)
# Set n=2, N=5, comparing overlap distributions between questions and answers:
# - Overlaps:
#   - Both questions and answers include the bigrams ('of', 'the') and ('in', 'the')
#     among their most frequent. This indicates a shared focus on specific 
#     entities or locations.
# - Differences:
#   - The bigram ("what","is") is common in questions, reflecting an inquiry-based approach.
#   - Answers feature the bigram (',', 'and') and ((')', ',') which indicate a tendency 
#     towards more descriptive or explanatory content.
# 
# Overall, while there's some overlap in common phrases, the differences 
# highlight the distinct purposes of questions and answers.

def stats_top_stem_ngrams(csv_file_path, n, N):
    # Load the dataset
    df = pd.read_csv(csv_file_path)
    
    # Extract unique questions and concatenate them into a single string
    questions_text = ' '.join(df['question'].unique())
    # Concatenate all answer texts into a single string
    answers_text = " ".join(df['sentence text'])

    # Define a Porter stemmer from the NLTK package for stemming
    stemmer = nltk.PorterStemmer()
    def tokenize_and_stem(text):
        # Tokenize the text into sentences using NLTK
        sentences = sent_tokenize(text)
        stemmed_sentences = []
        for sentence in sentences:
            # Tokenize each sentence into words
            words = word_tokenize(sentence)
            # Stem each word and collect the result
            stemmed_words = [stemmer.stem(word) for word in words]
            stemmed_sentences.append(stemmed_words)
        return stemmed_sentences

    # Apply tokenization and stemming to both questions and answers
    questions_stem = tokenize_and_stem(questions_text)
    answers_stem = tokenize_and_stem(answers_text)
    
    def get_ngrams(stemmed_sentences, n):
        all_ngrams = []
        for sentence in stemmed_sentences:
            # Generate n-grams of specified length 'n' if the sentence is long enough
            sentence_ngrams = list(ngrams(sentence, n)) if len(sentence) >= n else []
            all_ngrams.extend(sentence_ngrams)
        return all_ngrams

    # Generate n-grams for questions and answers
    questions_ngrams = get_ngrams(questions_stem, n)
    answers_ngrams = get_ngrams(answers_stem, n)
    
    def calculate_normalized_frequencies(ngrams_list):
        # Compute frequency distribution of n-grams using NLTK's FreqDist
        freq_dist = FreqDist(ngrams_list)
        total = sum(freq_dist.values())  # Calculate total occurrences
        # Normalize frequencies and sort by frequency in descending order
        normalized_frequencies = [(ngram, round(freq / total, 4)) for ngram, freq in freq_dist.items()]
        # Return the top N n-grams by frequency
        return sorted(normalized_frequencies, key=lambda x: x[1], reverse=True)[:N]

    # Calculate normalized frequencies for questions and answers, keeping the top N
    questions_freq_stem = calculate_normalized_frequencies(questions_ngrams)
    answers_freq_stem = calculate_normalized_frequencies(answers_ngrams)
    
    return questions_freq_stem, answers_freq_stem


# Task 3 (2 marks)
# Named entity statistics for questions and answers:
# - Commonality:
#   - Both questions and answers heavily feature 'ORG' (organization) entities,
#     with 'PERSON' also being prominent. This aligns with common topics in
#     biomedical texts, where organizations and individuals play key roles.
#
# - Differences:
#   - 'CARDINAL' entities (numeric values) are much more frequent in answers
#     (0.1887) compared to questions (0.0966), suggesting that answers
#     often contain more precise numeric information.
#   - 'GPE' (geopolitical entities) appear more in questions (0.1172) than
#     in answers (0.0564), indicating that questions may focus more on
#     geographical or political topics.
#   - 'DATE' entities are more common in questions (0.0207) compared to answers
#     (0.013), possibly indicating a focus on specific time frames in questions.
#   - Conversely, 'ORDINAL' entities, indicating positions in a series, are
#     more frequent in answers (0.026) than in questions (0.0115).

def stats_ne(csv_file_path):
    # Load the en_core_web_sm 
    nlp = en_core_web_sm.load()
    
    # Load the dataset
    df = pd.read_csv(csv_file_path)

    # Extract unique questions and concatenate them into a single string
    questions_text = " ".join(df['question'].unique())
    # Concatenate all answer texts into a single string
    answers_text = " ".join(df['sentence text'])

    # Ensure both questions and answers have the same text length
    min_length = min(len(questions_text), len(answers_text))
    questions_text = questions_text[:min_length]
    answers_text = answers_text[:min_length]

    # Process the questions text with spaCy to extract named entities
    questions_doc = nlp(questions_text)
    questions_entities = [ent.label_ for ent in questions_doc.ents]

    # Process the answers text with spaCy to extract named entities
    answers_doc = nlp(answers_text)
    answers_entities = [ent.label_ for ent in answers_doc.ents]

    # Count the named entity types for questions and answers
    questions_ne_counts = Counter(questions_entities)
    answers_ne_counts = Counter(answers_entities)

    # Calculate the total counts for normalization
    total_questions_ents = sum(questions_ne_counts.values())
    total_answers_ents = sum(answers_ne_counts.values())

    # Normalize and sort the frequencies for questions
    questions_freqs = [(ent, round(count / total_questions_ents, 4)) 
                       for ent, count in sorted(questions_ne_counts.items())]
    # Normalize and sort the frequencies for answers
    answers_freqs = [(ent, round(count / total_answers_ents, 4)) 
                     for ent, count in sorted(answers_ne_counts.items())]

    # Return the normalized frequencies as tuples for questions and answers
    return questions_freqs, answers_freqs


# Task 4 (2 marks)
# - The result indicates that for approximately 48.76% of the questions,
#   the sentence with the highest cosine similarity based on tf-idf falls
#   within the corresponding answers. This suggests that nearly half of
#   the questions have at least one sentence that closely aligns with
#   their intended answers.
# - The outcome highlights the challenge of achieving perfect alignment
#   between questions and their correct answers using tf-idf similarity,
#   potentially due to variations in wording or focus between them.
# - The use of `TfidfVectorizer` with `stop_words='english'` helped in
#   reducing noise from common words, focusing the similarity
#   calculations on more meaningful terms.
def stats_tfidf(csv_file_path):
    # Load the dataset
    df = pd.read_csv(csv_file_path)
    
    # Extracting questions and their corresponding answers
    unique_questions = df['question'].unique().tolist()
    sentences_df = df[['qid', 'sentid', 'sentence text', 'label']]
    sentences = df['sentence text'].tolist()
    
    # Combine all unique questions and answers 
    all_texts = unique_questions + sentences

    # Vectorize the text using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix into questions and sentences parts
    questions_tfidf = tfidf_matrix[:len(unique_questions)]
    sentences_tfidf = tfidf_matrix[len(unique_questions):]
    
    correct_count = 0
    for i, question_tfidf in enumerate(questions_tfidf):
        # Calculate cosine similarity between each question and all sentences
        cosine_similarities = cosine_similarity(question_tfidf, sentences_tfidf)
        # Find the index of the highest similarity sentence for each question
        most_similar_index = cosine_similarities.argmax()
        # Retrieve the most similar sentence
        most_similar_sentence = sentences_df.iloc[most_similar_index]
        # Retrieve the question ID for the current question
        question_id = df[df['question'] == unique_questions[i]].iloc[0]['qid']
        # Check if the most similar sentence is a correct answer
        if (most_similar_sentence['qid'] == question_id and
            most_similar_sentence['label'] == 1):
            correct_count += 1
            
    # Calculate the ratio of questions where the most similar sentence is correct
    ratio = round(correct_count / len(unique_questions), 4)
    return ratio


# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    print("---------Task 1---------------")
    print(stats_pos('data/dev_test.csv'))
  
    print("---------Task 2---------------")
    print(stats_top_stem_ngrams('data/dev_test.csv', 2, 5))

    print("---------Task 3---------------")
    print(stats_ne('data/dev_test.csv'))

    print("---------Task 4---------------")
    print(stats_tfidf('data/dev_test.csv'))
  
