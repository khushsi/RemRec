import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as pairwise
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from lib.nlp import processed, lower, split
from lib.evaluate import evaluate_file
import warnings
warnings.simplefilter(action='ignore', category=Warning)

stemming_text=True
stemming_concept=False
binary_vector=False
is_tfidf=True

if is_tfidf:
    binary_vector=False

def get_match(vec_file, cfile):


    #Load Concepts
    clist = pd.read_csv(cfile)
    sec_or_q_to_concepts = clist.groupby('item_id')['concept'].apply(set).to_dict()

    # Load Book Sections and text
    df_book = pd.read_csv('data/section2text.csv', encoding='utf8')

    book_ids = df_book['section'].tolist()
    book_texts = df_book['text'].apply(lower).tolist()

    # Quiz Text
    df_quiz = pd.read_csv('data/quiz_text_section.csv').fillna('')
    quiz_ids = df_quiz['quizid'].tolist()
    quiz_texts = df_quiz['text'].apply(lower).tolist()


    book_texts = [processed(i, stemming=stemming_text) for i in book_texts]
    quiz_texts = [processed(i, stemming=stemming_text) for i in quiz_texts]


    concept_all_list = list(set([j for i in sec_or_q_to_concepts for j in sec_or_q_to_concepts[i]]))


    # Text Representation BoW approach
    if is_tfidf:
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()

    all_texts = book_texts + quiz_texts

    vectorizer.fit_transform(all_texts)

    Q_vector_content = vectorizer.transform(quiz_texts).toarray()
    B_vector_content = vectorizer.transform(book_texts).toarray()

    # Concept Representation Binary Vector Approach
    B_vector_concept = [
        [1 if i in sec_or_q_to_concepts[s] else 0 for i in concept_all_list] for s in book_ids
    ]

    Q_vector_concept = [
        [1 if i in sec_or_q_to_concepts[s] else 0 for i in concept_all_list] for s in quiz_ids]

    dict_Q_vector_content = dict(zip(quiz_ids, Q_vector_content))
    dict_Q_vector_concept = dict(zip(quiz_ids, Q_vector_concept))

    df = pd.read_csv(vec_file)

    features = df.columns[6:].tolist()

    features = [processed(i,stemming=stemming_concept) for i in features]

    df_q_0 = df[(df['quiz_id'].str.startswith('q')) & (df['outcome'].isin(('0', 0)))]

    matrix_content = []
    matrix_concept = []
    matrix_knowledge = []

    r = []
    for row in df_q_0.values:
        interid, type_, quiz_id, concepts, student_id, outcome, quiz_sec = row[:7]

        content_vector = dict_Q_vector_content[quiz_id]
        concept_vector = dict_Q_vector_concept[quiz_id]
        concept_vector_knowledge = np.array(concept_vector, dtype=np.float64)

        knowledge_v = row[7:]
        concepts = sec_or_q_to_concepts[quiz_id]
        for k in concepts:
            if k in features:

                i = features.index(k)
            else:
                i=-1
                # print(k,"not in list")
            if i >= 0:
                v = knowledge_v[i]
                if 0. <= v <= 1.0:
                    index = concept_all_list.index(k)
                    concept_vector_knowledge[index] = 1-v

        matrix_content.append(content_vector)
        matrix_concept.append(concept_vector)
        matrix_knowledge.append(concept_vector_knowledge)


    sim_matrix_content = pairwise.cosine_similarity(matrix_content, B_vector_content)
    sim_matrix_concept = pairwise.cosine_similarity(matrix_concept, B_vector_concept)
    sim_matrix_knowledge = pairwise.cosine_similarity(matrix_knowledge, B_vector_concept)


    # find matches
    def get_match_top_n(sim, n=5):
        best = []
        for row in sim:
            tmp = []
            for idx in reversed(row.argsort()[-n:]):
                if row[idx] > 0.0:
                    tmp.append(book_ids[idx])

            best.append(tmp)
        return best

    df_match = pd.DataFrame()
    df_match['interaction_id'] = df_q_0['interaction_id']
    df_match['content_match'] = get_match_top_n(sim_matrix_content)
    df_match['concept_match'] = get_match_top_n(sim_matrix_concept)
    df_match['knowledge_match'] = get_match_top_n(sim_matrix_knowledge)

    alpha=0.6
    sim = alpha * sim_matrix_knowledge + (1-alpha) * sim_matrix_content
    df_match['sim_com_knowledge_norm_concat_{:.2f}_match'.format(alpha)] = get_match_top_n(sim)

    sim = alpha * sim_matrix_knowledge + (1-alpha) * sim_matrix_concept
    df_match['sim_com_knowledge_norm_concept_{:.2f}_match'.format(alpha)] = get_match_top_n(sim)

    df = df[['interaction_id', 'quiz_id', 'outcome', 'student_id', 'quiz_section']]
    df = df.merge(df_match, on='interaction_id', how='left')

    return df


if __name__ == '__main__':

    vec_file= 'data/vec/pfa_outcome_concept_list_expert.csv'
    concept_file='data/concept_files/list_filter_stemmed_TopicRank_concepts.txt'
    df = get_match(vec_file,concept_file)
    df.to_csv("match.csv", index=False)

    print('matching file-created',"data/match.csv")
    evaluate_file('match.csv')

