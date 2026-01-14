from sklearn.metrics.pairwise import cosine_similarity

def verify(e1, e2, threshold=0.6):
    score = float(cosine_similarity([e1], [e2])[0][0])
    return score >= threshold, score
