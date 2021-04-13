
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
