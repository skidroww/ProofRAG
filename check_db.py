import pickle

with open("./corpus_info_kiwi.pkl", 'rb') as f:
    corpus_info = pickle.load(f)

doc_info = {}

for item in corpus_info:
    doc_id = item['chunk_id'].split('_chunk_')[0]

    title = item["metadata"]["title"]
    company = item["metadata"]["company"]

    doc_info[doc_id] = {
        "title": title,
        "company": company
    }

print("현재 DB 문서 목록:")
#for i, (doc_id, info) in enumerate(doc_info.items(), 1):
#    print(f"{i}. [{info['company']}] {doc_id} | {info['title']}")

seen = set()

for item in corpus_info:
    m = item["metadata"]
    key = (m["title"], m["company"], m["violation"])

    if key not in seen:
        seen.add(key)
        print({
            "title": m["title"],
            "company": m["company"],
            "violation": m["violation"]
        })


for item in corpus_info:
    m = item["metadata"]

    print({
        "chunk_id": item["chunk_id"],
        "title": m["title"],
        "company": m["company"],
        "violation": m["violation"]
    })


print(len(corpus_info))