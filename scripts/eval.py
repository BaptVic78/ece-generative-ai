import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from rag import answer_question
from db import db_connection


def main():
    dossier = Path(__file__).resolve().parent

    test_path = dossier.parent / "tests_rag" / "test.jsonl"
    with test_path.open(encoding="utf-8") as fichier:
        tests = [
            json.loads(ligne)
            for ligne in fichier
            if ligne.strip()
        ]

    date_run = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    sortie = dossier / f"resultats-{date_run}.jsonl"

    with db_connection() as connection:
        with sortie.open("x", encoding="utf-8") as fichier:
            print(f"Résultats enregistrés dans {sortie}", flush=True)

            for numero, test in enumerate(tests, start=1):
                question = test["question"]
                print(
                    f"\n[{numero}/{len(tests)}] {test['id']} : {question}",
                    flush=True,
                )

                debut = perf_counter()
                answer, passages = answer_question(
                    question=question,
                    connection=connection,
                )
                duree = perf_counter() - debut

                resultat = {
                    "id": test["id"],
                    "question": question,
                    "reponse_attendue": test["reponse_attendue"],
                    "reponse_obtenue": answer,
                    "source_attendue": test["source"],
                    "passages_recuperes": passages,
                    "duree_secondes": round(duree, 2),
                    "réponse_correcte": None,
                    "citation_correcte": None,
                    "commentaire": "",
                }

                fichier.write(
                    json.dumps(resultat, ensure_ascii=False) + "\n"
                )
                fichier.flush()
                print(answer, flush=True)


if __name__ == "__main__":
    main()