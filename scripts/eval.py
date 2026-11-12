import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from rag import answer_question
from db import db_connection

def calculate_recall(test, passages):
    if test["source"] is None:
        return {
            "rang_preuve": None,
            "recall@1": None,
            "recall@3": None,
            "recall@5": None,
        }

    preuve = test.get("preuve_attendue")
    if not preuve or not preuve.strip():
        raise ValueError(f"Preuve attendue manquante pour {test['id']}")

    # Ignore les différences de casse et d'espacement.
    preuve = " ".join(preuve.split()).casefold()
    rang_preuve = None

    for rang, passage in enumerate(passages, start=1):
        texte = " ".join(passage["text"].split()).casefold()

        if passage["url"] == test["source"] and preuve in texte:
            rang_preuve = rang
            break

    return {
        "rang_preuve": rang_preuve,
        "recall@1": int(rang_preuve is not None and rang_preuve <= 1),
        "recall@3": int(rang_preuve is not None and rang_preuve <= 3),
        "recall@5": int(rang_preuve is not None and rang_preuve <= 5),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Évaluer le RAG sur un corpus.")
    parser.add_argument("corpus", choices=["rag", "python"], nargs="?", default="rag")
    parser.add_argument("--limit", type=int, help="Nombre de questions à exécuter pour un essai.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit doit être supérieur ou égal à 1.")
    return args


def load_tests(test_path, limit=None):
    with test_path.open(encoding="utf-8") as fichier:
        tests = [
            json.loads(ligne)
            for ligne in fichier
            if ligne.strip()
        ]

    if limit is not None:
        tests = tests[:limit]

    # Vérifier les annotations avant les appels coûteux à Ollama.
    for test in tests:
        if test["source"] is not None and not test.get("preuve_attendue", "").strip():
            raise ValueError(f"Preuve attendue manquante pour {test['id']}")

    return tests


def check_corpus(connection, corpus):
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE corpus = %s", (corpus,))
        nombre_chunks = cursor.fetchone()[0]

    if nombre_chunks == 0:
        raise ValueError(f"Aucun chunk dans le corpus {corpus}. Lance index.py {corpus}.")


def evaluate_question(connection, test, corpus):
    debut = perf_counter()
    answer, passages, tokens = answer_question(
        question=test["question"],
        connection=connection,
        corpus=corpus,
    )
    duree = perf_counter() - debut

    resultat = {
        "id": test["id"],
        "corpus": corpus,
        "question": test["question"],
        "reponse_attendue": test["reponse_attendue"],
        "reponse_obtenue": answer,
        "source_attendue": test["source"],
        "passages_recuperes": passages,
        "duree_secondes": round(duree, 2),
        "réponse_correcte": None,
        "citation_correcte": None,
        "commentaire": "",
        "tokens_entree": tokens["tokens_entree"],
        "tokens_sortie": tokens["tokens_sortie"],
        "tokens_total": tokens["tokens_total"],
    }
    resultat.update(calculate_recall(test, passages))
    return resultat


def evaluate_tests(connection, tests, corpus, sortie):
    with sortie.open("x", encoding="utf-8") as fichier:
        print(f"Résultats enregistrés dans {sortie}", flush=True)

        for numero, test in enumerate(tests, start=1):
            print(
                f"\n[{numero}/{len(tests)}] {test['id']} : {test['question']}",
                flush=True,
            )
            resultat = evaluate_question(connection, test, corpus)
            fichier.write(json.dumps(resultat, ensure_ascii=False) + "\n")
            fichier.flush()
            print(resultat["reponse_obtenue"], flush=True)


def main():
    args = parse_arguments()
    dossier = Path(__file__).resolve().parent
    test_path = dossier.parent / "tests" / args.corpus / "test.jsonl"
    tests = load_tests(test_path, args.limit)

    date_run = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dossier_resultats = dossier.parent / "tests" / args.corpus
    dossier_resultats.mkdir(parents=True, exist_ok=True)
    sortie = dossier_resultats / f"resultats-{args.corpus}-{date_run}.jsonl"

    with db_connection() as connection:
        check_corpus(connection, args.corpus)
        evaluate_tests(connection, tests, args.corpus, sortie)


if __name__ == "__main__":
    main()
