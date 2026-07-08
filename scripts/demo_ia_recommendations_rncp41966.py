#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from data_sources.ia_recommendations import load_ia_recommendations_csv
from domain.ia_recommendation_matching import match_ia_recommendations


RNCP41966_SKILLS = [
    "Animer la relation de partenariat avec les acteurs de la filière",
    "Appliquer les dispositions juridiques et réglementaires du contrat de vente",
    "Assurer la mise en oeuvre de la stratégie de développement commercial",
    "Concevoir et mettre en oeuvre la stratégie marketing et commerciale",
    "Conduire les négociations commerciales avec les clients et les partenaires",
    "Déployer et animer le réseau de vente",
    "Développer la relation client et la relation partenariale",
    "Élaborer et piloter le plan d'actions commerciales",
    "Évaluer et optimiser la performance commerciale",
    "Exploiter les études de marché et les données commerciales",
    "Gérer les équipes commerciales",
    "Manager l'activité commerciale",
    "Mettre en oeuvre un système d'information commercial",
    "Négocier les contrats de vente et de partenariat",
    "Organiser et planifier l'activité de la force de vente",
    "Piloter le déploiement de la stratégie marketing et commerciale",
    "Réaliser des études de marché",
    "Recruter et intégrer les équipes commerciales",
    "Mettre en place la veille informationnelle",
    "Analyser les besoins clients et adapter l'offre",
    "Gérer le portefeuille clients grands comptes",
    "Conduire des entretiens de vente",
    "Utiliser un CRM pour le suivi de l'activité commerciale",
    "Élaborer le reporting commercial",
    "Animer des réunions commerciales",
    "Former les équipes commerciales aux techniques de vente",
    "Définir les objectifs commerciaux",
    "Segmenter le portefeuille clients",
    "Mettre en place des actions de fidélisation",
    "Développer un réseau de partenaires",
    "Gérer les litiges clients",
    "Suivre la satisfaction client",
    "Élaborer des offres commerciales",
    "Négocier avec les fournisseurs et prestataires",
]


def main() -> None:
    csv_path = Path('data/raw/recommandations_IA_consolide.csv')
    if not csv_path.exists():
        print(f'ERREUR: {csv_path} introuvable')
        sys.exit(1)

    records, report = load_ia_recommendations_csv(csv_path)
    if report.valid_lines == 0:
        print('ERREUR: aucune recommandation chargee')
        sys.exit(1)

    skills = [{'name': s, 'normalized_name': None} for s in RNCP41966_SKILLS]
    matches = match_ia_recommendations(skills, records, embedding_model=None)

    print(f'=== Demo IA Recommandations - RNCP41966 ===')
    print(f'Competences du referentiel:  {len(skills)}')
    print(f'Recommandations chargees:    {report.valid_lines}')
    print(f'Matchs trouves:              {len(matches)}')

    method_counts = Counter(m.match_method for m in matches)
    conf_counts = Counter(m.confidence_label for m in matches)
    print(f'\nPar methode de matching:')
    for method, count in method_counts.most_common():
        print(f'  {method:20s}: {count}')
    print(f'\nPar niveau de confiance:')
    for conf, count in conf_counts.most_common():
        print(f'  {conf:20s}: {count}')

    unique_skills = set(m.skill_original for m in matches)
    print(f'\nCompetences avec au moins 1 recommandation: {len(unique_skills)} / {len(skills)}')

    print(f'\n=== Exemples de recommandations ===')
    displayed_skills: set[str] = set()
    for m in matches:
        if m.skill_original not in displayed_skills and len(displayed_skills) < 10:
            displayed_skills.add(m.skill_original)
            print(f'\n  Competence:        {m.skill_original[:70]}')
            print(f'  Mot-cle reconnu:   {m.matched_keyword}')
            print(f'  Recommandation:    {m.recommendation[:100]}...')
            print(f'  Methode:           {m.match_method} (score: {m.score})')
            print(f'  Confiance:         {m.confidence_label}')

    if unique_skills:
        print(f'\n=== VERDICT ===')
        print(f'  {len(matches)} recommandations pour {len(unique_skills)} competences')
        print(f'  {method_counts.get("EXACT", 0)} exactes, {method_counts.get("EMBEDDING", 0)} semantiques')
        if conf_counts.get('LOW', 0) > 0 or conf_counts.get('DEFAULT', 0) > 0:
            print(f'  ATTENTION: {conf_counts.get("LOW", 0)} recommandations a verifier')
    else:
        print('\n=== VERDICT ===')
        print('  AUCUNE recommandation trouvee')
        if report.default_rules:
            print('  Appliquer la regle par defaut')


if __name__ == '__main__':
    main()
