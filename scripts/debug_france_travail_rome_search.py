
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from france_travail.client import FranceTravailClient, SearchCriteria, count_returned_rome_codes, normalize_rome_code
from france_travail.normalizer import normalize_offer
from services.market_context import filter_offers_by_exact_rome


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Diagnostic de recherche France Travail par code ROME.')
    parser.add_argument("--rome", required=True, help="Code ROME confirmé par l'utilisateur.")
    parser.add_argument('--departement', required=True, help='Code département ou commune.')
    parser.add_argument('--commune', default=None, help='Code commune optionnel.')
    parser.add_argument("--limit", type=int, default=50, help="Nombre max d'offres à collecter.")
    parser.add_argument('--save-raw', default=None, help='Chemin JSON de sauvegarde des résultats bruts.')
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    requested = normalize_rome_code(args.rome)
    client = FranceTravailClient()
    criteria = SearchCriteria(rome_code=requested, departement=args.departement, commune=args.commune, size=min(args.limit, 20))
    LOGGER.info('Paramètres envoyés: %s', criteria.to_params())
    result = client.search_offers(criteria)
    normalized = [normalize_offer(offer).to_dict() for offer in result.offers]
    accepted, rejected = filter_offers_by_exact_rome(normalized, requested)
    distribution = count_returned_rome_codes(result.offers)

    print(json.dumps({
        'requested_rome': requested,
        'url': result.raw.get('url') if isinstance(result.raw, dict) else None,
        'status': result.status_code,
        'raw_count': len(result.offers),
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        'distribution': distribution,
        'titles': [offer.get('title') for offer in normalized],
        'rome_codes': [offer.get('rome_code') for offer in normalized],
    }, ensure_ascii=False, indent=2))

    if args.save_raw:
        payload = {
            'query': {
                'rome_code': requested,
                'departement': args.departement,
                'commune': args.commune,
            },
            'raw_count': len(result.offers),
            'accepted_count': len(accepted),
            'rejected_count': len(rejected),
            'offers': normalized,
            'rejections': rejected,
        }
        output = Path(args.save_raw)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        LOGGER.info('Résultats bruts sauvegardés dans %s', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
