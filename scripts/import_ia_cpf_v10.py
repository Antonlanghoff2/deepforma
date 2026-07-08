import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from data.cpf_ia_v10 import inspect_excel, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('import_ia_cpf_v10')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Importer le dataset CPF IA v10',
    )
    p.add_argument(
        '--input',
        default='data/raw/Dataset_IA_V10_CPF.xlsx',
        help='Chemin vers le fichier Excel Dataset_IA_V10_CPF.xlsx',
    )
    p.add_argument(
        '--output-dir',
        default='data/processed/ia_cpf_v10',
        help='Répertoire de sortie pour les fichiers traités',
    )
    p.add_argument(
        '--inspect-only',
        action='store_true',
        help='Inspecter le fichier sans lancer le pipeline complet',
    )
    p.add_argument(
        '--inspect-output',
        default=None,
        help='Fichier JSON de sortie pour le rapport d\'inspection',
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error('Fichier introuvable: %s', input_path)
        sys.exit(1)

    if args.inspect_only:
        logger.info('Mode inspection uniquement')
        inspection = inspect_excel(input_path)
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        if args.inspect_output:
            out_path = Path(args.inspect_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(inspection, f, ensure_ascii=False, indent=2)
            logger.info('Rapport d\'inspection écrit dans %s', out_path)
        return

    report = run_pipeline(
        input_path=input_path,
        output_dir=args.output_dir,
    )
    report_dict = {
        'rows_total': report.rows_total,
        'rncp_count': report.rncp_count,
        'rs_count': report.rs_count,
        'missing_rome_count': report.missing_rome_count,
        'missing_level_count': report.missing_level_count,
        'missing_duration_count': report.missing_duration_count,
        'duplicate_certification_codes': report.duplicate_certification_codes,
        'rows_to_review': report.rows_to_review,
        'skills_total': report.skills_total,
        'skills_to_review': report.skills_to_review,
        'sectors': report.sectors,
        'generated_at': report.generated_at,
    }
    print(json.dumps(report_dict, ensure_ascii=False, indent=2))
    logger.info('Pipeline terminé avec succès.')


if __name__ == '__main__':
    main()
