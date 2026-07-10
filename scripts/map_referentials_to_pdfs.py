#!/usr/bin/env python3
"""
Script pour mapper les référentiels importés aux PDF originaux.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional


def find_pdf_for_referential(referential_id: str, base_dir: Path = None) -> Optional[str]:
    """
    Trouve le fichier PDF correspondant à un référentiel importé.
    
    Args:
        referential_id: ID du référentiel (document_id)
        base_dir: Répertoire de base du projet
        
    Returns:
        Nom du fichier PDF ou None si non trouvé
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Chemins
    imported_dir = base_dir / 'data' / 'referentials' / 'imported'
    pdf_dir = base_dir / 'data' / 'raw' / 'referentiel'
    
    if not imported_dir.exists() or not pdf_dir.exists():
        return None
    
    # Chercher le JSON correspondant
    for json_file in imported_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_id = data.get('document', {}).get('id', '')
            if doc_id == referential_id:
                # Trouver le PDF par hash
                sha256 = data.get('document', {}).get('sha256', '')
                if sha256:
                    for pdf_path in pdf_dir.glob('*.pdf'):
                        with open(pdf_path, 'rb') as pf:
                            pdf_hash = hashlib.sha256(pf.read()).hexdigest()
                        if pdf_hash == sha256:
                            return pdf_path.name
        except Exception as e:
            print(f"Erreur lors de la lecture de {json_file}: {e}")
            continue
    
    return None


def list_all_mappings(base_dir: Path = None) -> Dict[str, Optional[str]]:
    """
    Liste tous les mappings entre référentiels importés et PDF originaux.
    
    Returns:
        Dictionnaire {referential_id: pdf_filename}
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    imported_dir = base_dir / 'data' / 'referentials' / 'imported'
    pdf_dir = base_dir / 'data' / 'raw' / 'referentiel'
    
    mappings = {}
    
    if not imported_dir.exists():
        return mappings
    
    # Calculer les hashes de tous les PDF
    pdf_hashes = {}
    if pdf_dir.exists():
        for pdf_path in pdf_dir.glob('*.pdf'):
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_hash = hashlib.sha256(f.read()).hexdigest()
                pdf_hashes[pdf_hash] = pdf_path.name
            except Exception as e:
                print(f"Erreur lors du calcul du hash de {pdf_path}: {e}")
    
    # Mapper chaque JSON
    for json_file in imported_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc_id = data.get('document', {}).get('id', '')
            sha256 = data.get('document', {}).get('sha256', '')
            
            if doc_id:
                pdf_name = pdf_hashes.get(sha256)
                mappings[doc_id] = pdf_name
        except Exception as e:
            print(f"Erreur lors de la lecture de {json_file}: {e}")
    
    return mappings


if __name__ == '__main__':
    print("Mapping des référentiels importés aux PDF originaux:")
    print("=" * 60)
    
    mappings = list_all_mappings()
    
    for ref_id, pdf_name in mappings.items():
        status = "✅" if pdf_name else "❌"
        print(f"{status} {ref_id[:20]}... -> {pdf_name or 'Non trouvé'}")
    
    print(f"\nTotal: {len(mappings)} référentiels")
    print(f"Trouvés: {sum(1 for v in mappings.values() if v)}")
    print(f"Non trouvés: {sum(1 for v in mappings.values() if not v)}")
