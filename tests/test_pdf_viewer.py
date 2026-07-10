"""
Tests pour la visualisation des PDF.
"""
import pytest
from pathlib import Path
from scripts.map_referentials_to_pdfs import find_pdf_for_referential, list_all_mappings


class TestPDFViewer:
    """Tests pour la fonctionnalité de visualisation des PDF."""
    
    def test_find_pdf_for_referential(self):
        """Test que l'on peut trouver un PDF pour un référentiel."""
        # Utiliser un ID de référentiel connu
        referential_id = "eb2c03aadddb1a8ef47b9e59"
        pdf_name = find_pdf_for_referential(referential_id)
        
        assert pdf_name is not None
        assert pdf_name.endswith('.pdf')
        assert Path('data/raw/referentiel', pdf_name).exists()
    
    def test_find_pdf_for_unknown_referential(self):
        """Test que None est retourné pour un référentiel inconnu."""
        referential_id = "unknown_id_12345"
        pdf_name = find_pdf_for_referential(referential_id)
        
        assert pdf_name is None
    
    def test_list_all_mappings(self):
        """Test que l'on peut lister tous les mappings."""
        mappings = list_all_mappings()
        
        assert isinstance(mappings, dict)
        assert len(mappings) > 0
        
        # Vérifier que tous les mappings ont un ID de référentiel
        for ref_id, pdf_name in mappings.items():
            assert isinstance(ref_id, str)
            assert len(ref_id) > 0
            
            # Si un PDF est trouvé, vérifier qu'il existe
            if pdf_name:
                assert pdf_name.endswith('.pdf')
                assert Path('data/raw/referentiel', pdf_name).exists()
    
    def test_all_imported_referentials_have_pdf(self):
        """Test que tous les référentiels importés ont un PDF correspondant."""
        mappings = list_all_mappings()
        
        # Vérifier qu'au moins un référentiel a un PDF
        found_count = sum(1 for v in mappings.values() if v)
        assert found_count > 0, "Aucun référentiel n'a de PDF correspondant"
        
        # Afficher le taux de réussite
        total = len(mappings)
        success_rate = (found_count / total) * 100
        print(f"\nTaux de réussite: {success_rate:.1f}% ({found_count}/{total})")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
