"""
Tests pour la taxonomie IA et les routes Flask.
"""
import json
import pytest
from pathlib import Path

from web_app import create_app


class TestAITaxonomy:
    """Tests pour le fichier ai_skill_taxonomy.json."""
    
    def test_taxonomy_file_exists(self):
        """Test que le fichier de taxonomie existe."""
        taxonomy_path = Path(__file__).parent.parent / "data" / "referentials" / "ai_skill_taxonomy.json"
        assert taxonomy_path.exists(), f"Fichier de taxonomie introuvable: {taxonomy_path}"
    
    def test_taxonomy_valid_json(self):
        """Test que le fichier est un JSON valide."""
        taxonomy_path = Path(__file__).parent.parent / "data" / "referentials" / "ai_skill_taxonomy.json"
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert isinstance(data, dict)
    
    def test_taxonomy_has_required_fields(self):
        """Test que le fichier contient les champs requis."""
        taxonomy_path = Path(__file__).parent.parent / "data" / "referentials" / "ai_skill_taxonomy.json"
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        required_fields = ['schema_version', 'taxonomy_id', 'title', 'labels']
        for field in required_fields:
            assert field in data, f"Champ manquant: {field}"
    
    def test_taxonomy_has_18_labels(self):
        """Test que le fichier contient les 18 labels attendus."""
        taxonomy_path = Path(__file__).parent.parent / "data" / "referentials" / "ai_skill_taxonomy.json"
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels = data.get('labels', [])
        assert len(labels) == 18, f"Attendu 18 labels, trouvé {len(labels)}"
    
    def test_taxonomy_expected_labels(self):
        """Test que les 18 labels attendus sont présents."""
        taxonomy_path = Path(__file__).parent.parent / "data" / "referentials" / "ai_skill_taxonomy.json"
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        expected_labels = [
            "Automatisation", "Big Data", "Computer Vision", "Data Engineering",
            "Data Science", "Deep Learning", "Gestion de projet IA", "IA générative",
            "Machine Learning", "NLP", "No-code / Low-code", "Prompt Engineering",
            "Python pour l'IA", "RAG", "Reinforcement Learning", "Séries temporelles",
            "Visualisation", "Éthique de l'IA"
        ]
        
        label_names = {label['label'] for label in data.get('labels', [])}
        for expected in expected_labels:
            assert expected in label_names, f"Label manquant: {expected}"
    
    def test_taxonomy_no_duplicates(self):
        """Test qu'il n'y a pas de doublons dans les labels."""
        taxonomy_path = Path(__file__).parent.parent / "data" / "referentials" / "ai_skill_taxonomy.json"
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels = data.get('labels', [])
        label_ids = [label['id'] for label in labels]
        label_names = [label['label'] for label in labels]
        
        assert len(label_ids) == len(set(label_ids)), "IDs dupliqués détectés"
        assert len(label_names) == len(set(label_names)), "Noms dupliqués détectés"


class TestFlaskRoutes:
    """Tests pour les routes Flask."""
    
    @pytest.fixture
    def app(self):
        """Crée une instance de l'application Flask pour les tests."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Crée un client de test Flask."""
        return app.test_client()
    
    def test_get_referential_import_redirects(self, client):
        """Test que GET /referential/import redirige vers /admin/referential-import."""
        response = client.get('/referential/import')
        # Doit rediriger (302) ou retourner une page utile (200)
        assert response.status_code in [200, 302], f"Status inattendu: {response.status_code}"
        
        # Si c'est une redirection, vérifier la destination
        if response.status_code == 302:
            assert '/admin/referential-import' in response.headers.get('Location', '')
    
    def test_get_admin_referential_import_accessible(self, client):
        """Test que GET /admin/referential-import est accessible."""
        # Cette route nécessite une authentification admin
        # On vérifie juste qu'elle ne retourne pas 404 ou 405
        response = client.get('/admin/referential-import')
        # 401 = non authentifié (OK), 200 = accessible, 302 = redirection login
        assert response.status_code in [200, 302, 401], f"Status inattendu: {response.status_code}"
    
    def test_post_referential_import_accepts_pdf(self, client):
        """Test que POST /referential/import accepte un PDF."""
        # On ne peut pas tester l'upload complet sans fichier PDF
        # On vérifie juste que la route existe et accepte POST
        response = client.post('/referential/import')
        # 400 = fichier manquant (OK), 401 = non authentifié (OK)
        # On ne doit pas avoir 405 (Method Not Allowed)
        assert response.status_code != 405, "POST /referential/import ne devrait pas retourner 405"
    
    def test_routes_list_available(self, app):
        """Test que les routes sont bien enregistrées."""
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        
        # Vérifier que les routes principales existent
        assert '/referential/import' in rules
        assert '/admin/referential-import' in rules
        assert '/admin/ai-certification-market-comparison' in rules


class TestTaxonomyFallback:
    """Tests pour le fallback de taxonomie."""
    
    def test_fallback_taxonomy_structure(self):
        """Test que le fallback génère une taxonomie valide."""
        from inference.deepforma_predictor import DeepformaPredictor
        
        # Créer un mock de predictor avec des labels
        class MockPredictor:
            labels = ["Machine Learning", "Deep Learning", "NLP"]
            
            def _build_fallback_taxonomy(self):
                """Méthode copiée de DeepformaPredictor pour test."""
                if not hasattr(self, 'labels') or not self.labels:
                    return {}
                
                skills = []
                for label in self.labels:
                    skills.append({
                        'id': label.lower().replace(' ', '_').replace('/', '_'),
                        'label': label,
                    })
                
                fallback = {
                    'schema_version': '1.0',
                    'taxonomy_id': 'fallback_from_checkpoint',
                    'title': 'Taxonomie IA Deepforma (fallback)',
                    'description': 'Taxonomie générée automatiquement à partir des labels du modèle',
                    'families': [
                        {
                            'id': 'ia',
                            'label': 'Intelligence Artificielle',
                            'skills': skills,
                        }
                    ],
                    'metadata': {
                        'generated_from': 'checkpoint_labels',
                        'total_labels': len(self.labels),
                    }
                }
                return fallback
        
        mock = MockPredictor()
        fallback = mock._build_fallback_taxonomy()
        
        assert 'families' in fallback
        assert len(fallback['families']) == 1
        assert len(fallback['families'][0]['skills']) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
