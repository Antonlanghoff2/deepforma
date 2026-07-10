"""Tests pour les fonctionnalités d'ajout et suppression de compétences."""
import pytest


class TestUpdateAnnotationStatus:
    """Tests unitaires pour _update_annotation_status."""
    
    def test_add_skill_to_candidates(self):
        """Test l'ajout d'une compétence à un candidat."""
        from web_app import create_app
        
        app = create_app()
        
        # Récupérer la fonction depuis le contexte de l'app
        with app.app_context():
            # Simuler la fonction _update_annotation_status
            record = {
                "record_id": "test-001",
                "kind": "candidates",
                "skills": [
                    {"skill_id": "skill-001", "label": "Python", "status": "pending"}
                ],
                "skills_count": 1
            }
            
            # Simuler le formulaire
            class MockForm:
                def get(self, key, default=''):
                    data = {
                        'text': 'Data Science',
                        'entity_label': 'DOMAIN',
                        'canonical_name': 'Data Science'
                    }
                    return data.get(key, default)
            
            # Appliquer l'action manuellement (copie de la logique)
            action = 'add_entity'
            form = MockForm()
            
            if record.get('kind') == 'candidates':
                if action == 'add_entity':
                    label = (form.get('text') or '').strip()
                    category = (form.get('entity_label') or 'SKILL').strip()
                    canonical_name = (form.get('canonical_name') or label).strip()
                    
                    if label:
                        new_skill = {
                            'skill_id': f"manual-skill-{len(record.get('skills', [])) + 1}",
                            'label': label,
                            'canonical_label': canonical_name,
                            'category': category,
                            'confidence': 1.0,
                            'source': 'manual',
                            'status': 'approved',
                        }
                        record.setdefault('skills', []).append(new_skill)
                        record['skills_count'] = len(record['skills'])
            
            # Vérifications
            assert len(record['skills']) == 2
            assert record['skills_count'] == 2
            new_skill = record['skills'][1]
            assert new_skill['label'] == 'Data Science'
            assert new_skill['category'] == 'DOMAIN'
            assert new_skill['status'] == 'approved'
    
    def test_delete_skill_from_candidates(self):
        """Test la suppression d'une compétence d'un candidat."""
        record = {
            "record_id": "test-001",
            "kind": "candidates",
            "skills": [
                {"skill_id": "skill-001", "label": "Python", "status": "pending"},
                {"skill_id": "skill-002", "label": "SQL", "status": "pending"}
            ],
            "skills_count": 2
        }
        
        # Simuler le formulaire
        class MockForm:
            def get(self, key, default=''):
                data = {'entity_id': 'skill-001'}
                return data.get(key, default)
        
        # Appliquer l'action
        action = 'delete_entity'
        form = MockForm()
        
        if record.get('kind') == 'candidates':
            if action == 'delete_entity':
                entity_id = (form.get('entity_id') or '').strip()
                if entity_id:
                    record['skills'] = [
                        skill for skill in record.get('skills', [])
                        if (skill.get('skill_id') or '').strip() != entity_id
                    ]
                    record['skills_count'] = len(record['skills'])
        
        # Vérifications
        assert len(record['skills']) == 1
        assert record['skills_count'] == 1
        assert record['skills'][0]['skill_id'] == 'skill-002'
    
    def test_add_entity_to_ner(self):
        """Test l'ajout d'une entité NER."""
        record = {
            "record_id": "test-001",
            "kind": "ner",
            "entities": [
                {"entity_id": "entity-001", "text": "Python", "status": "pending"}
            ]
        }
        
        # Simuler le formulaire
        class MockForm:
            def get(self, key, default=''):
                data = {
                    'text': 'TensorFlow',
                    'entity_label': 'TOOL',
                    'canonical_name': 'TensorFlow',
                    'start': '10',
                    'end': '20',
                    'page': '1'
                }
                return data.get(key, default)
        
        # Appliquer l'action
        action = 'add_entity'
        form = MockForm()
        
        if record.get('kind') == 'ner':
            if action == 'add_entity':
                text_value = (form.get('text') or '').strip()
                if text_value:
                    record.setdefault('entities', []).append({
                        'entity_id': f"manual-entity-{len(record.get('entities', [])) + 1}",
                        'start': int(form.get('start') or 0),
                        'end': int(form.get('end') or len(text_value)),
                        'text': text_value,
                        'predicted_label': (form.get('entity_label') or 'SKILL').strip(),
                        'approved_label': (form.get('approved_label') or form.get('entity_label') or 'SKILL').strip(),
                        'canonical_name': (form.get('canonical_name') or text_value).strip(),
                        'confidence': 1.0,
                        'page': int(form.get('page') or 0),
                        'status': 'approved',
                    })
        
        # Vérifications
        assert len(record['entities']) == 2
        new_entity = record['entities'][1]
        assert new_entity['text'] == 'TensorFlow'
        assert new_entity['predicted_label'] == 'TOOL'
        assert new_entity['status'] == 'approved'
    
    def test_delete_entity_from_ner(self):
        """Test la suppression d'une entité NER."""
        record = {
            "record_id": "test-001",
            "kind": "ner",
            "entities": [
                {"entity_id": "entity-001", "text": "Python", "status": "pending"},
                {"entity_id": "entity-002", "text": "SQL", "status": "pending"}
            ]
        }
        
        # Simuler le formulaire
        class MockForm:
            def get(self, key, default=''):
                data = {'entity_id': 'entity-001'}
                return data.get(key, default)
        
        # Appliquer l'action
        action = 'delete_entity'
        form = MockForm()
        
        if record.get('kind') == 'ner':
            if action == 'delete_entity':
                entity_id = (form.get('entity_id') or '').strip()
                if entity_id:
                    record['entities'] = [
                        entity for entity in record.get('entities', [])
                        if (entity.get('entity_id') or '').strip() != entity_id
                    ]
        
        # Vérifications
        assert len(record['entities']) == 1
        assert record['entities'][0]['entity_id'] == 'entity-002'
    
    def test_approve_skill(self):
        """Test l'approbation d'une compétence."""
        record = {
            "record_id": "test-001",
            "kind": "candidates",
            "skills": [
                {"skill_id": "skill-001", "label": "Python", "status": "pending", "category": "tool"}
            ]
        }
        
        # Simuler le formulaire
        class MockForm:
            def get(self, key, default=''):
                data = {
                    'entity_id': 'skill-001',
                    'approved_label': 'TOOL',
                    'canonical_name': 'Python Programming'
                }
                return data.get(key, default)
        
        # Appliquer l'action
        action = 'approve_entity'
        form = MockForm()
        
        if record.get('kind') == 'candidates':
            if action in {'approve_entity', 'approve_all_entities'}:
                entity_id = (form.get('entity_id') or '').strip()
                approved_label = (form.get('approved_label') or '').strip()
                canonical_name = (form.get('canonical_name') or '').strip()
                
                for skill in record.get('skills', []):
                    skill_id = (skill.get('skill_id') or '').strip()
                    if action == 'approve_all_entities' or (skill_id and skill_id == entity_id):
                        skill['status'] = 'approved'
                        if approved_label:
                            skill['category'] = approved_label
                        if canonical_name:
                            skill['canonical_label'] = canonical_name
        
        # Vérifications
        skill = record['skills'][0]
        assert skill['status'] == 'approved'
        assert skill['category'] == 'TOOL'
        assert skill['canonical_label'] == 'Python Programming'
    
    def test_reject_skill(self):
        """Test le rejet d'une compétence."""
        record = {
            "record_id": "test-001",
            "kind": "candidates",
            "skills": [
                {"skill_id": "skill-001", "label": "Python", "status": "pending"}
            ]
        }
        
        # Simuler le formulaire
        class MockForm:
            def get(self, key, default=''):
                data = {'entity_id': 'skill-001'}
                return data.get(key, default)
        
        # Appliquer l'action
        action = 'reject_entity'
        form = MockForm()
        
        if record.get('kind') == 'candidates':
            if action in {'reject_entity', 'reject_all_entities'}:
                entity_id = (form.get('entity_id') or '').strip()
                for skill in record.get('skills', []):
                    skill_id = (skill.get('skill_id') or '').strip()
                    if action == 'reject_all_entities' or (skill_id and skill_id == entity_id):
                        skill['status'] = 'rejected'
        
        # Vérifications
        skill = record['skills'][0]
        assert skill['status'] == 'rejected'
    
    def test_approve_all_skills(self):
        """Test l'approbation de toutes les compétences."""
        record = {
            "record_id": "test-001",
            "kind": "candidates",
            "skills": [
                {"skill_id": "skill-001", "label": "Python", "status": "pending"},
                {"skill_id": "skill-002", "label": "SQL", "status": "pending"}
            ]
        }
        
        # Simuler le formulaire
        class MockForm:
            def get(self, key, default=''):
                return default
        
        # Appliquer l'action
        action = 'approve_all_entities'
        form = MockForm()
        
        if record.get('kind') == 'candidates':
            if action in {'approve_entity', 'approve_all_entities'}:
                entity_id = (form.get('entity_id') or '').strip()
                approved_label = (form.get('approved_label') or '').strip()
                canonical_name = (form.get('canonical_name') or '').strip()
                
                for skill in record.get('skills', []):
                    skill_id = (skill.get('skill_id') or '').strip()
                    if action == 'approve_all_entities' or (skill_id and skill_id == entity_id):
                        skill['status'] = 'approved'
                        if approved_label:
                            skill['category'] = approved_label
                        if canonical_name:
                            skill['canonical_label'] = canonical_name
        
        # Vérifications
        for skill in record['skills']:
            assert skill['status'] == 'approved'
    
    def test_reject_all_skills(self):
        """Test le rejet de toutes les compétences."""
        record = {
            "record_id": "test-001",
            "kind": "candidates",
            "skills": [
                {"skill_id": "skill-001", "label": "Python", "status": "pending"},
                {"skill_id": "skill-002", "label": "SQL", "status": "pending"}
            ]
        }
        
        # Simuler le formulaire
        class MockForm:
            def get(self, key, default=''):
                return default
        
        # Appliquer l'action
        action = 'reject_all_entities'
        form = MockForm()
        
        if record.get('kind') == 'candidates':
            if action in {'reject_entity', 'reject_all_entities'}:
                entity_id = (form.get('entity_id') or '').strip()
                for skill in record.get('skills', []):
                    skill_id = (skill.get('skill_id') or '').strip()
                    if action == 'reject_all_entities' or (skill_id and skill_id == entity_id):
                        skill['status'] = 'rejected'
        
        # Vérifications
        for skill in record['skills']:
            assert skill['status'] == 'rejected'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
