"""Tests de non-régression pour ReferentialImportStore."""
import tempfile
from pathlib import Path

import pytest

from referential_import.store import ReferentialImportStore


def test_store_initialization_no_recursion():
    """Test que l'initialisation du store ne cause pas de récursion infinie."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        
        # Première instanciation - ne doit pas lever RecursionError
        store1 = ReferentialImportStore(db_path)
        assert db_path.exists(), "La base de données doit être créée"
        
        # Deuxième instanciation - ne doit pas lever RecursionError
        store2 = ReferentialImportStore(db_path)
        assert db_path.exists(), "La base de données doit toujours exister"


def test_store_initialization_creates_parent_directory():
    """Test que le store crée le dossier parent s'il n'existe pas."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Chemin avec sous-dossier inexistant
        db_path = Path(tmpdir) / "subdir" / "nested" / "test.sqlite3"
        
        # Le dossier parent n'existe pas encore
        assert not db_path.parent.exists()
        
        # L'instanciation doit créer le dossier et la base
        store = ReferentialImportStore(db_path)
        
        assert db_path.parent.exists(), "Le dossier parent doit être créé"
        assert db_path.exists(), "La base de données doit être créée"


def test_store_tables_exist():
    """Test que toutes les tables requises sont créées."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        
        # Vérifier que les tables existent
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        expected_tables = {
            'imports', 'blocks', 'activities', 
            'competencies', 'criteria', 'derived_skills', 'issues'
        }
        
        conn.close()
        
        assert expected_tables.issubset(tables), f"Tables manquantes: {expected_tables - tables}"


def test_store_multiple_instances_same_db():
    """Test que plusieurs instances peuvent accéder à la même base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        
        # Créer plusieurs instances
        store1 = ReferentialImportStore(db_path)
        store2 = ReferentialImportStore(db_path)
        store3 = ReferentialImportStore(db_path)
        
        # Toutes doivent fonctionner sans erreur
        result1 = store1.list_imports()
        result2 = store2.list_imports()
        result3 = store3.list_imports()
        
        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert isinstance(result3, list)


def test_store_has_document_method():
    """Test que la méthode has_document fonctionne après initialisation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        
        # Tester avec un hash qui n'existe pas
        result = store.has_document("fake_hash", "v1.0")
        assert result is False


def test_store_list_imports_empty():
    """Test que list_imports retourne une liste vide pour une nouvelle base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        
        imports = store.list_imports()
        assert imports == []


def test_store_initialized_flag():
    """Test que le flag _initialized est correctement géré."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        
        # Le flag doit être True après initialisation
        assert store._initialized is True
        
        # Appeler _init_db() à nouveau ne doit pas causer de problème
        store._init_db()
        assert store._initialized is True
