#!/usr/bin/env python3
"""
Liste toutes les routes Flask de l'application Deepforma.
"""
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from web_app import create_app


def list_routes():
    """Liste toutes les routes avec leurs méthodes."""
    app = create_app()
    
    print("=" * 80)
    print("ROUTES FLASK - DEEPFORMA")
    print("=" * 80)
    print()
    
    # Grouper par préfixe
    routes_by_prefix = {}
    for rule in app.url_map.iter_rules():
        # Ignorer les routes statiques
        if rule.endpoint == 'static':
            continue
        
        # Extraire le préfixe
        parts = rule.rule.split('/')
        prefix = '/' + parts[1] if len(parts) > 1 else '/'
        
        if prefix not in routes_by_prefix:
            routes_by_prefix[prefix] = []
        
        # Formater les méthodes
        methods = ', '.join(sorted([m for m in rule.methods if m not in ['HEAD', 'OPTIONS']]))
        
        routes_by_prefix[prefix].append({
            'rule': rule.rule,
            'methods': methods,
            'endpoint': rule.endpoint
        })
    
    # Afficher par groupe
    for prefix in sorted(routes_by_prefix.keys()):
        print(f"\n{prefix}")
        print("-" * 80)
        for route in sorted(routes_by_prefix[prefix], key=lambda x: x['rule']):
            methods = route['methods'].ljust(12)
            print(f"  {methods} {route['rule']}")
            print(f"  {' ' * 12} → {route['endpoint']}")
    
    print()
    print("=" * 80)
    print(f"Total: {sum(len(routes) for routes in routes_by_prefix.values())} routes")
    print("=" * 80)


if __name__ == "__main__":
    list_routes()
