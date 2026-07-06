# Architecture

DeepForma is organized around a single business chain:

`profile or training` -> `skill extraction` -> `normalization` -> `market comparison` -> `gap analysis` -> `certification / training recommendation` -> `explanation`

## Current Structure

- `src/domain/` shared business models and exceptions.
- `src/services/` scoring, recommendation, market comparison, and analysis facades.
- `src/data_sources/` connectors to France Travail, France Compétences, and related official datasets.
- `src/deepforma/` CPF-specific embedding, training, and recommendation code.
- `src/referential_import/` and `src/training_import/` PDF import pipelines.
- `src/web_app.py` Flask entry point, kept thin and progressively reduced.
- `scripts/` CLI entry points for import, normalization, training, and evaluation.

## Design Rules

- Keep one implementation per business concept.
- Keep web routes thin.
- Keep data access separate from scoring and normalization.
- Keep outputs explicable and traceable.
- Prefer dataclasses for simple business objects.

## What Is Being Consolidated

- skill normalization into a shared service;
- market comparison into a single service facade;
- analysis result assembly into a dedicated builder;
- configuration into a central settings object;
- domain models into `src/domain/`.

## Legacy Code

Legacy modules are still present for compatibility and test coverage. They are kept until the new service layer fully covers their use cases.
