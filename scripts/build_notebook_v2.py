from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "entrainement_camembert_competences_ia_v2.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md("""# Entraînement CamemBERT v2

Notebook d'entraînement en deux étapes :

1. classification binaire `IA / non-IA` ;
2. classification multi-étiquette des compétences IA uniquement sur les formations IA confirmées.

Le notebook consomme `data/processed/dataset_entrainement.csv` et s'appuie sur le pipeline de nettoyage généré par `scripts/clean_and_merge_datasets.py`.
"""),
    code("""from __future__ import annotations

from pathlib import Path
import sys
import json
import math

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    set_seed,
)
from torch import nn


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / 'data' / 'processed' / 'dataset_entrainement.csv').exists():
            return candidate
    return start


ROOT = find_project_root(Path.cwd().resolve())
sys.path.insert(0, str(ROOT))

from scripts.clean_and_merge_datasets import build_text_modele, clean_text, normalize_unicode, parse_multi_values

set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('ROOT =', ROOT)
print('DEVICE =', DEVICE)
"""),
    md("""## 1. Chargement des données

Le fichier `dataset_entrainement.csv` contient uniquement les exemples IA confirmés et non-IA confirmés.
"""),
    code("""DATA_PATH = ROOT / 'data' / 'processed' / 'dataset_entrainement.csv'
if not DATA_PATH.exists():
    raise FileNotFoundError(f'Fichier introuvable : {DATA_PATH}')

df = pd.read_csv(DATA_PATH)
print('Dimensions :', df.shape)
print('Répartition :')
print(df['statut_annotation'].value_counts(dropna=False).to_string())

display(df.head(3))
"""),
    md("""## 2. Préparation du split sans fuite

Les groupes reposent sur `formation_group_id` afin de garder ensemble les doublons exacts et les variantes proches.
"""),
    code("""def grouped_train_val_test_split(frame: pd.DataFrame, group_col: str = 'formation_group_id', seed: int = 42):
    if group_col not in frame.columns:
        raise KeyError(f'Colonne manquante : {group_col}')

    groups = frame[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(splitter.split(frame, groups=groups))
    train_df = frame.iloc[train_idx].copy()
    temp_df = frame.iloc[temp_idx].copy()

    temp_groups = temp_df[group_col].astype(str)
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_idx, test_idx = next(splitter2.split(temp_df, groups=temp_groups))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    return train_df, val_df, test_df


train_df, val_df, test_df = grouped_train_val_test_split(df)
for name, subset in [('train', train_df), ('val', val_df), ('test', test_df)]:
    print(f'{name}: lignes={len(subset)} groupes={subset["formation_group_id"].nunique()}')

assert set(train_df['formation_group_id']).isdisjoint(set(val_df['formation_group_id']))
assert set(train_df['formation_group_id']).isdisjoint(set(test_df['formation_group_id']))
assert set(val_df['formation_group_id']).isdisjoint(set(test_df['formation_group_id']))
"""),
    md("""## 3. Analyse du déséquilibre

On calcule les distributions de classes pour le modèle binaire et les occurrences de compétences pour le modèle multi-étiquette.
"""),
    code("""binary_counts = train_df['statut_annotation'].value_counts()
print(binary_counts.to_string())

ia_train = train_df[train_df['statut_annotation'] == 'ia_confirmee'].copy()
comp_counter = {}
for value in ia_train['competences_ia'].fillna(''):
    for comp in parse_multi_values(value):
        comp_counter[comp] = comp_counter.get(comp, 0) + 1

comp_stats = pd.DataFrame(sorted(comp_counter.items(), key=lambda x: (-x[1], x[0])), columns=['competence', 'occurrences'])
print('\nCompétences les plus fréquentes :')
display(comp_stats.head(20))
print('\nCompétences rares (< 20 occurrences) :')
display(comp_stats[comp_stats['occurrences'] < 20])
"""),
    md("""## 4. Modèle 1 - classification IA / non-IA

On affine `camembert-base` avec une tête de classification binaire et une perte pondérée pour gérer le déséquilibre.
"""),
    code("""MODEL_NAME = 'camembert-base'
TEXT_COL = 'texte_modele'
LABEL_COL = 'binary_label'
MAX_LENGTH = 256

label_map = {'non_ia_confirmee': 0, 'ia_confirmee': 1}
train_bin = train_df.copy()
val_bin = val_df.copy()
test_bin = test_df.copy()
for frame in [train_bin, val_bin, test_bin]:
    frame[LABEL_COL] = frame['statut_annotation'].map(label_map).astype(int)

classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_bin[LABEL_COL].to_numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print('Class weights :', class_weights.tolist())

tokenizer_bin = AutoTokenizer.from_pretrained(MODEL_NAME)
model_bin = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
"""),
    code("""def tokenize_binary(batch):
    return tokenizer_bin(batch[TEXT_COL], truncation=True, max_length=MAX_LENGTH)

train_bin_ds = Dataset.from_pandas(train_bin[[TEXT_COL, LABEL_COL]])
val_bin_ds = Dataset.from_pandas(val_bin[[TEXT_COL, LABEL_COL]])
test_bin_ds = Dataset.from_pandas(test_bin[[TEXT_COL, LABEL_COL]])

train_bin_ds = train_bin_ds.map(tokenize_binary, batched=True)
val_bin_ds = val_bin_ds.map(tokenize_binary, batched=True)
test_bin_ds = test_bin_ds.map(tokenize_binary, batched=True)

train_bin_ds = train_bin_ds.rename_column(LABEL_COL, 'labels')
val_bin_ds = val_bin_ds.rename_column(LABEL_COL, 'labels')
test_bin_ds = test_bin_ds.rename_column(LABEL_COL, 'labels')

cols_to_remove = [TEXT_COL]
train_bin_ds = train_bin_ds.remove_columns([col for col in cols_to_remove if col in train_bin_ds.column_names])
val_bin_ds = val_bin_ds.remove_columns([col for col in cols_to_remove if col in val_bin_ds.column_names])
test_bin_ds = test_bin_ds.remove_columns([col for col in cols_to_remove if col in test_bin_ds.column_names])

collator_bin = DataCollatorWithPadding(tokenizer_bin)

class WeightedBinaryTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

binary_args = TrainingArguments(
    output_dir=str(ROOT / 'models' / 'binary_ia_v2'),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    report_to='none',
    fp16=torch.cuda.is_available(),
)

binary_trainer = WeightedBinaryTrainer(
    model=model_bin,
    args=binary_args,
    train_dataset=train_bin_ds,
    eval_dataset=val_bin_ds,
    tokenizer=tokenizer_bin,
    data_collator=collator_bin,
    compute_metrics=compute_binary_metrics,
    class_weights=class_weights,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print('Entraînement modèle binaire...')
# binary_trainer.train()
"""),
    code("""def evaluate_binary(trainer, dataset, frame, title):
    metrics = trainer.predict(dataset)
    preds = np.argmax(metrics.predictions, axis=-1)
    labels = frame[LABEL_COL].to_numpy()
    cm = confusion_matrix(labels, preds)
    print(title)
    print('Accuracy :', accuracy_score(labels, preds))
    print('F1 :', f1_score(labels, preds, zero_division=0))
    print('Confusion matrix :\n', cm)
    return preds, cm

# Example after training:
# preds_val, cm_val = evaluate_binary(binary_trainer, val_bin_ds, val_bin, 'Validation binaire')
"""),
    md("""## 5. Modèle 2 - multi-étiquette des compétences IA

Le second modèle est entraîné uniquement sur les formations IA confirmées. Les cibles sont binarisées avec `MultiLabelBinarizer`.
"""),
    code("""ia_train = train_df[train_df['statut_annotation'] == 'ia_confirmee'].copy()
ia_val = val_df[val_df['statut_annotation'] == 'ia_confirmee'].copy()
ia_test = test_df[test_df['statut_annotation'] == 'ia_confirmee'].copy()

if ia_train.empty:
    raise ValueError('Aucune formation IA confirmée dans le train split.')

train_labels = [parse_multi_values(v) for v in ia_train['competences_ia'].fillna('')]
val_labels = [parse_multi_values(v) for v in ia_val['competences_ia'].fillna('')]
test_labels = [parse_multi_values(v) for v in ia_test['competences_ia'].fillna('')]

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_labels)
Y_val = mlb.transform(val_labels) if len(val_labels) else np.zeros((0, len(mlb.classes_)), dtype=int)
Y_test = mlb.transform(test_labels) if len(test_labels) else np.zeros((0, len(mlb.classes_)), dtype=int)
print('Nombre de compétences :', len(mlb.classes_))
print('Compétences :', list(mlb.classes_))

model_ml = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(mlb.classes_),
    problem_type='multi_label_classification',
)
tokenizer_ml = AutoTokenizer.from_pretrained(MODEL_NAME)
"""),
    code("""def tokenize_multilabel(batch):
    return tokenizer_ml(batch[TEXT_COL], truncation=True, max_length=MAX_LENGTH)

train_ml = Dataset.from_pandas(pd.DataFrame({TEXT_COL: ia_train[TEXT_COL].tolist(), 'labels': Y_train.tolist()}))
val_ml = Dataset.from_pandas(pd.DataFrame({TEXT_COL: ia_val[TEXT_COL].tolist(), 'labels': Y_val.tolist()})) if len(ia_val) else None
test_ml = Dataset.from_pandas(pd.DataFrame({TEXT_COL: ia_test[TEXT_COL].tolist(), 'labels': Y_test.tolist()})) if len(ia_test) else None

train_ml = train_ml.map(tokenize_multilabel, batched=True)
if val_ml is not None:
    val_ml = val_ml.map(tokenize_multilabel, batched=True)
if test_ml is not None:
    test_ml = test_ml.map(tokenize_multilabel, batched=True)

train_ml = train_ml.remove_columns([TEXT_COL])
if val_ml is not None:
    val_ml = val_ml.remove_columns([TEXT_COL])
if test_ml is not None:
    test_ml = test_ml.remove_columns([TEXT_COL])

train_label_counts = Y_train.sum(axis=0)
pos_weight = torch.tensor((len(Y_train) - train_label_counts) / np.clip(train_label_counts, 1, None), dtype=torch.float32)
print('Poids positifs :', pos_weight.tolist())

class WeightedMultiLabelTrainer(Trainer):
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels').float()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_multilabel_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))
    preds = (probabilities >= 0.5).astype(int)
    return {
        'f1_micro': f1_score(labels, preds, average='micro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'precision_micro': precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)[0],
        'recall_micro': precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)[1],
    }

multi_args = TrainingArguments(
    output_dir=str(ROOT / 'models' / 'multilabel_competences_v2'),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_micro',
    greater_is_better=True,
    report_to='none',
    fp16=torch.cuda.is_available(),
)

multi_trainer = WeightedMultiLabelTrainer(
    model=model_ml,
    args=multi_args,
    train_dataset=train_ml,
    eval_dataset=val_ml,
    tokenizer=tokenizer_ml,
    data_collator=DataCollatorWithPadding(tokenizer_ml),
    compute_metrics=compute_multilabel_metrics,
    pos_weight=pos_weight,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print('Entraînement modèle multi-étiquette...')
# multi_trainer.train()
"""),
    code("""def evaluate_multilabel(trainer, dataset, frame, mlb, threshold=0.5, title=''):
    if dataset is None or len(frame) == 0:
        print(title, 'aucun exemple')
        return None
    out = trainer.predict(dataset)
    probabilities = 1 / (1 + np.exp(-out.predictions))
    preds = (probabilities >= threshold).astype(int)
    labels = mlb.transform([parse_multi_values(v) for v in frame['competences_ia'].fillna('')])
    print(title)
    print('F1 micro :', f1_score(labels, preds, average='micro', zero_division=0))
    print('F1 macro :', f1_score(labels, preds, average='macro', zero_division=0))
    print('Précision micro :', precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)[0])
    print('Rappel micro :', precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)[1])
    report = classification_report(labels, preds, target_names=mlb.classes_, zero_division=0)
    print(report)
    return probabilities, preds

# Example after training:
# evaluate_multilabel(multi_trainer, test_ml, ia_test, mlb, threshold=0.5, title='Test multi-étiquette')
"""),
    md("""## 6. Fonction de prédiction

La prédiction s'effectue en cascade :

- le modèle binaire calcule `probabilite_ia` ;
- si la probabilité est sous le seuil, le modèle de compétences n'est pas sollicité ;
- sinon le second modèle renvoie les compétences au-dessus du seuil.
"""),
    code("""IA_THRESHOLD = 0.50
COMPETENCE_THRESHOLD = 0.35

def build_model_input(formation_data) -> pd.Series:
    if isinstance(formation_data, pd.Series):
        row = formation_data.copy()
    elif isinstance(formation_data, dict):
        row = pd.Series(formation_data)
    else:
        raise TypeError('formation_data doit être un dict ou une Series pandas')

    if 'texte_modele' not in row or not clean_text(row.get('texte_modele', '')):
        base = {
            'intitule': row.get('intitule', row.get('Intitulé', row.get('Intitulé de la formation', ''))),
            'description': row.get('description', ''),
            'objectifs': row.get('objectifs', ''),
            'programme': row.get('programme', ''),
            'public_cible': row.get('public_cible', row.get('Public cible', '')),
            'prerequis': row.get('prerequis', ''),
            'niveau': row.get('niveau', row.get('Niveau', '')),
            'modalite': row.get('modalite', row.get('Modalité', '')),
            'duree': row.get('duree', row.get('Durée', '')),
            'certification': row.get('certification', row.get('Type de certification', '')),
            'codes_rome': row.get('codes_rome', row.get('Codes ROME', '')),
            'organisme': row.get('organisme', row.get('Organisme de formation', '')),
        }
        row = pd.Series(base)
        row['texte_modele'] = build_text_modele(row)
    return row


def predict_formation(formation_data):
    row = build_model_input(formation_data)
    text = clean_text(row.get('texte_modele', ''))
    binary_inputs = tokenizer_bin(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    model_bin.eval()
    model_bin.to(DEVICE)
    with torch.no_grad():
        logits = model_bin(**binary_inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    probabilite_ia = float(probs[1])
    result = {
        'est_lie_ia': probabilite_ia >= IA_THRESHOLD,
        'probabilite_ia': probabilite_ia,
        'competences': [],
    }
    if probabilite_ia < IA_THRESHOLD:
        return result

    model_ml.eval()
    model_ml.to(DEVICE)
    multi_inputs = tokenizer_ml(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits = model_ml(**multi_inputs).logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    ranked = sorted(
        [
            {'nom': name, 'probabilite': float(prob)}
            for name, prob in zip(mlb.classes_, probabilities)
            if prob >= COMPETENCE_THRESHOLD
        ],
        key=lambda item: item['probabilite'],
        reverse=True,
    )
    result['competences'] = ranked
    return result

# Exemple d'usage :
# predict_formation({'intitule': 'Formation IA générative pour les RH', 'description': '...', 'public_cible': '...'})
"""),
    md("""## 7. Évaluation finale et sauvegarde

Une fois l'entraînement terminé, évaluez sur le test split, sauvegardez les modèles et documentez les seuils retenus.
"""),
    code("""# Exemple :
# binary_metrics_test = binary_trainer.evaluate(test_bin_ds)
# evaluate_binary(binary_trainer, test_bin_ds, test_bin, 'Test binaire')
# evaluate_multilabel(multi_trainer, test_ml, ia_test, mlb, threshold=COMPETENCE_THRESHOLD, title='Test multi-étiquette')
# binary_trainer.save_model(str(ROOT / 'models' / 'binary_ia_v2' / 'final'))
# multi_trainer.save_model(str(ROOT / 'models' / 'multilabel_competences_v2' / 'final'))
"""),
]

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.11',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Wrote {NOTEBOOK_PATH}')
