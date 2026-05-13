import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dermatology_database_1.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

features = [
    "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
    "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
    "knee_and_elbow_involvement", "scalp_involvement", "family_history",
    "melanin_incontinence", "eosinophils_infiltrate", "PNL_infiltrate",
    "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
    "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis",
    "disappearance_granular_layer", "vacuolisation_damage_basal_layer",
    "spongiosis", "saw_tooth_appearance_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate", "age"
]

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}\n"
            "Placez votre dataset original sous le nom data/dermatology_database_1.csv"
        )

    df = pd.read_csv(DATA_PATH)
    df = df.replace("?", pd.NA)

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["class"] = pd.to_numeric(df["class"], errors="coerce").astype(int)
    return df

def main():
    df = load_data()
    X = df[features]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("model", RandomForestClassifier(n_estimators=250, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(pipeline, MODEL_DIR / "model_dermatologie.pkl")
    joblib.dump(features, MODEL_DIR / "features.pkl")
    print("Modele sauvegarde dans le dossier model/")

if __name__ == "__main__":
    main()
