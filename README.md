# DeepSkyn - Render Ready

## Lancer localement

```bash
pip install -r requirements.txt
python app.py
```

Puis ouvrir :

```txt
http://127.0.0.1:5000
```

## Déployer sur Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
gunicorn app:app
```

Important : le projet contient `.python-version` avec Python 3.11.9 pour éviter les erreurs scikit-learn/numpy sur Render.
