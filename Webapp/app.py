from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__, static_folder='.', static_url_path='')

MODEL_FEATURES = ['G1', 'G2', 'absences']


def build_model():
    csv_path = Path(__file__).resolve().parent.parent / 'CodeBase' / 'data' / 'student-mat.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path, sep=';')
    df['pass_target'] = (df['G3'] >= 10).astype(int)
    X = df[MODEL_FEATURES]
    y = df['pass_target']
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(solver='liblinear', random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline


MODEL = build_model()


def transform_input(data):
    attendance = float(data.get('attendance', 0))
    presentation = float(data.get('presentation', 0))
    assignment = float(data.get('assignment', 0))
    quiz = float(data.get('quiz', 0))
    midterm = float(data.get('midterm', 0))

    g1 = round((presentation * 0.3 + assignment * 0.3 + quiz * 0.2 + midterm * 0.2) * 0.2)
    g2 = round((presentation * 0.2 + assignment * 0.2 + quiz * 0.2 + midterm * 0.4) * 0.2)
    absences = max(0, min(30, round((100.0 - attendance) * 0.3)))
    return [g1, g2, absences]


@app.route('/')
def index():
    return send_from_directory(Path(__file__).resolve().parent, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    try:
        features = transform_input(payload)
        probability = float(MODEL.predict_proba([features])[0][1])
        score = int(round(probability * 100))
        status = 'Pass' if probability >= 0.55 else 'At Risk'
        advice = []
        values = {
            'attendance': float(payload.get('attendance', 0)),
            'presentation': float(payload.get('presentation', 0)),
            'assignment': float(payload.get('assignment', 0)),
            'quiz': float(payload.get('quiz', 0)),
            'midterm': float(payload.get('midterm', 0))
        }
        low_scores = sorted(values.items(), key=lambda item: item[1])[:2]
        advice_map = {
            'attendance': 'Improve attendance to reduce missed lessons and stay on track.',
            'presentation': 'Practice presentations and clarify concepts to boost understanding.',
            'assignment': 'Complete assignments consistently to build knowledge and confidence.',
            'quiz': 'Use weekly quizzes to review and identify weak areas early.',
            'midterm': 'Review midterm material thoroughly and seek help on difficult topics.'
        }
        advice = [advice_map[key] for key, _ in low_scores]
        breakdown = {key: int(round(value)) for key, value in values.items()}

        return jsonify({
            'score': score,
            'probability': probability,
            'status': status,
            'message': 'Real backend prediction from project data.',
            'advice': advice,
            'breakdown': breakdown
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
