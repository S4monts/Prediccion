import os
from flask import current_app, render_template, request, redirect, url_for, flash
from numpy import integer
from app import app
from app.forms import IndividualPredictionForm, BatchPredictionForm
from app.models.logistic_regression import LogisticRegressionModel
from app.models.model_utils import retrain_model
from app.models.neural_network import NeuralNetworkModel
from app.models.svm_model import SVMModel
from app.models.fuzzy_map import FuzzyMapModel
from app.utils.data_validation import validate_batch_file
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/individual', methods=['GET', 'POST'])
def individual_prediction():
    form = IndividualPredictionForm()
    
    if form.validate_on_submit():
        try:
            # Mapeo de campos del formulario a las columnas del modelo
            input_data = [
                form.age.data,        # C1
                form.bmi.data,        # C2
                form.gestational_age.data,  # C3
                form.gravidity.data,  # C4
                form.parity.data,     # C5
                form.initial_symptoms.data,  # C6
                0.0,  # C7 (Gestational age of IOS onset) - No en formulario simplificado
                0.0,  # C8
                0.0,  # C9
                0.0,  # C10
                0.0,  # C11
                0.0,  # C12
                0.0,  # C13
                0.0,  # C14
                0,    # C15 (Expectant treatment)
                0,    # C16 (Anti-hypertensive therapy)
                0,    # C17 (Past history)
                form.max_systolic.data,  # C18
                form.max_diastolic.data,  # C19
                0,    # C20 (Reasons for delivery)
                0,    # C21 (Mode of delivery)
                form.bnp.data,       # C22
                form.creatinine.data,  # C23
                0.0,  # C24 (Uric acid)
                form.proteinuria.data,  # C25
                0.0,  # C26 (Total protein)
                0.0,  # C27 (Albumin)
                0.0,  # C28 (ALT)
                0.0,  # C29 (AST)
                form.platelet.data   # C30
            ]

            # Debug: Verifica los datos
            print("Datos enviados:", input_data)
            
            # Selección del modelo
            model = {
                'logistic': LogisticRegressionModel(),
                'neural': NeuralNetworkModel(),
                'svm': SVMModel(),
                'fuzzy': FuzzyMapModel()
            }[form.model_choice.data]
            
            # Predicción
            prediction, probabilities = model.predict(input_data)
            print(f"Predicción obtenida: {prediction}, Probabilidades: {probabilities}")
            
            return render_template('results.html',
                                results={
                                    'prediction': prediction,
                                    'probability': max(probabilities) * 100,
                                    'model_name': dict(form.model_choice.choices).get(form.model_choice.data)
                                },
                                form=form,
                                input_data=input_data)
            
        except Exception as e:
            print(f"Error en la predicción: {str(e)}")
            flash(f"Error al procesar la predicción: {str(e)}", 'danger') 
            return redirect(url_for('individual_prediction'))
    
    # Debug: Si el formulario no es válido
    if request.method == 'POST':
        print("Errores del formulario:", form.errors)
        flash("Por favor corrige los errores en el formulario", 'warning')
    
    return render_template('individual_prediction.html', form=form)

@app.route('/batch', methods=['GET', 'POST'])
def batch_prediction():
    form = BatchPredictionForm()
    
    if form.validate_on_submit():
        try:
            # Cargar el archivo subido
            file = request.files['file']
            df_uploaded = pd.read_excel(file) if file.filename.endswith('.xlsx') else pd.read_csv(file)
            
            # Cargar el dataset original para comparación
            dataset_path = os.path.join(current_app.root_path,'..', 'data', 'FGR_dataset (1).xlsx')
            df_original = pd.read_excel(dataset_path)
            
            # Procesar predicciones
            model = {
                'logistic': LogisticRegressionModel(),
                'neural': NeuralNetworkModel(),
                'svm': SVMModel(),
                'fuzzy': FuzzyMapModel()
            }[form.model_choice.data]
            
            results = []
            matched_records = []
            
            for idx, row in df_uploaded.iterrows():
                # Convertir la fila a lista (solo características, sin la columna objetivo)
                features = row.values[:30].tolist()
                
                # Hacer predicción
                prediction, probabilities = model.predict(features)
                
                # Buscar coincidencias en el dataset original
                match = df_original[(df_original.iloc[:, :30] == features).all(axis=1)]
                actual_value = match.iloc[0, 30] if not match.empty else None
                
                results.append({
                    'row': idx + 1,
                    'prediction': prediction,
                    'probability': max(probabilities) * 100,
                    'actual': actual_value,
                    'features': features,
                    'is_correct': actual_value is not None and prediction == actual_value
                })
                
                if actual_value is not None:
                    matched_records.append({
                        'original_row': match.index[0] + 1,
                        'uploaded_row': idx + 1
                    })
            
            # Calcular métricas
            accuracy = sum(1 for r in results if r['is_correct']) / len(results) if results else 0
            
            return render_template('batch_results.html',
                                results=results,
                                accuracy=accuracy,
                                matched_records=matched_records,
                                model_name=dict(form.model_choice.choices).get(form.model_choice.data))
        
        except Exception as e:
            flash(f"Error al procesar el archivo: {str(e)}", 'danger')
            return redirect(url_for('batch_prediction'))
    
    return render_template('batch_prediction.html', form=form)

@app.route('/retrain', methods=['POST'])
def retrain():
    model_type = request.form.get('model_type')
    train_size = float(request.form.get('train_size', 80)) / 100
    random_state = int(request.form.get('random_state', 42))
    
    # Llamar a la función de reentrenamiento con todos los parámetros
    success, message, metrics = retrain_model(
        model_type=model_type,
        train_size=train_size,
        random_state=random_state
    )
    
    flash_category = 'success' if success else 'danger'
    flash(message, flash_category)
    
    # Mostrar métricas adicionales si el entrenamiento fue exitoso
    if success and metrics:
        if 'accuracy' in metrics:
            flash(f"Precisión del modelo: {metrics['accuracy']:.2%}", 'info')
        if 'train_samples' in metrics:
            flash(f"Muestras de entrenamiento: {metrics['train_samples']}", 'info')
        if 'last_retrained' in metrics:
            flash(f"Último reentrenamiento: {metrics['last_retrained']}", 'info')
    
    return redirect(request.referrer or url_for('index'))